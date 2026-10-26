"""Rollout planner: choose each bot's tour by playing the next rounds forward.

The greedy auction (planner.py) scores a tour by points per round on its own,
so it cannot see that a choice now decides where the bot ends up and what is
left to fetch. On the stores in ROLLOUT, each bot's top `candidates` tours
(the auction's pick, then the next best with the earlier picks banned) are
each played forward `horizon` rounds with the
greedy policy on the exact engine, the bot held to that tour until it is done,
against `samples` sampled futures (trip durations and orders, uniform as
measured, the same samples for every candidate). The tour with the best mean
score at the horizon, plus 1 per carried item an open order still needs, wins.
Bots are decided in ID order; later bots roll out with earlier choices held.
A chosen tour is kept until it is done or impossible, so rollouts run only
when a bot picks its next tour (and never past ROLLOUT_BUDGET_S per round;
the round deadline is 2 s).
"""

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np

from grocery_ai.control.controller import Controller
from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.plan.features import tour_features
from grocery_ai.plan.planner import Planner
from grocery_ai.plan.tours import Tour
from grocery_ai.sim.engine import Engine
from grocery_ai.sim.selfplay import UniformOracle

# store -> (candidates, horizon, samples); stores left out play the plain auction
# Easy: samples 4 -> 12 +2.3 (W/L 11/2); horizon 80 -3.9; 8 candidates -0.8. Medium on: sim +14.9 (14/2)
ROLLOUT = {"easy": (4, 40, 12), "medium": (4, 40, 12)}
ROLLOUT_BUDGET_S = 1.0
# store -> (candidates, model file in scorer/): pick among the auction's top tours with a model
# fitted to rollout decisions (tools/collect.py, tools/fit_scorer.py) instead of rolling out
SCORER: dict[str, tuple[int, str]] = {}
STORE_BY_BOTS = {1: "easy", 3: "medium", 5: "hard", 10: "expert", 20: "nightmare"}


class RolloutPlanner:
    def __init__(self, smap: StaticMap):
        self.smap = smap
        self.base = Planner(smap)
        self.tours = self.base.tours
        self.types = sorted(smap.access)
        self.chosen: dict[int, tuple] = {}   # bot id -> Tour.key it was given by a rollout
        self.model: dict | None = None
        self.log: list[dict] | None = None   # set to a list to record every decision (tools/collect.py)

    @property
    def trace(self) -> list[dict]:
        return self.base.trace

    def plan(self, world: World) -> dict[int, Tour | None]:
        store = STORE_BY_BOTS.get(len(world.bots))
        if store in SCORER:
            return self._plan_scored(world, *SCORER[store])
        if store not in ROLLOUT:
            return self.base.plan(world)
        n_candidates, horizon, samples = ROLLOUT[store]
        started = time.perf_counter()
        force: dict[int, tuple] = {}
        for bot in sorted(world.bots, key=lambda b: b.id):
            key = self.chosen.get(bot.id)
            if key is not None:
                kept = self.base.plan(world, force={**force, bot.id: key}, commit=False)[bot.id]
                if kept is not None and kept.key == key:
                    force[bot.id] = key
                    continue
            candidates: list[Tour] = []
            banned: frozenset = frozenset()
            for _ in range(n_candidates):
                tour = self.base.plan(world, force=force, ban={bot.id: banned}, commit=False)[bot.id]
                if tour is None:
                    break
                candidates.append(tour)
                banned |= {tour.key}
            if len(candidates) > 1 and time.perf_counter() - started < ROLLOUT_BUDGET_S:
                # sample by sample over all candidates, so the budget can stop it between samples
                totals = [0.0] * len(candidates)
                for j in range(samples):
                    seed = hash((world.round, bot.id, j)) & 0xFFFFFFFF
                    for i, c in enumerate(candidates):
                        totals[i] += self._rollout(world, store, {**force, bot.id: c.key}, seed, horizon)
                    if time.perf_counter() - started > ROLLOUT_BUDGET_S:
                        break
                # ties go to the earlier candidate, i.e. the auction's own order
                force[bot.id] = candidates[max(range(len(candidates)), key=lambda i: (totals[i], -i))].key
                if self.log is not None:
                    others = self.base.plan(world, force=force, commit=False)
                    self.log.append({"store": store, "round": world.round, "bot": bot.id, "samples": j + 1,
                                     "x": [tour_features(self.tours, world, c, i, others, self.smap.index[bot.pos])
                                           for i, c in enumerate(candidates)],
                                     "y": [t / (j + 1) for t in totals]})
            elif candidates:
                force[bot.id] = candidates[0].key
            self.chosen[bot.id] = force.get(bot.id)
        return self.base.plan(world, force=force)

    def _plan_scored(self, world: World, n_candidates: int, model: str) -> dict[int, Tour | None]:
        if self.model is None:
            m = json.loads((Path(__file__).with_name("scorer") / model).read_text())
            self.model = {k: np.array(v) if isinstance(v, list) else v for k, v in m.items() if k != "names"}
        m = self.model
        force: dict[int, tuple] = {}
        for bot in sorted(world.bots, key=lambda b: b.id):
            key = self.chosen.get(bot.id)
            if key is not None:
                kept = self.base.plan(world, force={**force, bot.id: key}, commit=False)
                if kept[bot.id] is not None and kept[bot.id].key == key:
                    force[bot.id] = key
                    continue
            candidates, banned, others = [], frozenset(), None
            for _ in range(n_candidates):
                plan = self.base.plan(world, force=force, ban={bot.id: banned}, commit=False)
                others = others or plan
                if plan[bot.id] is None:
                    break
                candidates.append(plan[bot.id])
                banned |= {plan[bot.id].key}
            if len(candidates) > 1:
                pos = self.smap.index[bot.pos]
                x = np.array([tour_features(self.tours, world, c, i, others, pos) for i, c in enumerate(candidates)])
                score = np.tanh(((x - m["mu"]) / m["sd"]) @ m["W1"] + m["b1"]) @ m["W2"]
                force[bot.id] = candidates[int(np.argmax(score))].key
            elif candidates:
                force[bot.id] = candidates[0].key
            self.chosen[bot.id] = force.get(bot.id)
        return self.base.plan(world, force=force)

    def _rollout(self, world: World, store: str, force: dict[int, tuple], seed: int, horizon: int) -> float:
        planner = Planner(self.smap, self.tours)
        planner.previous = dict(self.base.previous)
        controller = Controller(self.smap, self.tours)
        engine = Engine(self.smap, UniformOracle(store, self.types, seed))
        held = dict(force)
        for _ in range(min(horizon, world.max_rounds - world.round)):
            tasks = planner.plan(world, force=held)
            held = {b: k for b, k in held.items() if tasks.get(b) is not None and tasks[b].key == k}
            world = engine.step(world, controller.act(world, tasks))
        needed = Counter()
        for t in world.trucks:
            if t.order is not None and not t.order.complete:
                needed += t.order.needed()
        return world.score + sum(min(n, needed[k]) for k, n in Counter(i for b in world.bots for i in b.inventory).items())
