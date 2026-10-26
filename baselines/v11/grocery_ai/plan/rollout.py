"""Rollout planner: choose each bot's tour by playing the next rounds forward.

The greedy auction (planner.py) scores a tour by points per round on its own,
so it cannot see that a choice now decides where the bot ends up and what is
left to fetch. On stores with few bots (ROLLOUT_MAX_BOTS), each bot's top
ROLLOUT_CANDIDATES tours (the auction's pick, then the next best with the
earlier picks banned) are each played forward ROLLOUT_HORIZON rounds with the
greedy policy on the exact engine, the bot held to that tour until it is done,
against ROLLOUT_SAMPLES sampled futures (trip durations and orders, uniform as
measured, the same samples for every candidate). The tour with the best mean
score at the horizon, plus 1 per carried item an open order still needs, wins.
Bots are decided in ID order; later bots roll out with earlier choices held.
A chosen tour is kept until it is done or impossible, so rollouts run only
when a bot picks its next tour (and never past ROLLOUT_BUDGET_S per round;
the round deadline is 2 s).
"""

import time
from collections import Counter

from grocery_ai.control.controller import Controller
from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.plan.planner import Planner
from grocery_ai.plan.tours import Tour
from grocery_ai.sim.engine import Engine
from grocery_ai.sim.selfplay import UniformOracle

ROLLOUT_MAX_BOTS = 3       # 1 -> 3 (Medium): sim +14.9, W/L 14/2
ROLLOUT_CANDIDATES = 4
ROLLOUT_HORIZON = 40
ROLLOUT_SAMPLES = 12      # 4 -> 12: Easy +2.3, W/L 11/2 (horizon 80: -3.9; 8 candidates: -0.8)
ROLLOUT_BUDGET_S = 1.0
STORE_BY_BOTS = {1: "easy", 3: "medium", 5: "hard", 10: "expert", 20: "nightmare"}


class RolloutPlanner:
    def __init__(self, smap: StaticMap):
        self.smap = smap
        self.base = Planner(smap)
        self.tours = self.base.tours
        self.types = sorted(smap.access)
        self.chosen: dict[int, tuple] = {}   # bot id -> Tour.key it was given by a rollout

    @property
    def trace(self) -> list[dict]:
        return self.base.trace

    def plan(self, world: World) -> dict[int, Tour | None]:
        store = STORE_BY_BOTS.get(len(world.bots))
        if store is None or len(world.bots) > ROLLOUT_MAX_BOTS:
            return self.base.plan(world)
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
            for _ in range(ROLLOUT_CANDIDATES):
                tour = self.base.plan(world, force=force, ban={bot.id: banned}, commit=False)[bot.id]
                if tour is None:
                    break
                candidates.append(tour)
                banned |= {tour.key}
            if len(candidates) > 1 and time.perf_counter() - started < ROLLOUT_BUDGET_S:
                # sample by sample over all candidates, so the budget can stop it between samples
                totals = [0.0] * len(candidates)
                for j in range(ROLLOUT_SAMPLES):
                    seed = hash((world.round, bot.id, j)) & 0xFFFFFFFF
                    for i, c in enumerate(candidates):
                        totals[i] += self._rollout(world, store, {**force, bot.id: c.key}, seed)
                    if time.perf_counter() - started > ROLLOUT_BUDGET_S:
                        break
                # ties go to the earlier candidate, i.e. the auction's own order
                force[bot.id] = candidates[max(range(len(candidates)), key=lambda i: (totals[i], -i))].key
            elif candidates:
                force[bot.id] = candidates[0].key
            self.chosen[bot.id] = force.get(bot.id)
        return self.base.plan(world, force=force)

    def _rollout(self, world: World, store: str, force: dict[int, tuple], seed: int) -> float:
        planner = Planner(self.smap, self.tours)
        planner.previous = dict(self.base.previous)
        controller = Controller(self.smap, self.tours)
        engine = Engine(self.smap, UniformOracle(store, self.types, seed))
        held = dict(force)
        for _ in range(min(ROLLOUT_HORIZON, world.max_rounds - world.round)):
            tasks = planner.plan(world, force=held)
            held = {b: k for b, k in held.items() if tasks.get(b) is not None and tasks[b].key == k}
            world = engine.step(world, controller.act(world, tasks))
        needed = Counter()
        for t in world.trucks:
            if t.order is not None and not t.order.complete:
                needed += t.order.needed()
        return world.score + sum(min(n, needed[k]) for k, n in Counter(i for b in world.bots for i in b.inventory).items())
