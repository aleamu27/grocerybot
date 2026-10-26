"""Play bot_v2 (planner + controller) against the local engine, no CLI.

The engine needs an Oracle for what the server hides: trip durations and new
orders. Both are drawn as measured on real runs: trip T uniform on the
store's range, order size uniform, item types uniform and independent (no
Markov structure: item-to-item chi-square ~ df, consecutive orders overlap
as much as random pairs, trip lag-1 r ~ 0). Starting states come from
recordings/<store>-1.jsonl (layouts are the same for every seed).
"""

import random
from pathlib import Path

from grocery_ai.control.controller import Controller
from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import Order, World
from grocery_ai.plan.planner import Planner
from grocery_ai.sim.engine import Engine
from grocery_ai.sim.fidelity import load_steps

TRIP = {"easy": (8, 12), "medium": (9, 15), "hard": (9, 15), "expert": (14, 22), "nightmare": (17, 27)}
SIZE = {"easy": (3, 4), "medium": (3, 5), "hard": (3, 5), "expert": (4, 6), "nightmare": (4, 7)}
RECORDINGS = Path(__file__).resolve().parents[2] / "recordings"
ROLLOUT = True   # play as bot_v2 does, with the rollout planner


class UniformOracle:
    def __init__(self, store: str, types: list[str], seed: int):
        self.store, self.types, self.rng, self.n = store, types, random.Random(seed), 0

    def trip_duration(self, truck, round: int) -> int:
        return self.rng.randint(*TRIP[self.store])

    def next_order(self, truck, round: int) -> Order:
        self.n += 1
        size = self.rng.randint(*SIZE[self.store])
        return Order(f"o{self.n}", tuple(self.rng.choice(self.types) for _ in range(size)))


def start_state(store: str) -> dict:
    return load_steps(str(RECORDINGS / f"{store}-1.jsonl"))[0]["state"]


def play(store: str, seed: int) -> int:
    """Final score of one simulated game."""
    state = start_state(store)
    smap = StaticMap.from_state(state)
    from grocery_ai.plan.rollout import RolloutPlanner   # here: rollout imports this module
    planner = RolloutPlanner(smap) if ROLLOUT else Planner(smap)
    controller = Controller(smap, planner.tours)
    engine = Engine(smap, UniformOracle(store, sorted(smap.access), seed))
    world = World.from_state(state)
    for _ in range(world.max_rounds):
        world = engine.step(world, controller.act(world, planner.plan(world)))
    return world.score
