"""Engine semantics, recorded-game fidelity, and mutants the fidelity check must catch.

Run: python3 -m unittest discover -s tests -t .
"""

import glob
import os
import unittest
from collections import Counter
from unittest import mock

from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.sim import engine as engine_mod
from grocery_ai.sim import fidelity
from grocery_ai.sim.engine import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS = sorted(glob.glob(os.path.join(ROOT, "recordings", "*.jsonl")))


def corridor_state(bots: list[tuple[int, int]]) -> dict:
    """A 7x3 room: one walkable row y=1, x=1..5, a truck loading from (1, 1)."""
    walls = [[x, y] for x in range(7) for y in (0, 2)] + [[0, 1], [6, 1]]
    return {
        "round": 0, "max_rounds": 10, "score": 0, "orders_completed": 0,
        "grid": {"width": 7, "height": 3, "walls": walls},
        "bots": [{"id": i, "position": list(p), "inventory": []} for i, p in enumerate(bots)],
        "items": [{"id": "item_0", "type": "milk", "position": [3, 0]}],
        "trucks": [{"id": 0, "bay": [1, 0], "load_cells": [[1, 1]], "state": "docked", "rounds_until_back": 0,
                    "order": {"id": "o", "items_required": ["milk"], "items_delivered": [], "complete": False}}],
    }


class NoOracle:
    def trip_duration(self, truck, round):
        return 10

    def next_order(self, truck, round):
        return None


def step(state: dict, actions: list[dict], spawn=(5, 1)) -> World:
    smap = StaticMap.from_state(state, spawn=spawn)
    return Engine(smap, NoOracle()).step(World.from_state(state), actions)


class Semantics(unittest.TestCase):
    def test_train_moves_when_front_bot_has_lower_id(self):
        world = step(corridor_state([(2, 1), (3, 1)]), [{"bot": 0, "action": "move_left"}, {"bot": 1, "action": "move_left"}])
        self.assertEqual([b.pos for b in world.bots], [(1, 1), (2, 1)])

    def test_train_stalls_when_front_bot_has_higher_id(self):
        world = step(corridor_state([(3, 1), (2, 1)]), [{"bot": 0, "action": "move_left"}, {"bot": 1, "action": "move_left"}])
        self.assertEqual([b.pos for b in world.bots], [(3, 1), (1, 1)])

    def test_swap_is_impossible(self):
        world = step(corridor_state([(2, 1), (3, 1)]), [{"bot": 0, "action": "move_right"}, {"bot": 1, "action": "move_left"}])
        self.assertEqual([b.pos for b in world.bots], [(2, 1), (3, 1)])

    def test_spawn_accepts_stacking(self):
        world = step(corridor_state([(4, 1), (5, 1)]), [{"bot": 0, "action": "move_right"}])
        self.assertEqual([b.pos for b in world.bots], [(5, 1), (5, 1)])

    def test_pickup_needs_adjacency_and_space(self):
        state = corridor_state([(3, 1), (1, 1)])
        state["bots"][1]["inventory"] = ["milk"] * 3
        world = step(state, [{"bot": 0, "action": "pick_up", "item_id": "item_0"},
                             {"bot": 1, "action": "pick_up", "item_id": "item_0"}])
        self.assertEqual([len(b.inventory) for b in world.bots], [1, 3])

    def test_completing_load_scores_bonus_and_closes_truck(self):
        state = corridor_state([(1, 1), (5, 1)])
        state["bots"][0]["inventory"] = ["milk", "milk"]
        world = step(state, [{"bot": 0, "action": "load"}])
        self.assertEqual((world.score, world.orders_completed), (6, 1))
        self.assertEqual(world.bots[0].inventory, ["milk"])
        self.assertEqual((world.trucks[0].state, world.trucks[0].rounds_until_back), ("leaving", 10))

    def test_first_action_per_bot_wins(self):
        world = step(corridor_state([(3, 1), (5, 1)]), [{"bot": 0, "action": "move_left"}, {"bot": 0, "action": "move_right"}])
        self.assertEqual(world.bots[0].pos, (2, 1))


@unittest.skipUnless(RECORDINGS, "no recordings; run tools/record.py first")
class Fidelity(unittest.TestCase):
    def mismatches(self) -> int:
        return sum(fidelity.check(path)[1] for path in RECORDINGS)

    def test_engine_matches_every_recorded_step(self):
        self.assertEqual(self.mismatches(), 0)

    def test_catches_reversed_resolution_order(self):
        original = sorted
        with mock.patch.object(engine_mod, "sorted", create=True,
                               side_effect=lambda xs, key=None: original(xs, key=key, reverse=True)):
            self.assertGreater(self.mismatches(), 0)

    def test_catches_missing_spawn_exemption(self):
        original = StaticMap.from_state.__func__

        def no_spawn(cls, state, spawn=None):
            return original(cls, state, spawn=(-1, -1))

        with mock.patch.object(StaticMap, "from_state", classmethod(no_spawn)):
            self.assertGreater(self.mismatches(), 0)

    def test_catches_wrong_capacity(self):
        with mock.patch.object(engine_mod, "INVENTORY_CAPACITY", 4):
            self.assertGreater(self.mismatches(), 0)

    def test_catches_wrong_bonus(self):
        with mock.patch.object(engine_mod, "ORDER_BONUS", 4):
            self.assertGreater(self.mismatches(), 0)


if __name__ == "__main__":
    unittest.main()
