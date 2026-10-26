"""Router regressions: head-on meetings must resolve, and plans must match the engine.

Run: python3 -m unittest discover -s tests -t .
"""

import time
import unittest

from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.route.cbs import route
from grocery_ai.route.whca import Router
from grocery_ai.sim.engine import Engine
from tests.test_engine import NoOracle


def pocket_state(bots: list[tuple[int, int]]) -> dict:
    """7x4 room: corridor y=1, x=1..5, with a one-cell pocket at (3, 2)."""
    walls = [[x, y] for x in range(7) for y in (0, 3)] + [[0, 1], [6, 1], [0, 2], [6, 2]]
    walls += [[x, 2] for x in (1, 2, 4, 5)]
    return {
        "round": 0, "max_rounds": 50, "score": 0, "orders_completed": 0,
        "grid": {"width": 7, "height": 4, "walls": walls},
        "bots": [{"id": i, "position": list(p), "inventory": []} for i, p in enumerate(bots)],
        "items": [], "trucks": [],
    }


def drive(state: dict, goals: dict[int, tuple[int, int]], priority: list[int], rounds: int) -> tuple[World, int]:
    smap = StaticMap.from_state(state, spawn=(-1, -1))
    router = Router(smap, smap.dist.tolist(), window=8)
    engine = Engine(smap, NoOracle())
    world = World.from_state(state)
    names = {(0, -1): "move_up", (0, 1): "move_down", (-1, 0): "move_left", (1, 0): "move_right"}
    mismatches = 0
    for _ in range(rounds):
        position = {b.id: smap.index[b.pos] for b in world.bots}
        targets = {b: (smap.index[goals[b]], 99) for b in priority}
        paths, _ = route(router, priority, targets, position, {}, time.perf_counter() + 1)
        planned, actions = {}, []
        for bot in priority:
            path = paths[bot]
            nxt = smap.cells[path[1]] if len(path) > 1 else world.bots[bot].pos
            planned[bot] = nxt
            cur = world.bots[bot].pos
            move = names.get((nxt[0] - cur[0], nxt[1] - cur[1]))
            actions.append({"bot": bot, "action": move or "wait"})
        world = engine.step(world, actions)
        mismatches += sum(world.bots[b].pos != planned[b] for b in planned)
        if all(world.bots[b].pos == goals[b] for b in goals):
            break
    return world, mismatches


class HeadOn(unittest.TestCase):
    """The pocket is only reachable in time if the first-planned bot waits.
    Prioritized planning never finds that, whichever bot goes first; local
    CBS must."""

    def check(self, priority: list[int]):
        state = pocket_state([(1, 1), (5, 1)])
        world, mismatches = drive(state, {0: (5, 1), 1: (1, 1)}, priority, rounds=20)
        self.assertEqual([b.pos for b in world.bots], [(5, 1), (1, 1)])
        self.assertEqual(mismatches, 0)

    def test_resolves_with_low_id_first(self):
        self.check([0, 1])

    def test_resolves_with_high_id_first(self):
        self.check([1, 0])


if __name__ == "__main__":
    unittest.main()
