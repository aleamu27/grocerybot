"""Guidance graph: parameters tie every move, files round-trip, tolls steer the router.

Run: python3 -m unittest discover -s tests -t .
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from grocery_ai.core.model import StaticMap
from grocery_ai.route import guidance
from grocery_ai.route.whca import Router
from grocery_ai.sim.selfplay import start_state


class Guidance(unittest.TestCase):
    def setUp(self):
        self.smap = StaticMap.from_state(start_state("hard"))

    def test_every_move_has_a_parameter(self):
        keys = guidance.edge_keys(self.smap)
        moves = sum(len(n) for n in self.smap.neighbours)
        self.assertEqual(len(keys), moves)
        self.assertIn(("lane",), guidance.param_names(self.smap))

    def test_save_load_round_trip(self):
        params = {n: 0.0 for n in guidance.param_names(self.smap)}
        params[("lane",)] = 1.5
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(guidance, "DIR", Path(tmp)):
            self.assertIsNone(guidance.load(self.smap))
            guidance.save(self.smap, params, {"store": "hard"})
            self.assertEqual(guidance.load(self.smap), params)

    def test_edge_toll_changes_the_route(self):
        """Tolling 'down' moves in one aisle makes a bot avoid going down it."""
        dist = self.smap.dist.tolist()
        router = Router(self.smap, dist, window=40)
        top = self.smap.index[(1, 1)]
        bottom = self.smap.index[(1, 11)]
        free = router.search(0, top, bottom, 0)
        self.assertEqual(len(free) - 1, dist[top][bottom])
        tolls = guidance.edge_tolls(self.smap, {("v", 1, "down"): 50.0})
        router.set_tolls(set(), 0, 0, tolls)
        detour = router.search(0, top, bottom, 0)
        self.assertEqual(detour[-1], bottom)
        self.assertGreater(len(detour) - 1, dist[top][bottom])


if __name__ == "__main__":
    unittest.main()
