"""Tour generation: up to three pickups followed by one truck's load cell.

A tour's picks are chosen from a truck's residual demand. Each pickup goes to the
nearest pickup cell of its type from wherever the bot is at that point, and every
pickup order is tried, so a tour costs a handful of list lookups to evaluate.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, permutations

from grocery_ai.core.model import MOVES, StaticMap
from grocery_ai.sim.engine import INVENTORY_CAPACITY, ORDER_BONUS


@dataclass
class Tour:
    truck_id: int
    picks: list[tuple[str, int]]      # (item type, pickup cell index), in order
    load_cell: int                    # cell index of the load cell to finish on
    carried: tuple[str, ...]          # items already in inventory this tour delivers
    cost: int                         # rounds of moving and picking before the load
    ready: int                        # rounds until the load has happened
    value: float
    bid: float = field(default=0.0)

    @property
    def key(self) -> tuple:
        """Identity for hysteresis: same truck, same items delivered."""
        return self.truck_id, tuple(sorted(self.carried + tuple(t for t, _ in self.picks)))


class TourPlanner:
    def __init__(self, smap: StaticMap):
        self.smap = smap
        self.dist = smap.dist.tolist()
        # type -> (nearest pickup cell, its distance) from every cell
        self.near: dict[str, tuple[list[int], list[int]]] = {}
        for item_type, acc in smap.access.items():
            sub = smap.dist[:, acc]
            self.near[item_type] = ([acc[a] for a in sub.argmin(axis=1).tolist()], sub.min(axis=1).tolist())
        # (pickup cell, type) -> an item id on an adjacent shelf
        self.item_at: dict[tuple[int, str], str] = {}
        for item_id, (x, y) in smap.item_cells.items():
            for dx, dy in MOVES.values():
                idx = smap.index.get((x + dx, y + dy))
                if idx is not None:
                    self.item_at.setdefault((idx, smap.item_types[item_id]), item_id)
        self._truck_leg: dict[int, tuple[list[int], list[int]]] = {}

    def truck_leg(self, truck_id: int, load_cells) -> tuple[list[int], list[int]]:
        """(nearest load cell, its distance) from every cell, for one truck."""
        if truck_id not in self._truck_leg:
            idx = [self.smap.index[c] for c in sorted(load_cells)]
            sub = self.smap.dist[:, idx]
            self._truck_leg[truck_id] = ([idx[a] for a in sub.argmin(axis=1).tolist()], sub.min(axis=1).tolist())
        return self._truck_leg[truck_id]

    def best_tour(
        self,
        pos: int,
        carried: tuple[str, ...],
        free: int,
        residual: Counter,
        truck,
        order_size: int,
        rounds_left: int,
        sticky: tuple | None = None,
        stickiness: float = 1.25,
    ) -> Tour | None:
        """Highest-bid tour for one bot and one truck, or None if nothing is worth doing."""
        leg_cell, leg_dist = self.truck_leg(truck.id, truck.load_cells)
        wait = truck.rounds_until_back
        kinds = sorted(t for t, n in residual.items() if n > 0 and t in self.near)
        combos: list[tuple[str, ...]] = [()] if carried else []
        for size in range(1, min(free, INVENTORY_CAPACITY) + 1):
            for combo in combinations_with_replacement(kinds, size):
                if all(combo.count(t) <= residual[t] for t in set(combo)):
                    combos.append(combo)

        best: Tour | None = None
        for combo in combos:
            n = len(carried) + len(combo)
            value = n + ORDER_BONUS * n / max(order_size, 1)
            for order in sorted(set(permutations(combo))):  # sorted: set order follows string hashing
                cur, cost, picks = pos, 0, []
                for item_type in order:
                    cells, dists = self.near[item_type]
                    cost += dists[cur] + 1
                    cur = cells[cur]
                    picks.append((item_type, cur))
                cost += leg_dist[cur]
                ready = max(cost, wait) + 1
                if ready > rounds_left:
                    continue
                tour = Tour(truck.id, picks, leg_cell[cur], carried, cost, ready, value)
                tour.bid = value / ready
                if sticky is not None and tour.key == sticky:
                    tour.bid *= stickiness
                if best is None or tour.bid > best.bid:
                    best = tour
        return best
