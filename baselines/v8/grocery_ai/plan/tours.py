"""Tour generation: up to three pickups followed by one truck's load cell,
optionally followed by a second, nearby truck's load cell.

A tour's picks are chosen from a truck's residual demand. Each pickup goes to the
nearest pickup cell of its type from wherever the bot is at that point, and every
pickup order is tried, so a tour costs a handful of list lookups to evaluate.
Single-truck tours left a bot driving to the load row with one item in 53% of
Easy loads while the neighbouring truck (load cells 2 apart) still needed
items, so a tour may split its picks between its truck and one `then` truck.
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
    ready: int                        # rounds until the (last) load has happened
    value: float
    bid: float = field(default=0.0)
    then_truck: int | None = None     # second truck, loaded right after truck_id
    then_items: tuple[str, ...] = ()  # everything delivered to it, carried or picked
    then_picks: tuple[str, ...] = ()  # the picked part of then_items

    @property
    def first_picks(self) -> Counter:
        """Picked item types that go to truck_id."""
        return Counter(t for t, _ in self.picks) - Counter(self.then_picks)

    @property
    def key(self) -> tuple:
        """Identity for hysteresis: same trucks, same items delivered."""
        return (self.truck_id, tuple(sorted(self.carried + tuple(self.first_picks.elements()))),
                self.then_truck, tuple(sorted(self.then_items)))


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
        then: tuple | None = None,
    ) -> Tour | None:
        """Highest-bid tour for one bot and one truck, or None if nothing is worth doing.

        then = (truck, residual, order_size, carried) of a second truck: only
        tours that deliver to both, first `truck`, then that one, are tried."""
        leg_cell, leg_dist = self.truck_leg(truck.id, truck.load_cells)
        wait = truck.rounds_until_back
        room = min(free, INVENTORY_CAPACITY)
        first = [c for c in self._combos(residual, room) if carried or c]
        if then is None:
            pairs = [(c, ()) for c in first if c or carried]
        else:
            truck2, residual2, order_size2, carried2 = then
            leg2_cell, leg2_dist = self.truck_leg(truck2.id, truck2.load_cells)
            wait2 = truck2.rounds_until_back
            pairs = [(a, b) for a in first for b in self._combos(residual2, room - len(a)) if carried2 or b]

        best: Tour | None = None
        for combo, combo2 in pairs:
            n = len(carried) + len(combo)
            value = n + ORDER_BONUS * n / max(order_size, 1)
            if then is not None:
                n2 = len(carried2) + len(combo2)
                value += n2 + ORDER_BONUS * n2 / max(order_size2, 1)
            for order in sorted(set(permutations(combo + combo2))):  # sorted: set order follows string hashing
                cur, cost, picks = pos, 0, []
                for item_type in order:
                    cells, dists = self.near[item_type]
                    cost += dists[cur] + 1
                    cur = cells[cur]
                    picks.append((item_type, cur))
                cost += leg_dist[cur]
                ready = max(cost, wait) + 1
                if then is not None:
                    ready = max(ready + leg2_dist[leg_cell[cur]], wait2) + 1
                if ready > rounds_left:
                    continue
                tour = Tour(truck.id, picks, leg_cell[cur], carried, cost, ready, value)
                if then is not None:
                    tour.then_truck, tour.then_items, tour.then_picks = truck2.id, tuple(carried2) + combo2, combo2
                tour.bid = value / ready
                if sticky is not None and tour.key == sticky:
                    tour.bid *= stickiness
                if best is None or tour.bid > best.bid:
                    best = tour
        return best

    def _combos(self, residual: Counter, room: int) -> list[tuple[str, ...]]:
        """Every multiset of up to `room` item types the residual demand still needs, () included."""
        kinds = sorted(t for t, n in residual.items() if n > 0 and t in self.near)
        combos: list[tuple[str, ...]] = [()]
        for size in range(1, room + 1):
            for combo in combinations_with_replacement(kinds, size):
                if all(combo.count(t) <= residual[t] for t in set(combo)):
                    combos.append(combo)
        return combos
