"""Global planner: decide which tour every bot runs this round.

1. Items already carried are matched to the truck that needs most of them,
   preferring docked trucks, and taken out of that truck's residual demand.
   The rest may be matched to a truck next to it, for the same tour.
2. A greedy auction: each free bot bids its best tour per truck (points per
   round), alone or followed by a nearby truck (MULTI_TRUCK). The single
   highest bid is committed, its picks leave the residual demand, and only
   bids that use the trucks it touched are recomputed. Repeat until nobody bids.
3. Hysteresis: a bot's previous plan (same truck, same items) bids 25% higher,
   so near-ties do not flip a bot back and forth.
4. Stock: bots left without a tour fetch items nobody has asked for yet, of
   the types that are slowest to fetch for the trucks. An order's last items
   otherwise arrive long after its truck docks (a far shelf is a 30-60 round
   trip; the order is visible only T rounds ahead), and item types in orders
   are roughly uniform, so a carried item is soon needed. Step 1 then hands
   it to the truck. There is no drop action, so each bot keeps
   INVENTORY_CAPACITY - STOCK_SLOTS slots free for real tours.
"""

from collections import Counter

from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.plan.tours import Tour, TourPlanner
from grocery_ai.sim.engine import INVENTORY_CAPACITY

STOCK_SLOTS = 2        # items a bot may carry on speculation
STOCK_PER_TYPE = 1     # unclaimed items of each type to keep in stock
STOCK_MAX_DENSITY = 0.06   # bots per walkable cell above which stocking is off
                           # (Nightmare, 0.086: -13/game in SPRT; Expert, 0.042: +17)
MULTI_TRUCK = True         # tours may load a second, nearby truck right after the first, when bots <= trucks
                           # (sim, 40 seeds: Easy +2.9, Medium +5.4, Hard 0; Expert -10, Nightmare -41, as
                           # the second leg runs along the one-lane load row)
MULTI_TRUCK_RADIUS = 4     # max distance between the two trucks' load cells


class Planner:
    def __init__(self, smap: StaticMap, tours: TourPlanner | None = None):
        self.smap = smap
        self.tours = tours or TourPlanner(smap)
        self.previous: dict[int, tuple] = {}   # bot id -> Tour.key it ran last round
        self.trace: list[dict] = []
        self.fetch: dict[str, int] | None = None   # type -> rounds from its shelf to the nearest truck
        self.near_trucks: dict[int, list[int]] | None = None

    def plan(self, world: World, force: dict[int, tuple] | None = None, ban: dict[int, frozenset] | None = None,
             commit: bool = True) -> dict[int, Tour | None]:
        """force: bot -> Tour.key it must keep if still possible; ban: bot -> keys it may not
        take (both for rollouts, grocery_ai.plan.rollout). commit=False leaves the
        hysteresis state alone."""
        force, ban = force or {}, ban or {}
        rounds_left = world.max_rounds - world.round
        trucks = {
            t.id: t for t in world.trucks
            if t.order is not None and not t.order.complete and t.state != "leaving"
        }
        residual = {tid: t.order.needed() for tid, t in trucks.items()}
        order_size = {tid: sum(r.values()) for tid, r in residual.items()}
        index = self.smap.index

        multi = MULTI_TRUCK and len(world.bots) <= len(world.trucks)

        # 1. carried items
        carried: dict[int, tuple[int, tuple[str, ...]]] = {}
        carried_then: dict[int, tuple[int, tuple[str, ...]]] = {}
        for bot in sorted(world.bots, key=lambda b: -len(b.inventory)):
            if not bot.inventory:
                continue
            options = []
            for tid, truck in trucks.items():
                match = list((Counter(bot.inventory) & residual[tid]).elements())
                if match:
                    _, leg = self.tours.truck_leg(tid, truck.load_cells)
                    options.append((len(match), truck.state == "docked", -leg[index[bot.pos]], tid, match))
            if options:
                *_, tid, match = max(options)
                residual[tid] -= Counter(match)
                carried[bot.id] = (tid, tuple(match))
                # the rest may go to a truck next to it, on the same tour
                left = Counter(bot.inventory) - Counter(match)
                nexts = [(len(m), tid2, m) for tid2 in self._near_trucks(world, tid) if tid2 in trucks
                         for m in [list((left & residual[tid2]).elements())] if m]
                if multi and nexts:
                    _, tid2, match2 = max(nexts)
                    residual[tid2] -= Counter(match2)
                    carried_then[bot.id] = (tid2, tuple(match2))

        # 2. auction
        bots = {b.id: b for b in world.bots}
        free = {b.id: INVENTORY_CAPACITY - len(b.inventory) for b in world.bots}

        def bid(bot_id: int, tid: int) -> Tour | None:
            bot = bots[bot_id]
            if bot_id in carried and carried[bot_id][0] != tid:
                return None
            cargo = carried.get(bot_id, (tid, ()))[1]
            if not cargo and not any(residual[tid].values()):
                return None
            args = (index[bot.pos], cargo, free[bot_id], residual[tid], trucks[tid], order_size[tid], rounds_left)
            sticky = {"sticky": force[bot_id], "stickiness": 1e9} if bot_id in force \
                else {"sticky": self.previous.get(bot_id)}
            sticky["ban"] = ban.get(bot_id, frozenset())
            best = self.tours.best_tour(*args, **sticky)
            if not multi:
                return best
            if bot_id in carried_then:
                seconds = [carried_then[bot_id]]
            else:
                seconds = [(tid2, ()) for tid2 in self._near_trucks(world, tid) if tid2 in trucks]
            for tid2, cargo2 in seconds:
                if not cargo2 and (not any(residual[tid2].values())
                                   or sum(residual[tid].values()) >= free[bot_id]):
                    continue    # nothing for it, or the first truck alone fills the bot
                tour = self.tours.best_tour(*args, **sticky,
                                            then=(trucks[tid2], residual[tid2], order_size[tid2], cargo2))
                if tour is not None and (best is None or tour.bid > best.bid):
                    best = tour
            return best

        bids = {(b, tid): bid(b, tid) for b in bots for tid in trucks}
        assigned: dict[int, Tour | None] = {b: None for b in bots}
        open_bots = set(bots)
        while open_bots:
            best = max(
                ((b, tid, tour) for (b, tid), tour in bids.items() if b in open_bots and tour is not None),
                key=lambda x: x[2].bid, default=None,
            )
            if best is None:
                break
            b, tid, tour = best
            assigned[b] = tour
            open_bots.discard(b)
            residual[tid] -= tour.first_picks
            touched = {tid}
            if tour.then_truck is not None:
                residual[tour.then_truck] -= Counter(tour.then_picks)
                touched.add(tour.then_truck)
            for other in open_bots:
                for t in trucks:
                    old = bids[(other, t)]
                    if t in touched or (old is not None and old.then_truck in touched):
                        bids[(other, t)] = bid(other, t)

        reserved = {b: (tid, cargo + carried_then.get(b, (None, ()))[1]) for b, (tid, cargo) in carried.items()}
        self._stock(world, assigned, reserved, free, rounds_left)
        if not commit:
            return assigned
        self.previous = {b: t.key for b, t in assigned.items() if t is not None}
        self.trace = [
            {"bot": b, "truck": t.truck_id, "carried": t.carried, "picks": [p for p, _ in t.picks],
             "then": t.then_truck, "then_items": t.then_items, "ready": t.ready, "bid": round(t.bid, 3)}
            for b, t in assigned.items() if t is not None
        ]
        return assigned

    def _fetch(self, world: World) -> dict[str, int]:
        if self.fetch is None:
            index, near = self.smap.index, self.tours.near
            load = [index[c] for t in world.trucks for c in t.load_cells]
            self.fetch = {k: min(self.tours.dist[c][l] for c in self.smap.access[k] for l in load) for k in near}
        return self.fetch

    def _near_trucks(self, world: World, tid: int) -> list[int]:
        """Trucks whose load cells lie within MULTI_TRUCK_RADIUS of truck tid's."""
        if self.near_trucks is None:
            dist, index = self.tours.dist, self.smap.index
            cells = {t.id: [index[c] for c in t.load_cells] for t in world.trucks}
            self.near_trucks = {
                a: [b for b in cells if b != a and min(dist[x][y] for x in cells[a] for y in cells[b]) <= MULTI_TRUCK_RADIUS]
                for a in cells
            }
        return self.near_trucks.get(tid, [])

    def _stock(self, world: World, assigned, carried, free, rounds_left: int) -> None:
        """Give each tourless bot with a spare stock slot one speculative pickup
        (a Tour with truck_id None): the nearest type still short of
        STOCK_PER_TYPE unclaimed items, among the slower half to fetch."""
        index, near = self.smap.index, self.tours.near
        if len(world.bots) > STOCK_MAX_DENSITY * len(self.smap.cells):
            return
        self._fetch(world)
        stock = Counter()
        for bot in world.bots:
            stock += Counter(bot.inventory) - Counter(carried.get(bot.id, (None, ()))[1])
        kinds = sorted(near, key=lambda k: (-self.fetch[k], k))[: max(1, len(near) // 2)]
        for bot in sorted(world.bots, key=lambda b: b.id):
            spare = len(bot.inventory) - len(carried.get(bot.id, (None, ()))[1])
            if assigned[bot.id] is not None or free[bot.id] == 0 or spare >= STOCK_SLOTS:
                continue
            short = [k for k in kinds if stock[k] < STOCK_PER_TYPE]
            if not short:
                break
            pos = index[bot.pos]
            k = min(short, key=lambda k: (near[k][1][pos], k))
            d = near[k][1][pos] + 1
            if 2 * d > rounds_left:
                continue
            assigned[bot.id] = Tour(None, [(k, near[k][0][pos])], -1, (), d, d, 0.0)
            stock[k] += 1
