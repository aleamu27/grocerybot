"""Global planner: decide which tour every bot runs this round.

1. Items already carried are matched to the truck that needs most of them,
   preferring docked trucks, and taken out of that truck's residual demand.
2. A greedy auction: each free bot bids its best tour per truck (points per
   round). The single highest bid is committed, its picks leave the residual
   demand, and only bids for that truck are recomputed. Repeat until nobody bids.
3. Hysteresis: a bot's previous plan (same truck, same items) bids 25% higher,
   so near-ties do not flip a bot back and forth.
"""

from collections import Counter

from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import World
from grocery_ai.plan.tours import Tour, TourPlanner
from grocery_ai.sim.engine import INVENTORY_CAPACITY


class Planner:
    def __init__(self, smap: StaticMap):
        self.smap = smap
        self.tours = TourPlanner(smap)
        self.previous: dict[int, tuple] = {}   # bot id -> Tour.key it ran last round
        self.trace: list[dict] = []

    def plan(self, world: World) -> dict[int, Tour | None]:
        rounds_left = world.max_rounds - world.round
        trucks = {
            t.id: t for t in world.trucks
            if t.order is not None and not t.order.complete and t.state != "leaving"
        }
        residual = {tid: t.order.needed() for tid, t in trucks.items()}
        order_size = {tid: sum(r.values()) for tid, r in residual.items()}
        index = self.smap.index

        # 1. carried items
        carried: dict[int, tuple[int, tuple[str, ...]]] = {}
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
            return self.tours.best_tour(
                index[bot.pos], cargo, free[bot_id], residual[tid], trucks[tid], order_size[tid],
                rounds_left, sticky=self.previous.get(bot_id),
            )

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
            residual[tid] -= Counter(t for t, _ in tour.picks)
            for other in open_bots:
                bids[(other, tid)] = bid(other, tid)

        self.previous = {b: t.key for b, t in assigned.items() if t is not None}
        self.trace = [
            {"bot": b, "truck": t.truck_id, "carried": t.carried, "picks": [p for p, _ in t.picks],
             "ready": t.ready, "bid": round(t.bid, 3)}
            for b, t in assigned.items() if t is not None
        ]
        return assigned
