"""Features of a tour candidate, for learning from rollouts (expert iteration).

Rollouts (grocery_ai.plan.rollout) pick better tours than the greedy auction
but are too slow for stores with many bots. tools/collect.py logs, for every
rollout decision, these features of each candidate with its rollout value; a
model fitted to them can then score tours as cheaply as the auction's bid.
"""

from grocery_ai.core.state import World
from grocery_ai.plan.tours import Tour, TourPlanner

NAMES = [
    "bid", "value", "ready", "cost", "wait", "items", "carried", "picks", "then",
    "completes", "order_left", "docked", "truck_bots", "end_to_demand", "rank", "stock",
    "first_dist", "walk", "end_x", "end_y", "end_to_other_trucks",
]


def tour_features(tours: TourPlanner, world: World, tour: Tour, rank: int, tasks: dict, pos: int) -> list[float]:
    """One row per candidate; `tasks` are the other bots' current tours, `pos` the bot's cell."""
    trucks = {t.id: t for t in world.trucks}
    truck = trucks.get(tour.truck_id)    # None for a speculative stock pickup
    needed = truck.order.needed() if truck is not None and truck.order else {}
    order_left = sum(needed.values())
    end = tour.load_cell if truck is not None else tour.picks[-1][1]
    first = len(tour.carried) + sum(tour.first_picks.values())
    items = first + len(tour.then_items)
    # demand still open after this tour, and how far the tour's end cell is from it
    left = {t.id: (t.order.needed() if t.order and not t.order.complete else {}) for t in world.trucks}
    if truck is not None:
        for k, n in tour.first_picks.items():
            left[truck.id][k] = left[truck.id].get(k, 0) - n
    cells = [tours.near[k][0][end] for need in left.values() for k, n in need.items() if n > 0 and k in tours.near]
    end_to_demand = min((tours.dist[end][c] for c in cells), default=0)
    truck_bots = sum(1 for t in tasks.values() if t is not None and t.truck_id == tour.truck_id)
    route = [pos] + [c for _, c in tour.picks]
    walk = sum(tours.dist[a][b] for a, b in zip(route, route[1:]))
    others = [tours.smap.index[c] for t in world.trucks if t.id != tour.truck_id and t.order and not t.order.complete
              for c in t.load_cells]
    end_to_other = min((tours.dist[end][c] for c in others), default=0)
    ex, ey = tours.smap.cells[end]
    return [
        tour.bid, tour.value, tour.ready, tour.cost, truck.rounds_until_back if truck else 0, items,
        len(tour.carried), len(tour.picks), float(tour.then_truck is not None), float(first >= order_left > 0),
        order_left, float(truck is not None and truck.state == "docked"), truck_bots, end_to_demand, rank,
        float(truck is None), tours.dist[pos][route[1]] if len(route) > 1 else 0, walk, ex, ey, end_to_other,
    ]
