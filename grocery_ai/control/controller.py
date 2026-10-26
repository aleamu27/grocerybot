"""Turn tours into one action per bot.

Per round: bots that can load or pick up do so; every other bot gets a goal
(next pickup cell, a load cell of its truck, or for idle bots its own cell, or
the nearest cell off the load row) and the time-space router plans them in
priority order. Priority: bots working for a docked truck, then bots heading
to a truck, then pickers, then idle bots, with bots that have waited longer moved up (aging), so nobody starves.
Bots that prioritized planning leaves stuck are untangled with local CBS
(grocery_ai.route.cbs).
The load row is a one-lane corridor through every truck's load cells, so
rounds spent on it away from one's goal carry a toll (LANE_TOLL): bots reach
their truck through the aisle below it instead of along the row.
Idle bots park: on open cells away from pickup cells and the load row, or on
the spawn cell, where any number of bots may stack without blocking anyone.
A bot that has not left one or two cells for FROZEN_LIMIT rounds although its
goal is elsewhere takes a seeded random side step or wait, as a last-resort
safety net.
"""

import random
import time
from collections import deque

from grocery_ai.core.model import MOVES, StaticMap
from grocery_ai.core.state import World
from grocery_ai.plan.tours import Tour, TourPlanner
from grocery_ai.route import guidance
from grocery_ai.route.cbs import route
from grocery_ai.route.whca import Router
from grocery_ai.sim.engine import INVENTORY_CAPACITY

FROZEN_LIMIT = 12
PARK_WEIGHT = 3.0     # idle bots park on central cells, not where they stand (Nightmare +15/game in sim, 70 seeds)
PARK_MIN_BOTS_PER_TRUCK = 3   # only with this many bots per truck (Hard, Expert: -1 to -4)
SWING_UNSTICK = True  # also unstick bots swinging between two cells (head-on livelock in an aisle)
AGING = 4             # rounds of waiting that outweigh one priority class
HOLD_LOAD = 99        # a bot on a load cell keeps it until it has loaded
HOLD_PICK = 1
ROUTE_BUDGET_S = 0.8
LANE_TOLL = 1         # extra cost per round on the load row away from one's goal
LANE_RADIUS = 2
GUIDANCE = None       # guidance params to use instead of the saved file (tools/guidance_opt.py)
LOAD_ANYWHERE = False  # True: also load into another docked truck one passes (lost on every store)


class Controller:
    def __init__(self, smap: StaticMap, tours: TourPlanner, window: int = 12):
        self.smap = smap
        self.tours = tours
        self.router = Router(smap, tours.dist, window)
        self.move_name = {}
        for i, (x, y) in enumerate(smap.cells):
            for name, (dx, dy) in MOVES.items():
                j = smap.index.get((x + dx, y + dy))
                if j is not None:
                    self.move_name[(i, j)] = name
        self.trace: list[dict] = []
        self.history: dict[int, deque] = {}
        self.waited: dict[int, int] = {}
        self.rng = random.Random(0)
        self.parking: list[int] | None = None

    def act(self, world: World, tasks: dict[int, Tour | None]) -> list[dict]:
        started = time.perf_counter()
        index = self.smap.index
        trucks = {t.id: t for t in world.trucks}
        load_owner = {index[c]: t.id for t in world.trucks for c in t.load_cells}
        position = {b.id: index[b.pos] for b in world.bots}

        actions: dict[int, dict] = {}
        goal: dict[int, tuple[int, int]] = {}     # bot -> (cell, hold)
        rank: dict[int, int] = {}
        for bot in world.bots:
            pos, task = position[bot.id], tasks.get(bot.id)
            here = trucks.get(load_owner.get(pos))
            if here is not None and here.state == "docked" and here.order and not here.order.complete \
                    and (LOAD_ANYWHERE or task is None or task.truck_id in (None, here.id)) \
                    and any(here.order.needed()[i] > 0 for i in bot.inventory):
                actions[bot.id] = {"bot": bot.id, "action": "load"}
                continue
            if task is not None and task.picks and len(bot.inventory) < INVENTORY_CAPACITY:
                item_type, cell = task.picks[0]
                if pos == cell:
                    actions[bot.id] = {"bot": bot.id, "action": "pick_up", "item_id": self.tours.item_at[(cell, item_type)]}
                    continue
                goal[bot.id], rank[bot.id] = (cell, HOLD_PICK), 1 if task.truck_id is not None else 2
            elif task is not None and task.truck_id is not None:
                occupied = set(position.values()) - {pos}
                goal[bot.id], rank[bot.id] = (self._load_cell(pos, trucks[task.truck_id], occupied, task.load_cell), HOLD_LOAD), 0
            else:
                goal[bot.id], rank[bot.id] = (pos, self.router.window), 3

        for bot_id, task in tasks.items():
            truck = trucks.get(task.truck_id) if task is not None and task.truck_id is not None else None
            if bot_id in rank and truck is not None and truck.state == "docked":
                rank[bot_id] = -1   # the truck sits idle until its items arrive (Expert +9/game in simulation)

        if self.parking is None:
            self.parking = self._parking_cells(set(load_owner))
            pickup = sorted({c for cells in self.smap.access.values() for c in cells})
            loads = sorted(load_owner)
            d = self.tours.dist
            self.central = {c: sum(d[c][x] for x in pickup) / len(pickup) + sum(d[c][x] for x in loads) / len(loads)
                            for c in self.parking}
            load_rows = {self.smap.cells[c][1] for c in load_owner}
            lane = {c for c, (x, y) in enumerate(self.smap.cells) if y in load_rows}
            params = GUIDANCE if GUIDANCE is not None else guidance.load(self.smap)
            if params is not None:
                self.router.set_tolls(lane, max(0.0, params.get(("lane",), 0.0)), LANE_RADIUS,
                                      guidance.edge_tolls(self.smap, params))
            else:
                # with no more bots than trucks the row is quiet and the detour only costs (Medium: -7/game)
                toll = LANE_TOLL if len(world.bots) > len(world.trucks) else 0
                self.router.set_tolls(lane, toll, LANE_RADIUS)
        claimed: set[int] = set()
        spawn = index.get(self.smap.spawn)
        for bot_id in sorted(b for b, r in rank.items() if r == 3):
            pos = position[bot_id]
            free = [c for c in self.parking if c not in claimed or c == spawn]
            if PARK_WEIGHT and len(world.bots) >= PARK_MIN_BOTS_PER_TRUCK * len(world.trucks):
                # park where the next job is likely near: weigh the walk there against the cell's mean
                # distance to the shelves and trucks (bots stayed on the spawn corner all game)
                free = [c for c in free if c != spawn] or free
                target = min(free, key=lambda c: (self.tours.dist[pos][c] + PARK_WEIGHT * self.central[c], c))
            else:
                target = pos if pos in free else min(free, key=lambda c: (self.tours.dist[pos][c], c))
            claimed.add(target)
            goal[bot_id] = (target, self.router.window)

        order = sorted(goal, key=lambda b: (rank[b] * AGING - self.waited.get(b, 0), self.tours.dist[position[b]][goal[b][0]], b))
        fixed = {b: position[b] for b in actions}
        paths, modes = route(self.router, order, goal, position, fixed, started + ROUTE_BUDGET_S)

        self.trace = []
        for bot_id in order:
            pos = position[bot_id]
            target = goal[bot_id][0]
            path, mode = paths[bot_id], modes[bot_id]
            nxt = path[1] if len(path) > 1 else pos
            if target != pos and self._frozen(bot_id):
                # stuck in place, or swinging between two cells with another bot doing the same
                # (a head-on meeting in an aisle, both stepping in and back every round: 21
                # points in an online Medium game); a seeded random step or wait breaks the symmetry
                free = [c for c in self.smap.neighbours[pos] if self.router.can_enter(bot_id, pos, c, 0)]
                options = free + ([pos] if nxt != pos else [])
                if options:
                    nxt, mode = self.rng.choice(options), "unstick"
                    self.history[bot_id].clear()
            actions[bot_id] = {"bot": bot_id, "action": self.move_name[(pos, nxt)]} if nxt != pos \
                else {"bot": bot_id, "action": "wait"}
            self.trace.append({"bot": bot_id, "mode": mode, "goal": target, "path": path[:4]})

        for bot_id, pos in position.items():
            history = self.history.setdefault(bot_id, deque(maxlen=FROZEN_LIMIT))
            busy = actions[bot_id]["action"] in ("load", "pick_up")
            history.append(-1 if busy else pos)
            waiting = actions[bot_id]["action"] == "wait" and bot_id in goal and goal[bot_id][0] != pos
            self.waited[bot_id] = self.waited.get(bot_id, 0) + 1 if waiting else 0
        return [actions[b] for b in sorted(actions)]

    def _frozen(self, bot_id: int) -> bool:
        h = self.history.get(bot_id)
        return h is not None and len(h) == FROZEN_LIMIT and len(set(h)) <= (2 if SWING_UNSTICK else 1) and -1 not in h

    def _load_cell(self, pos: int, truck, occupied: set[int], preferred: int) -> int:
        cells = [self.smap.index[c] for c in truck.load_cells]
        if pos in cells:
            return pos
        free = [c for c in cells if c not in occupied] or cells
        dist = self.tours.dist[pos]
        return min(free, key=lambda c: (dist[c], c != preferred))

    def _parking_cells(self, load_cells: set[int]) -> list[int]:
        """Open cells (3+ walkable neighbours) that are neither pickup cells nor
        on or next to the load row (every bot passes along it), plus the spawn cell."""
        pickup = {c for cells in self.smap.access.values() for c in cells}
        load_rows = {self.smap.cells[c][1] for c in load_cells}
        row = {c for c, (x, y) in enumerate(self.smap.cells) if y in load_rows}
        near_load = row | {n for c in row for n in self.smap.neighbours[c]}
        cells = [c for c in range(len(self.smap.cells))
                 if c not in pickup and c not in near_load and len(self.smap.neighbours[c]) >= 3]
        spawn = self.smap.index.get(self.smap.spawn)
        return cells + ([spawn] if spawn is not None and spawn not in cells else [])
