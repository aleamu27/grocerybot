"""Windowed time-space A* (WHCA*) with the engine's exact collision rules.

Bots are planned one at a time in priority order. Each plan is a cell per
round t = 0..window, written into a reservation table that later bots must
respect. Rules match grocery_ai.sim.engine:
- two bots never share a cell at the same t (the spawn cell is exempt);
- two bots never swap cells;
- in the round about to be played (t = 0), a bot may step into a cell another
  bot leaves only if that bot has the lower ID, since moves resolve in ID
  order. Later steps are replanned anyway, so there following is allowed.
Before planning, every bot reserves only its current cell at t = 0. A
higher-priority bot may plan into a cell a lower-priority bot stands on if
that bot has the lower ID (else the ID rule forbids it); the lower-priority
bot then cannot stay and must pull over when its turn comes. (Reserving
t = 1 as well, i.e. assuming unplanned bots stay put, makes every bot plan
"move next round" forever: a livelock.)
"""

import heapq

from grocery_ai.core.model import StaticMap


class Router:
    def __init__(self, smap: StaticMap, dist: list[list[int]], window: int = 12):
        self.smap = smap
        self.dist = dist
        self.window = window
        self.spawn = smap.index.get(smap.spawn)
        self.res: dict[tuple[int, int], int] = {}
        self.expanded = 0
        self.toll = [0] * len(smap.cells)
        self.edge_toll: dict[tuple[int, int], float] = {}
        self.toll_radius = 0
        self._tolled: dict[int, list[int]] = {}

    def set_tolls(self, cells, amount: float, radius: int, edges: dict[tuple[int, int], float] | None = None) -> None:
        """Each round spent on one of `cells` costs `amount` extra, and each
        move a -> b in `edges` its toll (a guidance graph), unless the cell
        entered is within `radius` of the bot's goal. The load row is a
        one-lane corridor, and bots crossing other trucks' load cells jam it."""
        self.toll = [amount if c in cells else 0 for c in range(len(self.smap.cells))]
        self.edge_toll = {e: w for e, w in (edges or {}).items() if w > 0}
        self.toll_radius = radius
        self._tolled = {}

    def _step_cost(self, a: int, b: int, goal: int) -> float:
        if self.dist[b][goal] <= self.toll_radius:
            return 1
        return 1 + self.toll[b] + self.edge_toll.get((a, b), 0)

    def heuristic(self, goal: int) -> list[int]:
        """Toll-aware distance to goal from every cell (plain distance without tolls)."""
        if not any(self.toll) and not self.edge_toll:
            return [row[goal] for row in self.dist]
        h = self._tolled.get(goal)
        if h is None:
            h = [1 << 30] * len(self.smap.cells)
            h[goal] = 0
            heap = [(0, goal)]
            while heap:
                d, n = heapq.heappop(heap)
                if d > h[n]:
                    continue
                for c in self.smap.neighbours[n]:
                    step = d + self._step_cost(c, n, goal)
                    if step < h[c]:
                        h[c] = step
                        heapq.heappush(heap, (step, c))
            self._tolled[goal] = h
        return h

    def reset(self, position: dict[int, int]) -> None:
        self.res = {}
        self.expanded = 0
        for bot, pos in position.items():
            if pos != self.spawn:
                self.res[(pos, 0)] = bot

    def can_enter(self, bot: int, a: int, b: int, t: int) -> bool:
        """May `bot` go from a (at t) to b (at t + 1)?"""
        if b == self.spawn:
            return True
        o1 = self.res.get((b, t + 1))
        if o1 is not None and o1 != bot:
            return False
        if a != b:
            o0 = self.res.get((b, t))
            if o0 is not None and o0 != bot:
                if self.res.get((a, t + 1)) == o0:
                    return False      # swap
                if o0 > bot and t == 0:
                    return False      # o0 resolves after us this round, so b is still occupied
        return True

    def _free(self, bot: int, cell: int, t0: int, t1: int) -> bool:
        if cell == self.spawn:
            return True
        return all(self.res.get((cell, t), bot) == bot for t in range(t0, t1 + 1))

    def plan(self, bot: int, start: int, goal: int, hold: int) -> list[int]:
        """Cells for t = 0.. until the goal is reached and can be held for `hold`
        rounds (or the window ends). Reserves the result. Falls back to waiting."""
        path = self.search(bot, start, goal, hold)
        self.reserve(bot, path, hold if path[-1] == goal else 0)
        return path

    def cost(self, path: list[int], goal: int) -> int:
        """Estimated arrival round: exact if the path reaches the goal, else
        the window plus the remaining distance."""
        if path[-1] == goal:
            return len(path) - 1
        return self.window + self.dist[path[-1]][goal]

    def search(self, bot: int, start: int, goal: int, hold: int,
               vcons: frozenset = frozenset(), econs: frozenset = frozenset()) -> list[int]:
        """Like plan, but reserves nothing, and also avoids the extra vertex
        constraints (cell, t) and edge constraints (a, b, t) (used by CBS).

        Cost is rounds plus tolls (set_tolls); h is the toll-aware distance.
        Ties in f = g + h are broken by a penalty for waiting early (window - t
        per wait): of two equally good plans the one that moves now wins.
        Without that a bot that must make way plans "wait, then go", replans
        the same next round, and never goes."""
        window, neighbours = self.window, self.smap.neighbours
        h = self.heuristic(goal)
        step_cost = self._step_cost
        open_ = [(h[start], 0, h[start], 0, 0, start, None)]
        came: dict[tuple[int, int], tuple[int, int] | None] = {}
        end = None
        while open_:
            _, penalty, _, t, g, c, parent = heapq.heappop(open_)
            if (c, t) in came:
                continue
            came[(c, t)] = parent
            self.expanded += 1
            if c == goal and self._free(bot, c, t, min(t + hold, window)) \
                    and not any((c, u) in vcons for u in range(t, min(t + hold, window) + 1)):
                end = (c, t)
                break
            if t == window:
                end = (c, t)
                break
            for n in neighbours[c] + [c]:
                if (n, t + 1) in came or (n, t + 1) in vcons or (c, n, t) in econs \
                        or not self.can_enter(bot, c, n, t):
                    continue
                step_penalty = penalty + (window - t if n == c else 0)
                gn = g + step_cost(c, n, goal)
                heapq.heappush(open_, (gn + h[n], step_penalty, h[n], t + 1, gn, n, (c, t)))

        if end is None:
            path = [start, start]
        else:
            node, path = end, []
            while node is not None:
                path.append(node[0])
                node = came[node]
            path.reverse()
        return path

    def reserve(self, bot: int, path: list[int], hold: int) -> None:
        """Earlier reservations win: a fallback wait that clashes with another
        bot's plan must not erase it (the clash is what CBS looks for)."""
        for t, c in enumerate(path):
            if c != self.spawn:
                self.res.setdefault((c, t), bot)
        last = len(path) - 1
        for t in range(last + 1, min(last + hold, self.window) + 1):
            if path[-1] != self.spawn:
                self.res.setdefault((path[-1], t), bot)
