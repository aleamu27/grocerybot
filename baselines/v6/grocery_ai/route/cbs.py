"""Prioritized planning with local CBS for the clusters it cannot untangle.

route() first plans every bot in priority order with the WHCA* router. A
non-idle bot whose plan arrives CBS_DELAY or more rounds later than its
shortest path is "stuck" (idle bots heading for parking are not: they must
hold the cell for the whole window, so they look late all the time); the bots whose reservations lie on its shortest path are its
blockers. A head-on meeting in a dead end is the typical case: one bot has to
wait or duck into a pocket for the other, and no priority order finds that
(tests/test_router.py). The stuck bot and its blockers are then planned
jointly with conflict-based search (Sharon et al. 2015) against the
reservations of the bots planned before them, and every bot after them is
replanned. The joint plan is kept only if the summed arrival estimate of all
bots improves, with bots heading to a truck weighted LOAD_WEIGHT: unweighted,
CBS readily sends a bot one step from its truck back to let a picker pass.

Conflicts follow the engine rules: two bots in one cell at the same t (the
spawn cell is exempt), swaps, and a bot stepping into a cell that a
higher-ID bot leaves in the same round (moves resolve in ID order, so the
cell is still occupied). The router checks that last rule only at t = 0, as
its plans are redone every round; a joint plan has to hold for the whole
window, or it stalls the next round when the step comes due.
"""

import heapq
import time
from itertools import count

from grocery_ai.route.whca import Router

CBS_DELAY = 6         # rounds behind the shortest path that count as stuck (2 and 4 lost on Expert in simulation)
CBS_MAX_AGENTS = 4
CBS_MAX_NODES = 64
CBS_CLUSTERS = 3      # joint plans tried per round
LOAD_WEIGHT = 8       # arrival weight of bots heading to a truck (hold beyond the window); 1 lost on Nightmare


def route(router: Router, order: list[int], goals: dict[int, tuple[int, int]], position: dict[int, int],
          fixed: dict[int, int], deadline: float) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Paths for the bots in `order` (bot -> (goal, hold)). `fixed` bots
    (loading or picking up) stay on their cell this round. Leaves the router's
    table holding the returned plan."""
    paths, modes = _prioritized(router, order, goals, position, fixed, deadline, {})
    dist = router.dist
    tried: set[int] = set()
    for _ in range(CBS_CLUSTERS):
        if time.perf_counter() > deadline:
            break
        stuck = [b for b in order if b not in tried and modes[b] == "route" and goals[b][1] != router.window
                 and router.cost(paths[b], goals[b][0]) - dist[position[b]][goals[b][0]] >= CBS_DELAY]
        if not stuck:
            break
        bot = stuck[0]
        tried.add(bot)
        cluster = [bot] + [b for b in _blockers(router, bot, position[bot], goals[bot][0])
                           if b not in fixed and modes.get(b) == "route"][:CBS_MAX_AGENTS - 1]
        if len(cluster) < 2:
            continue
        first = min(order.index(b) for b in cluster)
        _base(router, position, fixed)
        for b in order[:first]:
            _reserve(router, b, paths[b], goals[b])
        joint = solve(router, {b: (position[b], *goals[b]) for b in cluster}, CBS_MAX_NODES)
        if joint is None:
            _restore(router, order, goals, position, fixed, paths)
            continue
        for b, path in joint.items():
            _reserve(router, b, path, goals[b])
        rest = [b for b in order[first:] if b not in joint]
        new, new_modes = _prioritized(router, rest, goals, position, fixed, deadline, joint, reset=False)
        new.update({b: paths[b] for b in order[:first]})
        new.update(joint)
        if _total(router, new, goals) < _total(router, paths, goals):
            paths = new
            modes = {**modes, **new_modes, **{b: "cbs" for b in joint}}
            tried.update(joint)
        else:
            _restore(router, order, goals, position, fixed, paths)
    return paths, modes


def solve(router: Router, agents: dict[int, tuple[int, int, int]], max_nodes: int) -> dict[int, list[int]] | None:
    """Conflict-free paths for agents (bot -> (start, goal, hold)) against the
    router's table, minimizing the summed arrival estimate; None if no
    solution within max_nodes expansions."""
    cons = {b: (frozenset(), frozenset()) for b in agents}
    paths = {b: router.search(b, s, g, h) for b, (s, g, h) in agents.items()}
    tie = count()
    open_ = [(_node_cost(router, agents, paths), next(tie), cons, paths)]
    for _ in range(max_nodes):
        if not open_:
            return None
        _, _, cons, paths = heapq.heappop(open_)
        conflict = _first_conflict(router, agents, paths)
        if conflict is None:
            return paths
        for bot, vertex, edge in conflict:
            v, e = cons[bot]
            child = {**cons, bot: (v | vertex, e | edge)}
            start, goal, hold = agents[bot]
            path = router.search(bot, start, goal, hold, *child[bot])
            child_paths = {**paths, bot: path}
            heapq.heappush(open_, (_node_cost(router, agents, child_paths), next(tie), child, child_paths))
    return None


def _weight(router: Router, hold: int) -> int:
    return LOAD_WEIGHT if hold > router.window else 1


def _node_cost(router: Router, agents, paths) -> int:
    return sum(_weight(router, agents[b][2]) * router.cost(paths[b], agents[b][1]) for b in agents)


def _cells(router: Router, path: list[int], goal: int, hold: int) -> list[int | None]:
    """The cell at t = 0..window; None once the plan (and its hold) is over."""
    end = min(len(path) - 1 + (hold if path[-1] == goal else 0), router.window)
    return [path[min(t, len(path) - 1)] if t <= end else None for t in range(router.window + 1)]


def _first_conflict(router: Router, agents, paths):
    """The earliest conflict as two (bot, vertex constraints, edge constraints)
    branches, or None."""
    cells = {b: _cells(router, paths[b], agents[b][1], agents[b][2]) for b in agents}
    bots = sorted(agents)
    for t in range(1, router.window + 1):
        for i, a in enumerate(bots):
            for b in bots[i + 1:]:
                ca, cb = cells[a], cells[b]
                if ca[t] is not None and ca[t] == cb[t] and ca[t] != router.spawn:
                    return [(a, frozenset({(ca[t], t)}), frozenset()), (b, frozenset({(cb[t], t)}), frozenset())]
                if None not in (ca[t - 1], ca[t], cb[t - 1], cb[t]) and ca[t - 1] != ca[t] \
                        and ca[t - 1] == cb[t] and ca[t] == cb[t - 1]:
                    return [(a, frozenset(), frozenset({(ca[t - 1], ca[t], t - 1)})),
                            (b, frozenset(), frozenset({(cb[t - 1], cb[t], t - 1)}))]
                for lo, hi in ((a, b), (b, a)):
                    cl, ch = cells[lo], cells[hi]
                    if None not in (cl[t - 1], cl[t], ch[t - 1]) and cl[t - 1] != cl[t] \
                            and cl[t] == ch[t - 1] != ch[t] and cl[t] != router.spawn:
                        # lo moves first, into the cell hi has not left yet
                        return [(lo, frozenset(), frozenset({(cl[t - 1], cl[t], t - 1)})),
                                (hi, frozenset({(ch[t - 1], t - 1)}), frozenset())]
    return None


def _blockers(router: Router, bot: int, start: int, goal: int) -> list[int]:
    """Bots whose reservations lie on a shortest path from start to goal, in
    the order they are met."""
    res, dist, found = router.res, router.dist, []
    cell = start
    for t in range(1, router.window + 1):
        if cell == goal:
            nxt = cell
        else:
            nxt = min(n for n in router.smap.neighbours[cell] if dist[n][goal] == dist[cell][goal] - 1)
        owners = [res.get((nxt, t))]
        swap = res.get((nxt, t - 1))
        if swap is not None and res.get((cell, t)) == swap:
            owners.append(swap)
        for o in owners:
            if o is not None and o != bot and o not in found and nxt != router.spawn:
                found.append(o)
        if cell == goal and t > 1:
            break
        cell = nxt
    return found


def _prioritized(router: Router, order, goals, position, fixed, deadline, keep, reset=True):
    if reset:
        _base(router, position, fixed)
    paths, modes = {}, {}
    for b in order:
        if b in keep:
            continue
        if time.perf_counter() > deadline:
            paths[b], modes[b] = [position[b], position[b]], "budget"
            router.reserve(b, paths[b], 1)
        else:
            paths[b], modes[b] = router.plan(b, position[b], *goals[b]), "route"
    return paths, modes


def _base(router: Router, position, fixed) -> None:
    router.reset(position)
    for b, cell in fixed.items():
        router.reserve(b, [cell, cell], 1)


def _reserve(router: Router, bot: int, path: list[int], goal: tuple[int, int]) -> None:
    router.reserve(bot, path, goal[1] if path[-1] == goal[0] else 0)


def _restore(router: Router, order, goals, position, fixed, paths) -> None:
    _base(router, position, fixed)
    for b in order:
        _reserve(router, b, paths[b], goals[b])


def _total(router: Router, paths, goals) -> int:
    return sum(_weight(router, goals[b][1]) * router.cost(p, goals[b][0]) for b, p in paths.items())
