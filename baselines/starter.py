"""Grocery Bot starter. Stdout is JSON; send diagnostic logs to stderr."""

import json
import sys
from collections import deque

INVENTORY_CAPACITY = 3
MOVES = {(0, -1): "move_up", (0, 1): "move_down", (-1, 0): "move_left", (1, 0): "move_right"}


def needed_items(order: dict) -> dict[str, int]:
    needed: dict[str, int] = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def blocked_cells(state: dict) -> set[tuple[int, int]]:
    """Walls and shelves. A cell holding an item is a shelf, and shelves block."""
    blocked = {(x, y) for x, y in state["grid"]["walls"]}
    blocked |= {tuple(item["position"]) for item in state["items"]}
    return blocked


def bfs(start: tuple[int, int], goals: set, blocked: set, width: int, height: int) -> list | None:
    """Shortest path from start to any goal, as a list of cells. None if unreachable."""
    if start in goals:
        return [start]

    queue = deque([start])
    came_from = {start: None}

    while queue:
        cell = queue.popleft()
        x, y = cell
        for dx, dy in MOVES:
            nxt = (x + dx, y + dy)
            if nxt in came_from or nxt in blocked:
                continue
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            came_from[nxt] = cell
            if nxt in goals:
                path = [nxt]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])
                return path[::-1]
            queue.append(nxt)

    return None


def step_toward(bot_id: int, path: list) -> dict:
    if not path or len(path) < 2:
        return {"bot": bot_id, "action": "wait"}
    (x, y), (nx, ny) = path[0], path[1]
    return {"bot": bot_id, "action": MOVES[(nx - x, ny - y)]}


def neighbours(cell: tuple[int, int], width: int, height: int) -> set:
    x, y = cell
    return {(x + dx, y + dy) for dx, dy in MOVES if 0 <= x + dx < width and 0 <= y + dy < height}


def nearest_item(pos: tuple, wanted: dict, state: dict, avoid: set, claimed: set) -> tuple:
    """The nearest unclaimed item of a wanted type, and the path to beside it."""
    width, height = state["grid"]["width"], state["grid"]["height"]
    target, best_path = None, None
    for item in state["items"]:
        if item["id"] in claimed or wanted.get(item["type"], 0) <= 0:
            continue
        goals = neighbours(tuple(item["position"]), width, height) - avoid
        path = bfs(pos, goals, avoid, width, height)
        if path and (best_path is None or len(path) < len(best_path)):
            target, best_path = item, path
    return target, best_path


def decide_actions(state: dict) -> list[dict]:
    """Each bot fills one truck, picking the shortest round trip."""
    width, height = state["grid"]["width"], state["grid"]["height"]
    blocked = blocked_cells(state)
    position = {b["id"]: tuple(b["position"]) for b in state["bots"]}

    docked = [t for t in state["trucks"] if t["state"] == "docked" and t["order"]]
    coming = [t for t in state["trucks"] if t["state"] != "docked" and t["order"]]
    need = {t["id"]: needed_items(t["order"]) for t in docked + coming}
    demand = {k: dict(v) for k, v in need.items()}
    cells = {t["id"]: {tuple(c) for c in t["load_cells"]} for t in state["trucks"]}

    def carry(inventory: list[str], truck_id: int) -> int:
        left = dict(need[truck_id])
        hits = 0
        for item in inventory:
            if left.get(item, 0) > 0:
                left[item] -= 1
                hits += 1
        return hits

    # Items already carried are on their way to the truck that suits them best.
    goal: dict[int, int | None] = {}
    for bot in state["bots"]:
        best = max(docked, key=lambda t: carry(bot["inventory"], t["id"]), default=None)
        if best is None or not carry(bot["inventory"], best["id"]):
            best = max(coming, key=lambda t: carry(bot["inventory"], t["id"]), default=None)
        goal[bot["id"]] = None
        if best is not None and carry(bot["inventory"], best["id"]):
            goal[bot["id"]] = best["id"]
            for item in bot["inventory"]:
                if demand[best["id"]].get(item, 0) > 0:
                    demand[best["id"]][item] -= 1

    def to_bay(cell: tuple, truck: dict) -> int:
        return abs(cell[0] - truck["bay"][0]) + cell[1]

    claimed: set[str] = set()
    reserved: set[tuple[int, int]] = set()
    actions = []

    for bot in state["bots"]:
        bot_id = bot["id"]
        pos = position[bot_id]
        inventory = bot["inventory"]
        target = goal[bot_id]

        # Lower IDs move first, so they get right of way: this bot also steers
        # around the next few cells they plan to walk through. Without that, two
        # bots meeting in the one-cell corridor swap detours every round.
        avoid = blocked | reserved | (set(position.values()) - {pos})

        here = next((t for t in docked if pos in cells[t["id"]]), None)
        action, path = None, None

        if here and target == here["id"]:
            action = {"bot": bot_id, "action": "load"}

        if action is None and len(inventory) < INVENTORY_CAPACITY:
            # Docked trucks first; pre-pick for the others when none need anything.
            if target is not None:
                choices = [t for t in docked + coming if t["id"] == target]
            else:
                choices = [t for t in docked if any(v > 0 for v in demand[t["id"]].values())]
                choices = choices or [t for t in coming if any(v > 0 for v in demand[t["id"]].values())]
            best = None
            for truck in choices:
                item, item_path = nearest_item(pos, demand[truck["id"]], state, avoid, claimed)
                if item:
                    trip = len(item_path) + to_bay(tuple(item["position"]), truck)
                    if best is None or trip < best[0]:
                        best = (trip, truck["id"], item, item_path)
            if best:
                _, truck_id, item, path = best
                claimed.add(item["id"])
                demand[truck_id][item["type"]] -= 1
                if len(path) == 1:
                    action = {"bot": bot_id, "action": "pick_up", "item_id": item["id"]}

        if action is None and path is None and target is not None:
            path = bfs(pos, cells[target] - avoid, avoid, width, height)

        if action is None:
            action = step_toward(bot_id, path)
        if path:
            reserved.update(path[1:4])
        actions.append(action)

    return actions


for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "hello":
        print(json.dumps({"type": "ready", "protocol_version": 1}), flush=True)
    elif message["type"] == "game_state":
        print(json.dumps({"round": message["round"], "actions": decide_actions(message)}), flush=True)
    elif message["type"] == "game_over":
        break
