"""Random-action bot that exercises edge cases for engine fidelity checks.

Mostly random moves (head-on swaps, blocked moves, spawn stacking), plus pickups
of random items (often not adjacent or with a full inventory) and loads anywhere.
A small bias toward load cells and shelves keeps scoring events in the trace.
"""

import json
import random
import sys

ACTIONS = ["move_up", "move_down", "move_left", "move_right", "wait"]
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "hello":
        print(json.dumps({"type": "ready", "protocol_version": 1}), flush=True)
    elif message["type"] == "game_state":
        items = message["items"]
        load_cells = {tuple(c) for t in message["trucks"] for c in t["load_cells"]}
        actions = []
        for bot in message["bots"]:
            x, y = bot["position"]
            near = [i for i in items if abs(i["position"][0] - x) + abs(i["position"][1] - y) == 1]
            roll = rng.random()
            if (x, y) in load_cells and roll < 0.6 or roll < 0.05:
                actions.append({"bot": bot["id"], "action": rng.choice(["load", "drop_off"])})
            elif near and roll < 0.5 or roll < 0.1:
                item = rng.choice(near or items)
                actions.append({"bot": bot["id"], "action": "pick_up", "item_id": item["id"]})
            elif roll < 0.12:
                actions.append({"bot": bot["id"], "action": "pick_up", "item_id": "item_missing"})
            else:
                bias = ["move_up"] * 2 if bot["inventory"] else []
                actions.append({"bot": bot["id"], "action": rng.choice(ACTIONS + bias)})
        if rng.random() < 0.05:
            actions.append({"bot": 0, "action": "move_left"})  # duplicate entry must be dropped
        rng.shuffle(actions)
        print(json.dumps({"round": message["round"], "actions": actions}), flush=True)
    elif message["type"] == "game_over":
        break
