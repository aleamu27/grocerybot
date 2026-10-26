"""Grocery Bot v2: tour auction planner (with rollouts on small stores) + controller.

Set GROCERY_TRACE=path to write one JSON line per round with each bot's plan.
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grocery_ai.control.controller import Controller  # noqa: E402
from grocery_ai.core.model import StaticMap  # noqa: E402
from grocery_ai.core.state import World  # noqa: E402
from grocery_ai.plan.rollout import RolloutPlanner  # noqa: E402


def main() -> None:
    planner = controller = None
    trace = open(os.environ["GROCERY_TRACE"], "w") if os.environ.get("GROCERY_TRACE") else None
    for line in sys.stdin:
        message = json.loads(line)
        if message["type"] == "hello":
            print(json.dumps({"type": "ready", "protocol_version": 1}), flush=True)
        elif message["type"] == "game_state":
            started = time.perf_counter()
            actions = []
            try:
                if planner is None:
                    smap = StaticMap.from_state(message)
                    planner = RolloutPlanner(smap)
                    controller = Controller(smap, planner.tours)
                world = World.from_state(message)
                actions = controller.act(world, planner.plan(world))
            except Exception:
                traceback.print_exc(file=sys.stderr)
            print(json.dumps({"round": message["round"], "actions": actions}), flush=True)
            if trace:
                trace.write(json.dumps({"round": message["round"], "ms": round((time.perf_counter() - started) * 1000, 2),
                                        "plan": planner.trace if planner else [],
                                        "control": controller.trace if controller else [], "actions": actions}) + "\n")
        elif message["type"] in ("game_over", "game_end"):
            break
    if trace:
        trace.close()


if __name__ == "__main__":
    main()
