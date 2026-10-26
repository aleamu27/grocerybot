"""Replay recordings through the engine and check every predicted state.

Usage: python3 -m grocery_ai.sim.fidelity recordings/*.jsonl
"""

import json
import sys

from grocery_ai.core.model import StaticMap
from grocery_ai.core.state import Order, Truck, World
from grocery_ai.sim.engine import Engine


class RecordingOracle:
    """Answers the hidden draws from the next recorded observation."""

    def __init__(self):
        self.next_state: World | None = None

    def trip_duration(self, truck: Truck, round: int) -> int:
        return self.next_state.trucks[truck.id].rounds_until_back

    def next_order(self, truck: Truck, round: int) -> Order | None:
        order = self.next_state.trucks[truck.id].order
        return None if order is None else Order(order.id, order.required, [], False)


def load_steps(path: str) -> list[dict]:
    """[{"state", "response"}] from tools/record.py output or a CLI run file
    (~/Library/Application Support/grocerybot/runs/*.jsonl: alternating game_state/actions lines)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if any("state" in r for r in rows):
        return [r for r in rows if "state" in r]
    steps = []
    for row in rows:
        if row.get("type") == "game_state":
            steps.append({"state": row, "response": {"actions": []}})
        elif row.get("type") == "actions" and steps and steps[-1]["state"]["round"] == row["round"]:
            steps[-1]["response"] = row
    return steps


def check(path: str) -> tuple[int, int, list[str]]:
    steps = load_steps(path)
    smap = StaticMap.from_state(steps[0]["state"], spawn=tuple(steps[0]["state"]["bots"][0]["position"]))
    oracle = RecordingOracle()
    engine = Engine(smap, oracle)
    mismatches = []
    for cur, nxt in zip(steps, steps[1:]):
        world = World.from_state(cur["state"])
        expected = World.from_state(nxt["state"])
        oracle.next_state = expected
        actions = cur["response"]["actions"] if nxt["state"]["action_status"] == "ok" else []
        got = engine.step(world, actions).snapshot()
        want = expected.snapshot()
        if got != want:
            diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
            mismatches.append(f"round {world.round}: {diff}")
    return len(steps) - 1, len(mismatches), mismatches


def main(paths: list[str]) -> int:
    failed = 0
    for path in paths:
        steps, bad, mismatches = check(path)
        failed += bad
        print(f"{path}: {steps - bad}/{steps} steps match")
        for line in mismatches[:3]:
            print("   ", line[:600])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
