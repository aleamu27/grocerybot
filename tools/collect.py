"""Record rollout decisions in self-play, for fitting a tour-scoring model (expert iteration).

    python3 tools/collect.py --store hard --games 20 --out data/hard.jsonl

Rollouts here have no time budget (offline), so stores with many bots work too,
just slowly. Each line is one decision: features of every candidate tour
(grocery_ai.plan.features.NAMES) and its mean rollout value. Seeds start at
60000, away from the SPRT and simulation seeds.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grocery_ai.plan.rollout as rollout  # noqa: E402
from grocery_ai.control.controller import Controller  # noqa: E402
from grocery_ai.core.model import StaticMap  # noqa: E402
from grocery_ai.core.state import World  # noqa: E402
from grocery_ai.sim import selfplay  # noqa: E402
from grocery_ai.sim.engine import Engine  # noqa: E402


def game(job) -> list[dict]:
    store, seed, cfg = job
    rollout.ROLLOUT = {store: cfg}
    rollout.ROLLOUT_BUDGET_S = float("inf")
    state = selfplay.start_state(store)
    smap = StaticMap.from_state(state)
    planner = rollout.RolloutPlanner(smap)
    planner.log = []
    controller = Controller(smap, planner.tours)
    engine = Engine(smap, selfplay.UniformOracle(store, sorted(smap.access), seed))
    world = World.from_state(state)
    for _ in range(world.max_rounds):
        world = engine.step(world, controller.act(world, planner.plan(world)))
    for row in planner.log:
        row["seed"] = seed
    return planner.log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=60000)
    ap.add_argument("--rollout", default="4,40,8", help="candidates,horizon,samples")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = tuple(int(x) for x in args.rollout.split(","))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    jobs = [(args.store, s, cfg) for s in range(args.seed_start, args.seed_start + args.games)]
    with Pool(args.workers) as pool, open(args.out, "a") as out:
        for rows in pool.imap_unordered(game, jobs):
            for row in rows:
                out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{len(rows)} decisions from seed {rows[0]['seed'] if rows else '?'}", flush=True)


if __name__ == "__main__":
    main()
