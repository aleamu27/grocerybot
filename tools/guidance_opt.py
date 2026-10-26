"""Tune the guidance graph of one store with CMA-ES against the simulator.

Needs the `cma` package (not needed by the bot), e.g. in a separate venv:
    python3 -m venv /tmp/optenv && /tmp/optenv/bin/pip install cma numpy
    /tmp/optenv/bin/python tools/guidance_opt.py --store nightmare --generations 40

Every candidate of a generation plays the same seeds (common random numbers),
and the seeds change each generation, so selection does not overfit a few
games. At the end the CMA mean and the best candidates seen are re-scored on
validation seeds (from 100000, untouched by tuning) against no guidance, and
the winner is saved to grocery_ai/route/guidance/ only if it beats it.
Confirm with tools/sprt.py on frozen copies before trusting it.
"""

import argparse
import json
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cma  # noqa: E402

from grocery_ai.core.model import StaticMap  # noqa: E402
from grocery_ai.route import guidance  # noqa: E402
from grocery_ai.sim import selfplay  # noqa: E402

VALIDATION_SEED = 100000


def _score(job):
    store, seed, params = job
    import grocery_ai.control.controller as controller
    controller.GUIDANCE = params
    return selfplay.play(store, seed)


def evaluate(pool, store, names, xs, seeds) -> list[float]:
    jobs = [(store, s, dict(zip(names, map(float, x)))) for x in xs for s in seeds]
    scores = pool.map(_score, jobs)
    k = len(seeds)
    return [statistics.mean(scores[i * k:(i + 1) * k]) for i in range(len(xs))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4, help="games per candidate per generation")
    ap.add_argument("--validation", type=int, default=24, help="validation games per finalist")
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    state = selfplay.start_state(args.store)
    smap = StaticMap.from_state(state)
    names = guidance.param_names(smap)
    lane0 = 1.0 if len(state["bots"]) > len(state["trucks"]) else 0.0   # the hand-set default
    x0 = [lane0 if n == ("lane",) else 0.0 for n in names]
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"bounds": [-1.0, 6.0], "seed": 1, "verbose": -9})
    seen: list[tuple[float, list[float]]] = []
    with Pool(args.workers) as pool:
        for gen in range(args.generations):
            t0 = time.time()
            xs = es.ask()
            seeds = range(gen * args.seeds, (gen + 1) * args.seeds)
            base, *fit = evaluate(pool, args.store, names, [x0] + xs, seeds)
            es.tell(xs, [-f for f in fit])
            seen += [(f - base, list(x)) for f, x in zip(fit, xs)]
            print(f"gen {gen:3} best {max(fit):7.1f} mean {statistics.mean(fit):7.1f} default {base:7.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        finalists = [list(es.result.xfavorite)] + [x for _, x in sorted(seen, reverse=True)[:4]]
        val = range(VALIDATION_SEED, VALIDATION_SEED + args.validation)
        base, *scores = evaluate(pool, args.store, names, [x0] + finalists, val)
    print(f"validation over {args.validation} seeds: default {base:.1f}; finalists "
          + ", ".join(f"{s:.1f}" for s in scores))
    best = max(range(len(finalists)), key=lambda i: scores[i])
    if scores[best] <= base:
        print("no finalist beats the default; nothing saved")
        return
    params = {n: round(max(0.0, float(v)), 3) for n, v in zip(names, finalists[best])}
    path = guidance.save(smap, params, {"store": args.store, "validation_default": base,
                                        "validation_score": scores[best], "generations": args.generations})
    print(f"saved {path}: {json.dumps({repr(k): v for k, v in params.items() if v > 0})}")


if __name__ == "__main__":
    main()
