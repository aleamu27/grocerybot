"""Paired SPRT: is the new bot really better than the old one, or is it noise?

Both bots play the same (store, seed) games on the official local engine via
`grocerybot practice`. Per seed d = new - old. After a pilot of 20 pairs fixes
the variance, a Gaussian SPRT tests H0: mean(d) = 0 against H1: mean(d) = +delta,
and a mirrored one against -delta catches regressions. A sign test (wins vs
losses, p = 0.5 vs 0.6, both directions) runs alongside, because a few
deadlocked games give d heavy tails and the sign test then decides far sooner.
Either test may stop the run, but only if the other points the same way.

    python3 tools/sprt.py --store medium hard --old "python3 -u bot.py" --new "python3 -u bot_v2.py"

Seeds 20000 and up are the holdout; they are refused unless --holdout is passed.
"""

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HOLDOUT_START = 20000
PILOT = 20


def play(store: str, seed: int, cmd: str) -> dict:
    out = subprocess.run(
        ["grocerybot", "--json", "--no-input", "practice", "--store", store, "--seed", str(seed), "--", *shlex.split(cmd)],
        capture_output=True, text=True, timeout=1800,
    )
    try:
        data = json.loads(out.stdout)["data"]
    except (json.JSONDecodeError, KeyError):
        raise RuntimeError(f"{store} seed {seed} `{cmd}` failed:\n{out.stdout[-800:]}\n{out.stderr[-800:]}")
    diag = data.get("diagnostics", {})
    if diag.get("timeouts") or diag.get("errors"):
        print(f"  warning: {store} seed {seed} `{cmd}`: {diag}", file=sys.stderr)
    return data


class Sprt:
    def __init__(self, alpha: float, beta: float, p_sign: float):
        self.upper = math.log((1 - beta) / alpha)
        self.lower = math.log(beta / (1 - alpha))
        self.p_sign = p_sign

    def gaussian(self, d: list[float], delta: float, sigma2: float) -> float:
        return sum(delta / sigma2 * (x - delta / 2) for x in d)

    def sign(self, d: list[float], direction: int = 1) -> float:
        """LLR of P(win) = p_sign against 0.5, ties ignored; direction -1 tests P(loss) instead."""
        wins, losses = sum(x * direction > 0 for x in d), sum(x * direction < 0 for x in d)
        return wins * math.log(self.p_sign / 0.5) + losses * math.log((1 - self.p_sign) / 0.5)


def run_store(store: str, args, sprt: Sprt) -> str:
    seeds = iter(range(args.seed_start, 10**9))
    old_scores, d = [], []
    sigma2 = delta = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(d) < args.max_pairs:
            batch = [next(seeds) for _ in range(min(args.workers, args.max_pairs - len(d)))]
            if not args.holdout and batch[-1] >= HOLDOUT_START:
                return "inconclusive (reached holdout seeds)"
            olds = list(pool.map(lambda s: play(store, s, args.old), batch))
            news = list(pool.map(lambda s: play(store, s, args.new), batch))
            for o, n in zip(olds, news):
                old_scores.append(o["score"])
                d.append(n["score"] - o["score"])
            if len(d) < PILOT:
                continue
            if sigma2 is None:
                sigma2 = max(statistics.pvariance(d), 1.0)
                delta = max(args.delta_pct / 100 * statistics.mean(old_scores), args.min_delta)
            up = sprt.gaussian(d, delta, sigma2)
            down = sprt.gaussian(d, -delta, sigma2)
            sign_up, sign_down = sprt.sign(d, 1), sprt.sign(d, -1)
            mean = statistics.mean(d)
            wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
            print(f"  {store:9} n={len(d):3} mean d={mean:+7.2f} (delta {delta:.2f}, sd {math.sqrt(sigma2):.2f}) "
                  f"LLR up={up:+6.2f} down={down:+6.2f} sign up={sign_up:+6.2f} down={sign_down:+6.2f} "
                  f"W/L={wins}/{losses}", flush=True)
            # Either test may decide, provided the other points the same way.
            if (down >= sprt.upper and losses >= wins) or (sign_down >= sprt.upper and mean < 0):
                return f"REGRESSION  mean d {mean:+.2f}, W/L {wins}/{losses} over {len(d)} seeds"
            if (up >= sprt.upper and wins >= losses) or (sign_up >= sprt.upper and mean > 0):
                return f"IMPROVEMENT mean d {mean:+.2f}, W/L {wins}/{losses} over {len(d)} seeds"
            if up <= sprt.lower and down <= sprt.lower:
                return f"NO CHANGE   mean d {statistics.mean(d):+.2f} over {len(d)} seeds (|effect| < delta)"
    return f"inconclusive after {len(d)} pairs, mean d {statistics.mean(d):+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", nargs="+", default=["easy", "medium", "hard", "expert", "nightmare"])
    parser.add_argument("--old", required=True, help="baseline bot command")
    parser.add_argument("--new", required=True, help="candidate bot command")
    parser.add_argument("--delta-pct", type=float, default=1.0, help="smallest effect worth detecting, %% of baseline mean")
    parser.add_argument("--min-delta", type=float, default=1.0, help="floor for delta in points")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--p-sign", type=float, default=0.6)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--holdout", action="store_true", help="allow seeds >= 20000")
    args = parser.parse_args()
    if args.holdout and args.seed_start < HOLDOUT_START:
        args.seed_start = HOLDOUT_START

    sprt = Sprt(args.alpha, args.beta, args.p_sign)
    verdicts = {store: run_store(store, args, sprt) for store in args.store}
    print()
    for store, verdict in verdicts.items():
        print(f"{store:9} {verdict}")
    return 1 if any(v.startswith("REGRESSION") for v in verdicts.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
