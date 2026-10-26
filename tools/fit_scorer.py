"""Fit a tour scorer to rollout decisions (tools/collect.py), numpy only.

    python3 tools/fit_scorer.py data/easy.jsonl data/medium.jsonl data/hard.jsonl

Target: a candidate's rollout value minus the mean over its decision's
candidates (its advantage). Models: ridge regression, and a small MLP (one
hidden layer, tanh) trained on the same target. Evaluation is by seed
(games), so no game is in both train and test, and reports regret per
decision: rollout value of the best candidate minus that of the one picked,
for the auction's own pick (rank 0), the ridge and the MLP pick.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grocery_ai.plan.features import NAMES  # noqa: E402


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if len(r["x"]) > 1:
                rows.append(r)
    return rows


def stack(rows):
    X, y, g = [], [], []
    for i, r in enumerate(rows):
        ys = np.array(r["y"])
        X += r["x"]
        y += list(ys - ys.mean())
        g += [i] * len(ys)
    return np.array(X, float), np.array(y), np.array(g)


def regret(rows, score):
    out = []
    for r in rows:
        ys = np.array(r["y"])
        s = score(np.array(r["x"], float))
        out.append(ys.max() - ys[int(np.argmax(s))])
    return float(np.mean(out))


def fit_ridge(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return lambda x: ((x - mu) / sd) @ w, (mu, sd, w)


def fit_mlp(X, y, hidden=32, epochs=300, lr=0.01, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    W1 = rng.normal(0, 1 / np.sqrt(Z.shape[1]), (Z.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), hidden); b2 = 0.0
    params = [W1, b1, W2]
    m = [np.zeros_like(p) for p in params] + [0.0]; v = [np.zeros_like(p) for p in params] + [0.0]
    n = len(Z)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, 256):
            bi = idx[start:start + 256]
            H = np.tanh(Z[bi] @ W1 + b1)
            err = H @ W2 + b2 - y[bi]
            gW2 = H.T @ err / len(bi) + l2 * W2
            gb2 = err.mean()
            dH = np.outer(err, W2) * (1 - H ** 2)
            gW1 = Z[bi].T @ dH / len(bi) + l2 * W1
            gb1 = dH.mean(0)
            t = epoch * (n // 256 + 1) + start // 256 + 1
            for k, (p, gr) in enumerate(zip([W1, b1, W2], [gW1, gb1, gW2])):
                m[k] = 0.9 * m[k] + 0.1 * gr; v[k] = 0.999 * v[k] + 0.001 * gr ** 2
                p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            m[3] = 0.9 * m[3] + 0.1 * gb2; v[3] = 0.999 * v[3] + 0.001 * gb2 ** 2
            b2 -= lr * (m[3] / (1 - 0.9 ** t)) / (np.sqrt(v[3] / (1 - 0.999 ** t)) + 1e-8)
    return lambda x: np.tanh(((x - mu) / sd) @ W1 + b1) @ W2 + b2, (mu, sd, W1, b1, W2, b2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--save", help="write the MLP to this .json")
    args = ap.parse_args()
    rows = load(args.paths)
    by_store = defaultdict(list)
    for r in rows:
        by_store[r["store"]].append(r)
    train, test = [], []
    for store, rs in by_store.items():
        seeds = sorted({r["seed"] for r in rs})
        cut = set(seeds[: max(1, int(len(seeds) * args.test_frac))])
        test += [r for r in rs if r["seed"] in cut]
        train += [r for r in rs if r["seed"] not in cut]
    X, y, _ = stack(train)
    ridge, ridge_p = fit_ridge(X, y)
    mlp, mlp_p = fit_mlp(X, y)
    print(f"train {len(train)} decisions, test {len(test)} (split by game)")
    print(f"{'store':9} {'n':>5} {'oracle gap':>10} {'auction':>8} {'ridge':>8} {'mlp':>8}   (mean regret, rollout points)")
    for store in by_store:
        ts = [r for r in test if r["store"] == store]
        gap = np.mean([np.max(r["y"]) - np.mean(r["y"]) for r in ts])
        print(f"{store:9} {len(ts):5} {gap:10.2f} {regret(ts, lambda x: -x[:, NAMES.index('rank')]):8.2f} "
              f"{regret(ts, ridge):8.2f} {regret(ts, mlp):8.2f}")
    w = ridge_p[2]
    print("ridge weights (standardised):", ", ".join(f"{n} {v:+.2f}" for n, v in sorted(zip(NAMES, w), key=lambda t: -abs(t[1]))[:10]))
    if args.save:
        mu, sd, W1, b1, W2, b2 = mlp_p
        Path(args.save).write_text(json.dumps({"names": NAMES, "mu": mu.tolist(), "sd": sd.tolist(), "W1": W1.tolist(),
                                               "b1": b1.tolist(), "W2": W2.tolist(), "b2": float(b2)}))
        print("saved", args.save)


if __name__ == "__main__":
    main()
