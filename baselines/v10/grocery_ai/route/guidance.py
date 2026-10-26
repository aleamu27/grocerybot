"""Guidance graph: per-direction move tolls, tuned offline per store layout.

A store is corridor rows (mostly walkable) crossed by one-cell aisles. A full
edge-weight table has ~900 entries, too many to tune by simulation, so the
weights are tied into a few parameters derived from the layout:
- ("v", x, d): vertical moves in aisle column x, d in up/down;
- ("vc", d): vertical moves between two corridor rows;
- ("h", y, d): horizontal moves along corridor row y, d in left/right;
- ("ha", d): sideways moves between two adjacent aisles;
- ("lane",): toll per round on the load row (the router's cell toll).
tools/guidance_opt.py tunes them with CMA-ES against the simulator (as in
online/offline guidance-graph optimization for lifelong MAPF, arXiv
2411.16506) and saves them under guidance/<layout key>.json; layouts are the
same for every seed of a store, so a file found at startup applies.
"""

import hashlib
import json
from pathlib import Path

from grocery_ai.core.model import StaticMap

DIR = Path(__file__).with_name("guidance")


def layout_key(smap: StaticMap) -> str:
    cells = sorted(smap.cells)
    return hashlib.sha1(json.dumps(cells).encode()).hexdigest()[:12]


def corridor_rows(smap: StaticMap) -> set[int]:
    xs = [x for x, _ in smap.cells]
    width = max(xs) - min(xs) + 1
    per_row: dict[int, int] = {}
    for _, y in smap.cells:
        per_row[y] = per_row.get(y, 0) + 1
    return {y for y, n in per_row.items() if n >= 0.7 * width}


def edge_keys(smap: StaticMap) -> dict[tuple[int, int], tuple]:
    """The parameter each directed move a -> b is tied to."""
    corridor = corridor_rows(smap)
    aisle_cols = {x for x, y in smap.cells if y not in corridor}
    keys = {}
    for a, (x, y) in enumerate(smap.cells):
        for b in smap.neighbours[a]:
            bx, by = smap.cells[b]
            if bx == x:
                d = "down" if by > y else "up"
                keys[(a, b)] = ("vc", d) if y in corridor and by in corridor else \
                    ("v", x, d) if x in aisle_cols else ("vc", d)
            else:
                d = "right" if bx > x else "left"
                keys[(a, b)] = ("h", y, d) if y in corridor else ("ha", d)
    return keys


def param_names(smap: StaticMap) -> list[tuple]:
    return sorted(set(edge_keys(smap).values()), key=repr) + [("lane",)]


def edge_tolls(smap: StaticMap, params: dict[tuple, float]) -> dict[tuple[int, int], float]:
    return {e: max(0.0, params.get(k, 0.0)) for e, k in edge_keys(smap).items() if params.get(k, 0.0) > 0}


def load(smap: StaticMap) -> dict[tuple, float] | None:
    path = DIR / f"{layout_key(smap)}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {tuple(k): v for k, v in data["params"]}


def save(smap: StaticMap, params: dict[tuple, float], meta: dict) -> Path:
    DIR.mkdir(exist_ok=True)
    path = DIR / f"{layout_key(smap)}.json"
    path.write_text(json.dumps({"meta": meta, "params": [[list(k), v] for k, v in sorted(params.items(), key=repr)]},
                               indent=1))
    return path
