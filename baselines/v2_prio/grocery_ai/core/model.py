"""Static store model: walkable cells, shelves, pickup cells and all-pairs distances.

The store layout is fixed per difficulty and items never run out, so everything
here is computed once per game from the first observation.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]
MOVES: dict[str, Cell] = {"move_up": (0, -1), "move_down": (0, 1), "move_left": (-1, 0), "move_right": (1, 0)}
UNREACHABLE = np.iinfo(np.int16).max


@dataclass
class StaticMap:
    width: int
    height: int
    walls: frozenset[Cell]
    shelves: dict[Cell, str]                 # shelf cell -> item type
    item_cells: dict[str, Cell]              # item id -> shelf cell
    item_types: dict[str, str]               # item id -> item type
    spawn: Cell
    cells: list[Cell] = field(default_factory=list)          # walkable cells, index order
    index: dict[Cell, int] = field(default_factory=dict)
    neighbours: list[list[int]] = field(default_factory=list)
    dist: np.ndarray = field(default=None)                   # int16 [n, n]
    access: dict[str, list[int]] = field(default_factory=dict)  # type -> walkable cells beside a shelf of it
    dist_to_type: dict[str, np.ndarray] = field(default_factory=dict)  # type -> int16 [n]

    @classmethod
    def from_state(cls, state: dict, spawn: Cell | None = None) -> "StaticMap":
        grid = state["grid"]
        width, height = grid["width"], grid["height"]
        walls = frozenset((x, y) for x, y in grid["walls"])
        shelves = {tuple(i["position"]): i["type"] for i in state["items"]}
        if spawn is None:
            spawn = (width - 2, height - 2)
        smap = cls(
            width=width,
            height=height,
            walls=walls,
            shelves=shelves,
            item_cells={i["id"]: tuple(i["position"]) for i in state["items"]},
            item_types={i["id"]: i["type"] for i in state["items"]},
            spawn=spawn,
        )
        smap._build()
        return smap

    def walkable(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.walls and cell not in self.shelves

    def _build(self) -> None:
        self.cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.walkable((x, y))]
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.neighbours = [
            [self.index[(x + dx, y + dy)] for dx, dy in MOVES.values() if (x + dx, y + dy) in self.index]
            for x, y in self.cells
        ]
        n = len(self.cells)
        self.dist = np.full((n, n), UNREACHABLE, dtype=np.int16)
        for source in range(n):
            row = self.dist[source]
            row[source] = 0
            queue = deque([source])
            while queue:
                cur = queue.popleft()
                step = row[cur] + 1
                for nxt in self.neighbours[cur]:
                    if row[nxt] == UNREACHABLE:
                        row[nxt] = step
                        queue.append(nxt)

        by_type: dict[str, set[int]] = {}
        for (x, y), item_type in self.shelves.items():
            for dx, dy in MOVES.values():
                i = self.index.get((x + dx, y + dy))
                if i is not None:
                    by_type.setdefault(item_type, set()).add(i)
        self.access = {t: sorted(s) for t, s in by_type.items()}
        self.dist_to_type = {t: self.dist[:, idx].min(axis=1) for t, idx in self.access.items()}

    def distance(self, a: Cell, b: Cell) -> int:
        return int(self.dist[self.index[a], self.index[b]])
