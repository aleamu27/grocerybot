"""Dynamic game state parsed from a game_state message."""

from collections import Counter
from dataclasses import dataclass, field

Cell = tuple[int, int]


@dataclass
class Order:
    id: str
    required: tuple[str, ...]
    delivered: list[str] = field(default_factory=list)
    complete: bool = False

    def needed(self) -> Counter:
        return Counter(self.required) - Counter(self.delivered)


@dataclass
class Bot:
    id: int
    pos: Cell
    inventory: list[str]


@dataclass
class Truck:
    id: int
    bay: Cell
    load_cells: frozenset[Cell]
    state: str               # docked | leaving | away | arriving
    rounds_until_back: int
    order: Order | None


@dataclass
class World:
    round: int
    max_rounds: int
    score: int
    orders_completed: int
    bots: list[Bot]
    trucks: list[Truck]

    @classmethod
    def from_state(cls, state: dict) -> "World":
        def order(o: dict | None) -> Order | None:
            if o is None:
                return None
            return Order(o["id"], tuple(o["items_required"]), list(o["items_delivered"]), o["complete"])

        return cls(
            round=state["round"],
            max_rounds=state["max_rounds"],
            score=state["score"],
            orders_completed=state["orders_completed"],
            bots=[Bot(b["id"], tuple(b["position"]), list(b["inventory"])) for b in state["bots"]],
            trucks=[
                Truck(
                    t["id"],
                    tuple(t["bay"]),
                    frozenset(tuple(c) for c in t["load_cells"]),
                    t["state"],
                    t["rounds_until_back"],
                    order(t["order"]),
                )
                for t in state["trucks"]
            ],
        )

    def snapshot(self) -> dict:
        """Comparable summary; multisets are sorted so list order never matters."""
        return {
            "round": self.round,
            "score": self.score,
            "orders_completed": self.orders_completed,
            "bots": [(b.id, b.pos, sorted(b.inventory)) for b in self.bots],
            "trucks": [
                (
                    t.id,
                    t.state,
                    t.rounds_until_back,
                    None if t.order is None else (t.order.id, sorted(t.order.delivered), t.order.complete),
                )
                for t in self.trucks
            ],
        }
