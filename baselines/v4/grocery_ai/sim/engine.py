"""Exact rules engine: step(world, actions) -> next world.

Only two things are hidden from bots: each truck trip's duration and the next
order. Those come from an Oracle, so the same engine serves fidelity checks
(oracle reads the recording) and rollouts (oracle samples or holds a guess).
"""

import copy
from collections import Counter
from typing import Protocol

from grocery_ai.core.model import MOVES, StaticMap
from grocery_ai.core.state import Order, Truck, World

INVENTORY_CAPACITY = 3
ORDER_BONUS = 5


class Oracle(Protocol):
    def trip_duration(self, truck: Truck, round: int) -> int: ...
    def next_order(self, truck: Truck, round: int) -> Order | None: ...


class Engine:
    def __init__(self, smap: StaticMap, oracle: Oracle):
        self.smap = smap
        self.oracle = oracle

    def step(self, world: World, actions: list[dict]) -> World:
        """Apply one round. `actions` uses the protocol format."""
        nxt = copy.deepcopy(world)
        chosen: dict[int, dict] = {}
        for a in actions[: len(nxt.bots)]:
            bot_id = a.get("bot")
            if isinstance(bot_id, int) and 0 <= bot_id < len(nxt.bots) and bot_id not in chosen:
                chosen[bot_id] = a

        occupied = Counter(b.pos for b in nxt.bots)
        for bot in sorted(nxt.bots, key=lambda b: b.id):
            act = chosen.get(bot.id, {"action": "wait"})
            name = act.get("action")
            if name in MOVES:
                dx, dy = MOVES[name]
                target = (bot.pos[0] + dx, bot.pos[1] + dy)
                if self.smap.walkable(target) and (occupied[target] == 0 or target == self.smap.spawn):
                    occupied[bot.pos] -= 1
                    occupied[target] += 1
                    bot.pos = target
            elif name == "pick_up":
                self._pick_up(bot, act.get("item_id"))
            elif name in ("load", "drop_off"):
                self._load(nxt, bot)

        for truck in nxt.trucks:
            self._advance(truck, nxt.round)
        nxt.round += 1
        return nxt

    def _pick_up(self, bot, item_id) -> None:
        cell = self.smap.item_cells.get(item_id)
        if cell is None or len(bot.inventory) >= INVENTORY_CAPACITY:
            return
        if abs(cell[0] - bot.pos[0]) + abs(cell[1] - bot.pos[1]) != 1:
            return
        bot.inventory.append(self.smap.item_types[item_id])

    def _load(self, world: World, bot) -> None:
        truck = next((t for t in world.trucks if bot.pos in t.load_cells), None)
        if truck is None or truck.state != "docked" or truck.order is None or truck.order.complete:
            return
        needed = truck.order.needed()
        kept = []
        for item in bot.inventory:
            if needed[item] > 0:
                needed[item] -= 1
                truck.order.delivered.append(item)
                world.score += 1
            else:
                kept.append(item)
        bot.inventory = kept
        if not truck.order.needed():
            truck.order.complete = True
            world.score += ORDER_BONUS
            world.orders_completed += 1

    def _advance(self, truck: Truck, round: int) -> None:
        if truck.state == "docked":
            if truck.order is not None and truck.order.complete:
                truck.state = "leaving"
                truck.rounds_until_back = self.oracle.trip_duration(truck, round)
            elif truck.order is None:
                truck.order = self.oracle.next_order(truck, round)
        elif truck.state == "leaving":
            truck.state = "away"
            truck.rounds_until_back -= 1
            truck.order = self.oracle.next_order(truck, round)
        elif truck.state == "away":
            truck.rounds_until_back -= 1
            if truck.rounds_until_back <= 1:
                truck.state = "arriving"
        elif truck.state == "arriving":
            truck.rounds_until_back -= 1
            truck.state = "docked"
