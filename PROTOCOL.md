# Grocery Bot process protocol

Keep one bot process alive per game. The CLI sends newline-delimited UTF-8 JSON on stdin. Write one JSON object per line to stdout and flush each response. Logs go to stderr. No networking, credentials or game-engine imports are needed in the bot.

First the CLI sends:

```json
{"type":"hello","protocol_version":1}
```

Reply within 10 seconds:

```json
{"type":"ready","protocol_version":1}
```

The CLI then sends public game states and waits for actions. It starts a new process for each benchmark case. Output is bounded to 64 KiB per line; emit no banners, extra lines, NaN or Infinity.

On completion the CLI sends `{"type":"game_over","score":0,"rounds_used":300,"items_delivered":0,"orders_completed":0,"end_reason":"completed"}` with the actual result. Online game results also include `game_id` and `run_id`. A run without a confirmed result instead receives `{"type":"game_end","status":"unconfirmed","run_id":"..."}`; void or interrupted runs may use those statuses. No score is claimed for an unconfirmed run.

The CLI then closes stdin. No reply is required. Save any bot diagnostics and exit promptly; the CLI allows 0.5 seconds before terminating the owned process tree. A crash or forced cancellation may only deliver EOF, so always handle it.

## Game state message

```json
{
  "type": "game_state",
  "round": 42,
  "max_rounds": 300,
  "action_status": "ok",
  "grid": {"width": 16, "height": 12, "walls": [[0, 0], [1, 0], [2, 0]]},
  "bots": [{"id": 0, "position": [4, 1], "inventory": ["milk"]}],
  "items": [{"id": "item_0", "type": "milk", "position": [2, 2], "department": "dairy", "fixture": "fridge"}],
  "trucks": [
    {
      "id": 0,
      "bay": [5, 0],
      "load_cells": [[4, 1], [5, 1], [6, 1]],
      "state": "docked",
      "rounds_until_back": 0,
      "order": {
        "id": "order_3",
        "items_required": ["milk", "butter", "bread"],
        "items_delivered": ["butter"],
        "complete": false
      }
    },
    {
      "id": 1,
      "bay": [9, 0],
      "load_cells": [[8, 1], [9, 1], [10, 1]],
      "state": "away",
      "rounds_until_back": 7,
      "order": {
        "id": "order_4",
        "items_required": ["rice", "apples", "flour", "bananas"],
        "items_delivered": [],
        "complete": false
      }
    }
  ],
  "score": 31,
  "orders_completed": 3,
  "total_orders": 50
}
```

| Field | Type | Description |
|---|---|---|
| `round` | int | Current round number (0-indexed) |
| `max_rounds` | int | Round cap (300, or 500 on Nightmare) |
| `action_status` | string | Result of your last response: `"ok"`, `"timeout"` or `"error"` |
| `grid.width` | int | Grid width in cells |
| `grid.height` | int | Grid height in cells |
| `grid.walls` | int[][] | `[x, y]` positions of all walls |
| `bots` | object[] | Every bot with `id`, `position` `[x, y]` and `inventory` |
| `items` | object[] | Every item with `id`, `type`, `position`, `department` and `fixture` |
| `trucks` | object[] | Every truck, in ID order |
| `trucks[].bay` | int[] | `[x, y]` of the bay in the back wall. It is listed in `grid.walls` |
| `trucks[].load_cells` | int[][] | Cells a bot can load this truck from. Walkable |
| `trucks[].state` | string | `"docked"`, `"leaving"`, `"away"` or `"arriving"`. Only a docked truck can be loaded |
| `trucks[].rounds_until_back` | int | Exact rounds until the truck is docked again, including leaving/arriving; `0` when docked. Each departure draws a new seeded trip duration; the next duration is not shown before departure |
| `trucks[].order` | object | While leaving: the order it just completed. Otherwise: the order it is loading, or will load when it is back. `null` only if the order pool is momentarily empty |
| `score` | int | Current score |
| `orders_completed` | int | Trucks filled so far |
| `total_orders` | int | Orders generated so far. It grows as the pool tops itself up, so do not treat it as a finish line |

`department` is `dairy`, `bakery`, `produce` or `pantry`. `fixture` is `fridge`, `rack`, `bin` or `shelf`; this describes appearance, and every pickup still takes one round.

Shelves are not listed in `grid.walls` — a cell holding an item is a shelf, and shelves are not walkable.

### `action_status`

| Value | Meaning |
|---|---|
| `"ok"` | Your actions were received and applied |
| `"timeout"` | You did not respond within the timeout — all bots waited |
| `"error"` | Your message could not be parsed, or its `round` did not match — all bots waited |

On round 0 it is always `"ok"`. Repeated `"timeout"` means your bot is too slow; `"error"` means your JSON is wrong.

## Bot response

Send within the round deadline (2 seconds by default; hosted services may configure it):

```json
{
  "round": 42,
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "load"}
  ]
}
```

The CLI requires `round` to be an integer matching the observation. Responses for older rounds are discarded. Include `actions`, even when it is empty. This is stricter than the legacy direct WebSocket interface.

The `bot` field is authoritative — the list index is ignored, so `[{"bot": 2, ...}]` is a valid message that moves only bot 2. Each bot gets **at most one action per round**: entries past the first for a given bot are dropped, as are ids outside `0 … bots-1` and anything past the first `bots` entries in the list.

### Actions

| Action | Extra fields | Description |
|---|---|---|
| `move_up` | — | Move one cell up (y−1) |
| `move_down` | — | Move one cell down (y+1) |
| `move_left` | — | Move one cell left (x−1) |
| `move_right` | — | Move one cell right (x+1) |
| `pick_up` | `item_id` | Pick up an item from an adjacent shelf |
| `load` | — | Load matching items into the truck whose load cell the bot stands on. `drop_off` is an alias |
| `wait` | — | Do nothing |

### Move rules

- Moves into walls, shelves or out of bounds fail silently (treated as `wait`)
- Moves into a cell occupied by another bot fail silently
- Actions resolve in **bot ID order** — bot 0 moves first, then bot 1, and so on, whatever order you list them in
- The spawn tile (bottom-right) is exempt from collision so bots can start stacked

### Pickup rules

- The bot must be **adjacent** (Manhattan distance 1) to the shelf holding the item
- Inventory must not be full (max 3 items)
- `item_id` must match an item on the map

### Loading rules

`load` means "load the truck whose load cell I am standing on". `drop_off` is accepted as an alias.

- The bot must stand on one of a docked truck's `load_cells`
- Items the truck's order still needs are loaded, +1 each; the rest stay in inventory
- The load that completes the order scores +5 and closes the truck. Bots later in the same round cannot load it
- After all bot actions, trucks advance one step of their cycle. The observation reports the current state and remaining rounds


The store and product locations are fixed per difficulty. Practice seeds vary orders and truck trips. Ranked uses private server-selected scenarios. Your bot receives public observations only, never private seeds or future orders.
