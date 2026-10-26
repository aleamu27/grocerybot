# Grocery Bot (NM i AI)

A bot for [Grocery Bot](https://grocerybot.no/how-to-play): a team of robots picks items from store
shelves and loads them into trucks that come and go, across five stores (Easy, Medium, Hard, Expert,
Nightmare). The bot is hierarchical and measurable instead of learned:

```
global planner (tour auction)  ->  router (WHCA* + local CBS + guidance graph)  ->  controller
```

Every change is gated by a paired SPRT against the previous frozen version, played on an exact local
copy of the game engine.

## Quick start

```sh
grocerybot login
grocerybot play --store easy          # runs the command in grocerybot.toml (bot_v2.py)
grocerybot benchmark --save baseline  # measure across stores
python3 -m unittest discover -s tests -t .
```

The bot itself needs only the Python 3 standard library. Its protocol (one JSON object per line on
stdin/stdout, logs on stderr) is described in [PROTOCOL.md](PROTOCOL.md).

## Layout

| Path | What it is |
| --- | --- |
| `bot_v2.py` | The bot. Entry point configured in `grocerybot.toml`. |
| `bot.py` | The original starter bot (also frozen as `baselines/starter.py`). |
| `grocery_ai/core/` | Static store map with all-pairs distances (`model.py`), parsed game state (`state.py`). |
| `grocery_ai/plan/` | Tour generation (up to three pickups, then one truck's load cell) and the global planner that auctions tours to bots each round. |
| `grocery_ai/route/` | Windowed time-space A* (`whca.py`), local CBS for clusters prioritized planning cannot untangle (`cbs.py`), guidance-graph move tolls (`guidance.py`, tuned files in `guidance/`). |
| `grocery_ai/control/` | Turns tours and routes into one action per bot. |
| `grocery_ai/sim/` | Exact rules engine (`engine.py`), fidelity check against recordings (`fidelity.py`), self-play without the CLI (`selfplay.py`). |
| `tools/` | `sprt.py` (compare bots), `record.py` (record a game), `fuzz_bot.py` (random bot for fidelity), `guidance_opt.py` (CMA-ES tuning). |
| `baselines/` | Frozen copies of every accepted version. |
| `recordings/` | Recorded games. Only `<store>-1.jsonl` is in git; `selfplay.py` reads the store layouts from them. |
| `tests/` | Engine, router and guidance tests. |

## How the bot works

- **Planner:** each round it builds candidate tours for the visible orders and assigns them to bots by
  auction, on distance-based ETAs. Bots load only into their own tour's truck. Bots without a tour may
  stock slow-to-fetch item types on speculation (`STOCK_*` in `plan/planner.py`, off on dense stores).
- **Router:** WHCA* with the engine's exact collision rules. Bots that stay stuck for `CBS_DELAY` rounds
  are planned jointly with their blockers by local CBS. Bots working for a docked truck route first.
  The load row (a one-lane corridor through every truck's load cells) carries a toll, so bots reach
  their truck through its own aisle.
- **Guidance graph:** per-direction move tolls tied to a few layout-derived parameters. Layouts are
  fixed per store, so they can be tuned offline per store. A tuned file is loaded at startup if one
  exists for the layout, otherwise the hand-set load-row toll is used.

## Measuring

```sh
# Paired SPRT (seeds from 1000; 20000+ is the holdout, only with --holdout)
python3 tools/sprt.py --store medium hard --old "python3 -u baselines/v5/bot_v2.py" --new "python3 -u baselines/v6/bot_v2.py"

# Record a game, then check that the local engine reproduces it exactly
grocerybot practice --store medium --seed 7 -- python3 -u tools/record.py --out recordings/m7.jsonl -- python3 -u bot_v2.py
python3 -m grocery_ai.sim.fidelity recordings/*.jsonl

# Per-round trace of every bot's plan, router mode, goal and path
GROCERY_TRACE=trace.jsonl grocerybot practice --store hard --seed 3
```

`grocerybot practice --seed N` (hidden from `--help`) plays any seed. The CLI also keeps every run's
states and actions in `~/Library/Application Support/grocerybot/runs/`, and `fidelity.py` accepts those
files too.

**Always compare frozen copies.** `sprt.py` starts a new bot process per game, and that process imports
`grocery_ai/` from disk, so editing the code during a run changes the candidate. Copy candidate and
baseline into `baselines/<name>/` first.

### Tuning the guidance graph

```sh
python3 -m venv .venv-opt && .venv-opt/bin/pip install cma numpy
.venv-opt/bin/python tools/guidance_opt.py --store nightmare --generations 60 --seeds 6
```

All candidates in a generation play the same seeds, and the seeds change each generation. Finalists are
re-scored on validation seeds from 100000. The result is saved to `grocery_ai/route/guidance/` only if it
beats the default, and **the bot uses it immediately**. Confirm with an SPRT on frozen copies before
keeping it.

## Version history

Each version was accepted by SPRT against the one before (mean score difference per store,
Easy / Medium / Hard / Expert / Nightmare).

| Version | Change | Mean d vs previous |
| --- | --- | --- |
| `v2_prio` | Tour auction + WHCA* router | vs starter: +5 / +46 / +87 / +112 / +126 |
| `v2_cbs` | Local CBS for stuck clusters | 0 / +0.5 / +2.4 / +4.5 / +69 |
| `v3_toll` | Load-row toll | – / – / +1.8 / +11 / +202 |
| `v4` | Speculative stocking (off on dense stores) | – / +2.6 / +2.2 / +17 / (off) |
| `v5` | Load only into your own tour's truck | +2.4 / +2.8 / +7.5 / +3.4 / +17.8 |
| `v6` | Bots for a docked truck route first (current) | 0 / +12.5 / +1.5 / +5.8 / +10.8 |

CLI mean score over seeds 5000–5009: starter 144 / 161 / 150 / 118 / 208, v5 153 / 211 / 241 / 256 / 590.
