"""Proxy a bot process and record every observation with the bot's response.

Usage (as the bot command of the CLI):
    grocerybot practice --store medium --seed 7 -- \
        python3 -u tools/record.py --out recordings/medium-7.jsonl -- python3 -u bot.py

Each output line is {"state": <game_state>, "response": <bot reply>} or the final
{"game_over": ...}. The bot's stderr passes through unchanged.
"""

import argparse
import json
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("bot", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bot_cmd = args.bot[1:] if args.bot[:1] == ["--"] else args.bot

    bot = subprocess.Popen(bot_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    with open(args.out, "w") as out:
        for line in sys.stdin:
            message = json.loads(line)
            bot.stdin.write(line if line.endswith("\n") else line + "\n")
            bot.stdin.flush()
            if message["type"] in ("game_over", "game_end"):
                out.write(json.dumps({"game_over": message}) + "\n")
                break
            reply = bot.stdout.readline()
            if message["type"] == "game_state":
                out.write(json.dumps({"state": message, "response": json.loads(reply)}) + "\n")
            sys.stdout.write(reply)
            sys.stdout.flush()
    try:
        bot.stdin.close()
        bot.wait(timeout=0.4)
    except (subprocess.TimeoutExpired, BrokenPipeError):
        bot.kill()


if __name__ == "__main__":
    main()
