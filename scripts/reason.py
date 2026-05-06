from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.reasoning import GAIReasoningRuntime, load_reasoning_profiles


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GAI-1 reasoning runtime scaffold.")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--profiles", default="configs/reasoning_modes.json")
    parser.add_argument("--task", required=True)
    parser.add_argument("--show-private-trace", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    profiles = load_reasoning_profiles(ROOT / args.profiles)
    runtime = GAIReasoningRuntime(level=args.level, profiles=profiles)
    trace = runtime.run(args.task, level=args.level)
    print(json.dumps(trace.to_record(include_private=args.show_private_trace), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
