from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.agent import GAI1Agent


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAI-1 agentic reasoning with basic verifier.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--level", default="medium")
    return parser.parse_args()


def main() -> int:
    configure_console()
    args = parse_args()
    result = GAI1Agent().run(args.task, level=args.level)
    print(json.dumps({"final": result.final, "verification": result.verification.__dict__}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

