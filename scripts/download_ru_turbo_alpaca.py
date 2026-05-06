from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Russian instruction SFT sample from Hugging Face.")
    parser.add_argument("--dataset", default="IlyaGusev/ru_turbo_alpaca")
    parser.add_argument("--filename", default="ru_turbo_alpaca.jsonl.zst")
    parser.add_argument("--out", default="data/raw/ru_turbo_alpaca_sample.jsonl")
    parser.add_argument("--max-records", type=int, default=5000)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_row(row: dict[str, Any]) -> dict[str, str] | None:
    instruction = clean_text(row.get("instruction") or row.get("prompt"))
    input_text = clean_text(row.get("input"))
    output = clean_text(row.get("output") or row.get("response"))
    if input_text == "<noinput>":
        input_text = ""
    if not instruction or not output:
        return None
    prompt = instruction if not input_text else f"{instruction}\n{input_text}"
    return {
        "prompt": prompt,
        "response": output,
        "source": "IlyaGusev/ru_turbo_alpaca",
    }


def main() -> int:
    args = parse_args()
    try:
        import zstandard as zstd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install huggingface-hub zstandard") from exc

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(hf_hub_download(repo_id=args.dataset, repo_type="dataset", filename=args.filename))
    written = 0
    with src_path.open("rb") as compressed, out_path.open("w", encoding="utf-8", newline="\n") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        text_stream = reader
        buffer = b""
        while written < args.max_records:
            chunk = text_stream.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line.decode("utf-8"))
                if not isinstance(row, dict):
                    continue
                record = normalize_row(row)
                if record is None:
                    continue
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_records:
                    break
        if written < args.max_records and buffer.strip():
            row = json.loads(buffer.decode("utf-8"))
            if isinstance(row, dict):
                record = normalize_row(row)
                if record is not None:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
    if written == 0:
        raise RuntimeError(f"No usable records written from {args.dataset}")
    print(f"Wrote {written} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
