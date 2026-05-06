from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune GAI-1 RTX 3060 training micro-batch for throughput.")
    parser.add_argument("--base-config", default="configs/train_gpu_large.json")
    parser.add_argument("--out-config", default="configs/train_gpu_autotuned.json")
    parser.add_argument("--dataset", default="data/raw/fineweb2_ru_large.jsonl")
    parser.add_argument("--candidates", default="1,2,3,4")
    parser.add_argument("--target-effective-batch", type=int, default=32)
    parser.add_argument("--probe-steps", type=int, default=8)
    return parser.parse_args()


def write_config(base: dict[str, object], out_path: Path, batch_size: int, target_effective_batch: int, dataset: str) -> None:
    payload = json.loads(json.dumps(base))
    accumulation = max(1, target_effective_batch // batch_size)
    payload["data"]["train_path"] = dataset
    payload["data"]["streaming"] = True
    payload["train"]["batch_size"] = batch_size
    payload["train"]["gradient_accumulation_steps"] = accumulation
    payload["train"]["max_steps"] = 5000
    payload["train"]["save_every"] = 999999
    payload["train"]["log_every"] = 1
    payload["train"]["resume"] = False
    payload["train"]["output_dir"] = f"outputs/autotune_bs{batch_size}"
    payload["train"]["fused_optimizer"] = True
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_tokens_per_s(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    last = ""
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if not last:
        return 0.0
    return float(json.loads(last).get("tokens_per_s", 0.0))


def main() -> int:
    args = parse_args()
    base = json.loads((ROOT / args.base_config).read_text(encoding="utf-8"))
    candidates = [int(item.strip()) for item in args.candidates.split(",") if item.strip()]
    report: list[dict[str, object]] = []
    tmp_dir = ROOT / "configs" / "autotune"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for batch_size in candidates:
        config_path = tmp_dir / f"train_bs{batch_size}.json"
        write_config(base, config_path, batch_size, args.target_effective_batch, args.dataset)
        cmd = [sys.executable, "scripts/train_pretrain.py", "--config", str(config_path.relative_to(ROOT)), "--max-steps", str(args.probe_steps)]
        print("+ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
        log_path = ROOT / f"outputs/autotune_bs{batch_size}/train_log.jsonl"
        entry = {
            "batch_size": batch_size,
            "accumulation": max(1, args.target_effective_batch // batch_size),
            "returncode": result.returncode,
            "tokens_per_s": latest_tokens_per_s(log_path),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        report.append(entry)

    valid = [entry for entry in report if entry["returncode"] == 0 and float(entry["tokens_per_s"]) > 0]
    if not valid:
        print("No valid candidate finished. Keep the existing config.")
        return 1
    best = max(valid, key=lambda row: float(row["tokens_per_s"]))
    out_config = ROOT / args.out_config
    write_config(base, out_config, int(best["batch_size"]), args.target_effective_batch, args.dataset)
    report_path = ROOT / "reports" / "autotune_rtx3060.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"best": best, "candidates": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best config: {out_config} tokens_per_s={best['tokens_per_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
