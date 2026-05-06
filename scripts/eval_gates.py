from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import PackedTextDataset
from gai1.loading import LoadOptions, load_model
from gai1.reasoning import GAIReasoningRuntime
from gai1.tokenizer import BPETokenizer, ByteTokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAI-1 release eval gates.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--gates", default="configs/eval_gates.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    return parser.parse_args()


def load_tokenizer(cfg):
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    gates = json.loads((ROOT / args.gates).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_model(LoadOptions(checkpoint_path=ROOT / args.checkpoint, device=device, dtype="auto"))
    tokenizer = load_tokenizer(cfg)
    dataset = PackedTextDataset(ROOT / (args.data or cfg.data.train_path), tokenizer, cfg.data.block_size, cfg.data.field)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    losses: list[float] = []
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if index >= 16:
                break
            x = x.to(device)
            y = y.to(device)
            _logits, loss, _info = model(x, y)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
    ppl = math.exp(min(20.0, sum(losses) / max(1, len(losses))))
    reasoning_ok = all(GAIReasoningRuntime(level=level).run("Проверка").final for level in gates["reasoning_levels"])
    passed = ppl <= float(gates["max_perplexity"]) and reasoning_ok
    report = {"passed": passed, "perplexity": ppl, "metadata": metadata, "reasoning_ok": reasoning_ok}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

