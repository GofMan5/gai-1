from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import PackedTextDataset, format_chat_prompt
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
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def load_tokenizer(cfg: Any):
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for char in letters if "\u0400" <= char <= "\u04ff")
    return cyr / len(letters)


def repetition_ratio(text: str) -> float:
    words = [word.casefold() for word in text.split() if word.strip()]
    if not words:
        return 1.0
    return 1.0 - len(set(words)) / len(words)


def prompt_echo_ratio(prompt: str, output: str) -> float:
    prompt_words = {word.casefold() for word in prompt.split() if len(word) > 2}
    output_words = [word.casefold() for word in output.split() if len(word) > 2]
    if not prompt_words or not output_words:
        return 0.0
    echoed = sum(1 for word in output_words if word in prompt_words)
    return echoed / len(output_words)


def has_mojibake(text: str) -> bool:
    markers = (chr(0x00C3), chr(0x00C2), chr(0x00D0), chr(0x00D1), "\ufffd")
    return any(marker in text for marker in markers)


def validate_gates(gates: dict[str, Any]) -> None:
    prompts = gates.get("sample_prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Eval gates must define a non-empty sample_prompts list")
    for prompt in prompts:
        if has_mojibake(str(prompt)):
            raise ValueError(f"Eval prompt contains mojibake: {prompt!r}")


def resolve_eval_data(args: argparse.Namespace, cfg: Any) -> str:
    eval_data = args.data or cfg.data.val_path
    if not eval_data:
        raise ValueError("Release eval requires --data or config data.val_path; refusing to fall back to train data")
    normalized_eval = Path(eval_data).as_posix()
    normalized_train = Path(cfg.data.train_path).as_posix()
    if normalized_eval == normalized_train:
        raise ValueError("Release eval data must be held out and cannot equal data.train_path")
    return eval_data


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(format_chat_prompt(prompt), add_bos=True)
    idx = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40)
    generated = out[0, len(input_ids) :].tolist()
    return tokenizer.decode(generated).strip()


def main() -> int:
    configure_console()
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    gates = json.loads((ROOT / args.gates).read_text(encoding="utf-8"))
    validate_gates(gates)
    eval_data = resolve_eval_data(args, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = ROOT / args.adapter if args.adapter else None
    model, metadata = load_model(
        LoadOptions(checkpoint_path=ROOT / args.checkpoint, adapter_path=adapter_path, device=device, dtype="auto")
    )
    tokenizer = load_tokenizer(cfg)
    dataset = PackedTextDataset(ROOT / eval_data, tokenizer, cfg.data.block_size, cfg.data.field)
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
    reasoning_ok = all(GAIReasoningRuntime(level=level).run("\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430").final for level in gates["reasoning_levels"])
    generation_reports = []
    for prompt in gates.get("sample_prompts", []):
        output = generate_answer(model, tokenizer, str(prompt), args.max_new_tokens)
        generation_reports.append(
            {
                "prompt": prompt,
                "output": output,
                "chars": len(output),
                "cyrillic_ratio": cyrillic_ratio(output),
                "repetition_ratio": repetition_ratio(output),
                "prompt_echo_ratio": prompt_echo_ratio(str(prompt), output),
                "mojibake": has_mojibake(output),
            }
        )
    generation_ok = all(
        item["chars"] >= int(gates["min_generation_chars"])
        and item["cyrillic_ratio"] >= float(gates["min_cyrillic_ratio"])
        and item["repetition_ratio"] <= float(gates["max_repetition_ratio"])
        and item["prompt_echo_ratio"] <= float(gates["max_prompt_echo_ratio"])
        and not item["mojibake"]
        for item in generation_reports
    )
    passed = ppl <= float(gates["max_perplexity"]) and reasoning_ok and generation_ok
    report = {
        "passed": passed,
        "perplexity": ppl,
        "eval_data": eval_data,
        "eval_data_is_train": False,
        "metadata": metadata,
        "reasoning_ok": reasoning_ok,
        "generation_ok": generation_ok,
        "generations": generation_reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
