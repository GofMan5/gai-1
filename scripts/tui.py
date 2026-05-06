from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import ASSISTANT_LABEL, USER_LABEL
from gai1.loading import LoadOptions, load_model
from gai1.reasoning import GAIReasoningRuntime, ReasoningTrace, load_reasoning_profiles
from gai1.tokenizer import BPETokenizer, ByteTokenizer


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    elapsed_s: float = 0.0
    tokens_per_s: float = 0.0
    total_context_tokens: int = 0
    visible_context_tokens: int = 0
    max_context_tokens: int = 0
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAI-1 terminal chat UI with reasoning trace and token stats.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--checkpoint", default="outputs/gai1_train_gpu/last.pt")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--rope-scaling", default=None, choices=("none", "linear", "dynamic_ntk"))
    parser.add_argument("--reasoning-profiles", default="configs/reasoning_modes.json")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--history-turns", type=int, default=6)
    return parser.parse_args()


def load_chat_tokenizer(config_path: str):
    cfg = load_config(ROOT / config_path)
    if cfg.tokenizer.kind == "byte":
        return ByteTokenizer()
    return BPETokenizer(ROOT / cfg.tokenizer.path)


def format_history(history: list[tuple[str, str]], user_text: str, max_turns: int) -> str:
    selected = history[-max_turns:] if max_turns > 0 else []
    parts: list[str] = []
    for user, assistant in selected:
        parts.append(f"{USER_LABEL}: {user}")
        parts.append(f"{ASSISTANT_LABEL}: {assistant}")
    parts.append(f"{USER_LABEL}: {user_text}")
    parts.append(f"{ASSISTANT_LABEL}:")
    return "\n".join(parts)


def memory_stats(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.memory_allocated(device) / 1024**3, torch.cuda.memory_reserved(device) / 1024**3


def make_stats_panel(stats: TurnStats, metadata: dict[str, object], level: str) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    table.add_row("level", level)
    table.add_row("device", str(metadata.get("device", "?")))
    table.add_row("dtype", str(metadata.get("dtype", "?")))
    table.add_row("quant", str(metadata.get("quantization", "?")))
    table.add_row("adapter", "yes" if metadata.get("adapter_path") else "no")
    table.add_row("prompt tok", str(stats.prompt_tokens))
    table.add_row("gen tok", str(stats.generated_tokens))
    table.add_row("visible ctx", f"{stats.visible_context_tokens}/{stats.max_context_tokens}")
    table.add_row("total ctx", str(stats.total_context_tokens))
    table.add_row("latency", f"{stats.elapsed_s:.2f}s")
    table.add_row("tok/s", f"{stats.tokens_per_s:.2f}")
    table.add_row("VRAM alloc", f"{stats.vram_allocated_gb:.2f}GB")
    table.add_row("VRAM reserv", f"{stats.vram_reserved_gb:.2f}GB")
    return Panel(table, title="Stats", border_style="green", box=box.ROUNDED)


def make_reasoning_panel(trace: ReasoningTrace | None) -> Panel:
    if trace is None:
        return Panel("Reasoning trace will appear after the next prompt.", title="Reasoning", border_style="yellow")
    lines: list[str] = [f"[bold]level:[/bold] {trace.level}", f"[bold]rollbacks:[/bold] {trace.rollbacks}", ""]
    lines.append("[bold]plan[/bold]")
    for item in trace.plan:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("[bold]verifier[/bold]")
    lines.extend(f"- {result}" for result in trace.verifier_results[:8])
    if trace.critiques and any(trace.critiques):
        lines.append("")
        lines.append("[bold]critique[/bold]")
        for batch in trace.critiques:
            for issue in batch:
                lines.append(f"- {issue}")
    return Panel("\n".join(lines), title="Reasoning Trace", border_style="yellow", box=box.ROUNDED)


def make_chat_panel(history: list[tuple[str, str]], partial: str = "") -> Panel:
    text = Text()
    for user, assistant in history[-5:]:
        text.append("You: ", style="bold cyan")
        text.append(user + "\n")
        text.append("GAI-1: ", style="bold magenta")
        text.append(assistant + "\n\n")
    if partial:
        text.append("GAI-1 streaming: ", style="bold magenta")
        text.append(partial)
    return Panel(text or "No messages yet.", title="Chat", border_style="blue", box=box.ROUNDED)


def render_screen(
    history: list[tuple[str, str]],
    trace: ReasoningTrace | None,
    stats: TurnStats,
    metadata: dict[str, object],
    level: str,
    partial: str = "",
) -> Table:
    outer = Table.grid(expand=True)
    outer.add_column(ratio=3)
    outer.add_column(ratio=2)
    right = Table.grid(expand=True)
    right.add_row(make_stats_panel(stats, metadata, level))
    right.add_row(make_reasoning_panel(trace))
    outer.add_row(make_chat_panel(history, partial), right)
    return outer


@torch.no_grad()
def generate_with_live(
    model: torch.nn.Module,
    tokenizer: object,
    prompt_text: str,
    history: list[tuple[str, str]],
    trace: ReasoningTrace,
    metadata: dict[str, object],
    level: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    console: Console,
) -> tuple[str, TurnStats]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, add_bos=True)  # type: ignore[attr-defined]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    stats = TurnStats(
        prompt_tokens=len(input_ids),
        total_context_tokens=len(input_ids),
        visible_context_tokens=min(len(input_ids), model.cfg.block_size),  # type: ignore[attr-defined]
        max_context_tokens=model.cfg.block_size,  # type: ignore[attr-defined]
    )
    start = time.perf_counter()
    generated: list[int] = []

    with Live(render_screen(history, trace, stats, metadata, level), console=console, refresh_per_second=8, transient=False) as live:
        for _ in range(max_new_tokens):
            context = idx[:, -model.cfg.block_size :]  # type: ignore[attr-defined]
            logits, _loss, _info = model(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                values, _indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            generated.append(token_id)
            idx = torch.cat((idx, next_token), dim=1)

            stats.generated_tokens = len(generated)
            stats.elapsed_s = time.perf_counter() - start
            stats.tokens_per_s = stats.generated_tokens / max(stats.elapsed_s, 1e-6)
            stats.total_context_tokens = idx.size(1)
            stats.visible_context_tokens = min(idx.size(1), model.cfg.block_size)  # type: ignore[attr-defined]
            stats.max_context_tokens = model.cfg.block_size  # type: ignore[attr-defined]
            stats.vram_allocated_gb, stats.vram_reserved_gb = memory_stats(device)
            partial = tokenizer.decode(generated)  # type: ignore[attr-defined]
            live.update(render_screen(history, trace, stats, metadata, level, partial=partial))

            if token_id in {getattr(tokenizer, "eos_id", -1), getattr(tokenizer, "eot_id", -2)}:
                break

    answer = tokenizer.decode(generated).strip()  # type: ignore[attr-defined]
    return answer, stats


def print_help(console: Console) -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "/help - commands",
                    "/quit or /exit - close TUI",
                    "/level low|medium|high|max - switch reasoning level",
                    "/effort low|medium|high|max - same as /level",
                    "/clear - clear chat history",
                    "/stats - show current runtime stats",
                ]
            ),
            title="Commands",
            border_style="cyan",
        )
    )


def main() -> int:
    configure_console()
    args = parse_args()
    console = Console()
    adapter_path = ROOT / args.adapter if args.adapter else None
    model, metadata = load_model(
        LoadOptions(
            checkpoint_path=ROOT / args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            adapter_path=adapter_path,
            context_length=args.context_length,
            rope_scaling=args.rope_scaling,
        )
    )
    tokenizer = load_chat_tokenizer(args.config)
    profiles = load_reasoning_profiles(ROOT / args.reasoning_profiles)
    reasoning = GAIReasoningRuntime(level=args.level, profiles=profiles)
    level = args.level
    history: list[tuple[str, str]] = []
    last_stats = TurnStats()

    loaded = f"Loaded {args.checkpoint}"
    if args.adapter:
        loaded += f"\nAdapter {args.adapter}"
    console.print(Panel(f"{loaded}\nType /help for commands.", title="GAI-1 TUI", border_style="green"))
    while True:
        user_text = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        if not user_text:
            continue
        command = user_text.casefold()
        if command in {"/quit", "/exit"}:
            break
        if command == "/help":
            print_help(console)
            continue
        if command == "/clear":
            history.clear()
            console.print("[green]History cleared.[/green]")
            continue
        if command == "/stats":
            console.print(make_stats_panel(last_stats, metadata, level))
            continue
        if command.startswith("/level") or command.startswith("/effort"):
            parts = user_text.split()
            if len(parts) != 2:
                console.print("[red]Usage: /level low|medium|high|max or /effort low|medium|high|max[/red]")
                continue
            try:
                reasoning.set_level(parts[1])
                level = parts[1]
                console.print(f"[green]Reasoning effort set to {level}[/green]")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
            continue

        trace = reasoning.run(user_text, level=level)
        prompt = format_history(history, user_text, args.history_turns)
        answer, last_stats = generate_with_live(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            history=history,
            trace=trace,
            metadata=metadata,
            level=level,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            console=console,
        )
        history.append((user_text, answer))
        console.print(render_screen(history, trace, last_stats, metadata, level))

    console.print("[green]Bye.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
