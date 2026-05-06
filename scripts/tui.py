from __future__ import annotations

import argparse
import difflib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

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
from gai1.tokenizer import BPETokenizer, ByteTokenizer, assert_tokenizer_compatible

COMMANDS: dict[str, str] = {
    "/help": "show commands",
    "/quit": "close TUI",
    "/exit": "close TUI",
    "/clear": "clear chat history",
    "/stats": "show runtime stats",
    "/config": "show loaded model/config metadata",
    "/history": "show recent turns",
    "/examples": "show useful prompts",
    "/level": "set reasoning level: /level low|medium|high|max",
    "/effort": "alias for /level",
    "/reasoning": "set reasoning view: /reasoning compact|full|off",
}
LEVELS = ("low", "medium", "high", "max")
REASONING_VIEWS = ("compact", "full", "off")
T = TypeVar("T")


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
    parser.add_argument("--reasoning-view", default="full", choices=REASONING_VIEWS)
    parser.add_argument("--reasoning-backend", default="model", choices=("model", "scaffold"))
    parser.add_argument("--reasoning-draft-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--history-turns", type=int, default=6)
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    parser.add_argument("--strict-tokenizer-path", action="store_true")
    return parser.parse_args()


class LocalModelGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    @torch.no_grad()
    def complete(self, prompt: str) -> str:
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, add_bos=True)  # type: ignore[attr-defined]
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            context = idx[:, -self.model.cfg.block_size :]  # type: ignore[attr-defined]
            logits, _loss, _info = self.model(context)
            logits = logits[:, -1, :] / max(self.temperature, 1e-6)
            logits = apply_repetition_penalty(logits, generated, self.repetition_penalty)
            if self.top_k > 0:
                values, _indices = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            if token_id in {getattr(self.tokenizer, "eos_id", -1), getattr(self.tokenizer, "eot_id", -2)}:
                break
            generated.append(token_id)
            idx = torch.cat((idx, next_token), dim=1)
        return trim_answer(self.tokenizer.decode(generated)).strip()  # type: ignore[attr-defined]


def load_chat_tokenizer(config_path: str):
    cfg = load_config(ROOT / config_path)
    if cfg.tokenizer.kind == "byte":
        return cfg, ByteTokenizer()
    return cfg, BPETokenizer(ROOT / cfg.tokenizer.path)


def format_history(history: list[tuple[str, str]], user_text: str, max_turns: int) -> str:
    selected = history[-max_turns:] if max_turns > 0 else []
    parts: list[str] = []
    for user, assistant in selected:
        parts.append(f"{USER_LABEL}: {user}")
        parts.append(f"{ASSISTANT_LABEL}: {assistant}")
    parts.append(f"{USER_LABEL}: {user_text}")
    parts.append(f"{ASSISTANT_LABEL}:")
    return "\n".join(parts)


def format_reasoning_context(trace: ReasoningTrace) -> str:
    profile = trace.profile
    lines = [
        "Internal reasoning controller:",
        f"- effort: {trace.level}",
        f"- drafts: {profile.get('draft_count', '?')}",
        f"- critic passes: {profile.get('critic_passes', '?')}",
        f"- verifier passes: {profile.get('verifier_passes', '?')}",
        "- plan:",
    ]
    lines.extend(f"  {index}. {item}" for index, item in enumerate(trace.plan, start=1))
    if trace.final:
        draft_summary = trace.final.replace("\n", " ").strip()
        lines.append(f"- current draft summary: {draft_summary[:500]}")
    if trace.critiques:
        issues = [issue for batch in trace.critiques for issue in batch]
        if issues:
            lines.append("- known weak spots to avoid:")
            lines.extend(f"  - {issue}" for issue in issues[:4])
    lines.append("Use this controller privately. Answer only as GAI-1, without dumping hidden chain-of-thought.")
    return "\n".join(lines)


def format_model_prompt(history: list[tuple[str, str]], user_text: str, max_turns: int, trace: ReasoningTrace) -> str:
    return format_reasoning_context(trace) + "\n\n" + format_history(history, user_text, max_turns)


def trim_answer(text: str) -> str:
    stops = (f"\n{USER_LABEL}:", "\nUser:", "\nYou:", f"\n{ASSISTANT_LABEL}:")
    cut = len(text)
    for stop in stops:
        pos = text.find(stop)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut].strip()


def memory_stats(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.memory_allocated(device) / 1024**3, torch.cuda.memory_reserved(device) / 1024**3


def format_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size >= 1024**3:
        return f"{size / 1024**3:.2f}GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.2f}MB"
    if size >= 1024:
        return f"{size / 1024:.2f}KB"
    return f"{size}B"


def make_loading_panel(steps: list[tuple[str, str]], active: str = "") -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column(style="white")
    for name, status in steps:
        if status == "done":
            marker = "[green]OK[/green]"
        elif status == "active":
            marker = "[yellow]..[/yellow]"
        elif status == "skip":
            marker = "[dim]--[/dim]"
        else:
            marker = "[dim]  [/dim]"
        table.add_row(marker, name)
    caption = active or "Preparing local inference runtime"
    return Panel(table, title=f"GAI-1 Loading - {caption}", border_style="cyan", box=box.ROUNDED)


def run_loading_step(console: Console, steps: list[tuple[str, str]], index: int, label: str, action: Callable[[], T]) -> T:
    steps[index] = (label, "active")
    console.clear()
    console.print(make_loading_panel(steps, active=label))
    start = time.perf_counter()
    result = action()
    steps[index] = (f"{label} ({time.perf_counter() - start:.1f}s)", "done")
    return result


def show_loaded_summary(console: Console, metadata: dict[str, object], elapsed_s: float, adapter: str | None) -> None:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    table.add_row("loaded in", f"{elapsed_s:.1f}s")
    table.add_row("checkpoint", str(metadata.get("checkpoint_path")))
    table.add_row("adapter", adapter or "none")
    table.add_row("device", str(metadata.get("device")))
    table.add_row("dtype", str(metadata.get("dtype")))
    table.add_row("quant", str(metadata.get("quantization")))
    table.add_row("context", str(metadata.get("context_length")))
    table.add_row("trained ctx", str(metadata.get("trained_context_length")))
    table.add_row("reasoning", str(metadata.get("reasoning_backend", "model")))
    console.clear()
    console.print(Panel(table, title="GAI-1 Ready", border_style="green", box=box.ROUNDED))
    time.sleep(0.7)


def make_stats_panel(stats: TurnStats, metadata: dict[str, object], level: str) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    rows = [
        ("effort", level),
        ("device", str(metadata.get("device", "?"))),
        ("dtype", str(metadata.get("dtype", "?"))),
        ("quant", str(metadata.get("quantization", "?"))),
        ("adapter", "yes" if metadata.get("adapter_path") else "no"),
        ("trained ctx", str(metadata.get("trained_context_length", "?"))),
        ("visible ctx", f"{stats.visible_context_tokens}/{stats.max_context_tokens}"),
        ("prompt/gen", f"{stats.prompt_tokens}/{stats.generated_tokens}"),
        ("latency", f"{stats.elapsed_s:.2f}s"),
        ("tok/s", f"{stats.tokens_per_s:.2f}"),
        ("VRAM", f"{stats.vram_allocated_gb:.2f}/{stats.vram_reserved_gb:.2f}GB"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    return Panel(table, title="Runtime", border_style="green", box=box.ROUNDED)


def make_reasoning_panel(trace: ReasoningTrace | None, view: str = "full") -> Panel:
    if view == "off":
        return Panel("Hidden. Enable with /reasoning compact or /reasoning full.", title="Reasoning", border_style="dim")

    if trace is None:
        return Panel(
            "Runtime trace appears after the next prompt.\nThis is the external reasoning scaffold, not neural hidden state.",
            title="Reasoning",
            border_style="yellow",
        )

    profile = trace.profile
    profile_table = Table.grid(padding=(0, 1))
    profile_table.add_column(style="cyan", justify="right")
    profile_table.add_column(style="white")
    profile_table.add_row("level", trace.level)
    profile_table.add_row("drafts", str(profile.get("draft_count", "?")))
    profile_table.add_row("critic", str(profile.get("critic_passes", "?")))
    profile_table.add_row("verifier", str(profile.get("verifier_passes", "?")))
    profile_table.add_row("rollbacks", str(trace.rollbacks))
    profile_table.add_row("source", "runtime scaffold")

    plan_text = Text()
    for index, item in enumerate(trace.plan, start=1):
        plan_text.append(f"{index}. ", style="bold yellow")
        plan_text.append(item + "\n")

    drafts_text = Text()
    for index, draft in enumerate(trace.drafts[:3], start=1):
        drafts_text.append(f"Draft {index}\n", style="bold magenta")
        drafts_text.append(draft.strip() + "\n")

    critique_text = Text()
    has_critiques = False
    for index, batch in enumerate(trace.critiques[:3], start=1):
        if not batch:
            continue
        has_critiques = True
        critique_text.append(f"Pass {index}\n", style="bold red")
        for issue in batch[:4]:
            critique_text.append(f"- {issue}\n", style="red")
    if not has_critiques:
        critique_text.append("no issues", style="green")

    verifier = Text()
    if trace.verifier_results:
        for result in trace.verifier_results[:6]:
            style = "green" if result.startswith("pass") else "red" if result.startswith("fail") else "dim"
            verifier.append(result + "\n", style=style)
    else:
        verifier.append("skipped", style="dim")

    grid = Table.grid(expand=True)
    if view == "compact":
        grid.add_column(ratio=1)
        grid.add_column(ratio=2)
        grid.add_row(profile_table, Panel(plan_text, title="Plan", border_style="yellow", box=box.ROUNDED))
        grid.add_row(Panel(verifier, title="Verifier", border_style="yellow", box=box.ROUNDED))
        return Panel(grid, title="Reasoning Trace", border_style="yellow", box=box.ROUNDED)

    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        Panel(profile_table, title="Profile", border_style="yellow", box=box.ROUNDED),
        Panel(plan_text, title="Plan", border_style="yellow", box=box.ROUNDED),
    )
    grid.add_row(
        Panel(drafts_text, title="Drafts", border_style="magenta", box=box.ROUNDED),
        Panel(critique_text, title="Critic", border_style="red", box=box.ROUNDED),
    )
    grid.add_row(
        Panel(verifier, title="Verifier", border_style="yellow", box=box.ROUNDED),
        Panel(trace.final.strip() or "empty", title="Reasoning Output", border_style="green", box=box.ROUNDED),
    )
    return Panel(grid, title="Reasoning Trace", border_style="yellow", box=box.ROUNDED)


def make_chat_panel(history: list[tuple[str, str]], partial: str = "") -> Panel:
    text = Text()
    if not history and not partial:
        text.append("No messages yet.\n", style="dim")
        text.append("Try: /examples or ask a question.", style="dim")
    for index, (user, assistant) in enumerate(history[-6:], start=max(1, len(history) - 5)):
        text.append(f"Turn {index}\n", style="bold blue")
        text.append("You: ", style="bold cyan")
        text.append(user.strip() + "\n")
        text.append("GAI-1: ", style="bold magenta")
        text.append((assistant.strip() or "[empty]").strip() + "\n")
        text.append("─" * 48 + "\n", style="dim")
    if partial:
        text.append(f"Turn {len(history) + 1}\n", style="bold blue")
        text.append("GAI-1 streaming: ", style="bold magenta")
        text.append(partial)
    return Panel(text, title=f"Chat ({len(history)} turns)", border_style="blue", box=box.ROUNDED)


def make_hint_panel() -> Panel:
    hint = Text()
    hint.append("Commands: ", style="bold cyan")
    hint.append("/help  /effort high  /reasoning full  /stats  /config  /clear  /quit", style="white")
    hint.append("\nTip: type '/' or an incomplete command to get suggestions.", style="dim")
    return Panel(hint, border_style="cyan", box=box.ROUNDED)


def render_screen(
    history: list[tuple[str, str]],
    trace: ReasoningTrace | None,
    stats: TurnStats,
    metadata: dict[str, object],
    level: str,
    reasoning_view: str,
    width: int,
    partial: str = "",
) -> Table:
    outer = Table.grid(expand=True)
    if width < 120 or reasoning_view == "full":
        top = Table.grid(expand=True)
        top.add_column(ratio=3)
        top.add_column(ratio=1)
        top.add_row(make_chat_panel(history, partial), make_stats_panel(stats, metadata, level))
        outer.add_row(top)
        outer.add_row(make_reasoning_panel(trace, reasoning_view))
    else:
        outer.add_column(ratio=3)
        outer.add_column(ratio=2)
        right = Table.grid(expand=True)
        right.add_row(make_stats_panel(stats, metadata, level))
        right.add_row(make_reasoning_panel(trace, reasoning_view))
        outer.add_row(make_chat_panel(history, partial), right)
    outer.add_row(make_hint_panel())
    return outer


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits
    unique_tokens = set(token_ids[-256:])
    for token_id in unique_tokens:
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


@torch.no_grad()
def generate_with_live(
    model: torch.nn.Module,
    tokenizer: object,
    prompt_text: str,
    history: list[tuple[str, str]],
    trace: ReasoningTrace,
    metadata: dict[str, object],
    level: str,
    reasoning_view: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
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

    with Live(
        render_screen(history, trace, stats, metadata, level, reasoning_view, console.width),
        console=console,
        refresh_per_second=6,
        transient=True,
        vertical_overflow="crop",
    ) as live:
        for _ in range(max_new_tokens):
            context = idx[:, -model.cfg.block_size :]  # type: ignore[attr-defined]
            logits, _loss, _info = model(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = apply_repetition_penalty(logits, generated, repetition_penalty)
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
            partial = trim_answer(tokenizer.decode(generated))  # type: ignore[attr-defined]
            live.update(render_screen(history, trace, stats, metadata, level, reasoning_view, console.width, partial=partial))

            if token_id in {getattr(tokenizer, "eos_id", -1), getattr(tokenizer, "eot_id", -2)}:
                break

    answer = trim_answer(tokenizer.decode(generated))  # type: ignore[attr-defined]
    return answer, stats


def command_suggestion(command: str) -> str:
    if command == "/":
        return "Available: " + "  ".join(COMMANDS)
    closest = difflib.get_close_matches(command.split()[0], COMMANDS.keys(), n=3)
    if closest:
        return "Did you mean: " + "  ".join(closest)
    return "Unknown command. Type /help."


def print_help(console: Console) -> None:
    table = Table(title="Commands", box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    for command, description in COMMANDS.items():
        table.add_row(command, description)
    console.print(table)


def print_examples(console: Console) -> None:
    examples = [
        "Объясни коротко, что такое tokenizer.",
        "Составь план улучшения качества модели.",
        "Проверь этот аргумент на слабые места: ...",
        "/effort high",
        "/reasoning full",
        "/stats",
    ]
    console.print(Panel("\n".join(examples), title="Examples", border_style="cyan"))


def print_config(console: Console, metadata: dict[str, object], args: argparse.Namespace) -> None:
    table = Table(title="Loaded Model", box=box.ROUNDED)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key in (
        "checkpoint_path",
        "adapter_path",
        "format",
        "device",
        "dtype",
        "quantization",
        "context_length",
        "trained_context_length",
        "tested_context_length",
        "rope_scaling",
        "reasoning_backend",
    ):
        table.add_row(key, str(metadata.get(key)))
    table.add_row("max_new_tokens", str(args.max_new_tokens))
    table.add_row("temperature", str(args.temperature))
    table.add_row("top_k", str(args.top_k))
    table.add_row("repetition_penalty", str(args.repetition_penalty))
    console.print(table)


def handle_command(
    user_text: str,
    console: Console,
    reasoning: GAIReasoningRuntime,
    history: list[tuple[str, str]],
    metadata: dict[str, object],
    last_stats: TurnStats,
    level: str,
    reasoning_view: str,
    args: argparse.Namespace,
) -> tuple[bool, bool, str, str]:
    command = user_text.casefold().strip()
    if command in {"/quit", "/exit"}:
        return True, True, level, reasoning_view
    if command == "/" or (command.startswith("/") and command.split()[0] not in COMMANDS):
        console.print(Panel(command_suggestion(command), title="Command hint", border_style="cyan"))
        return True, False, level, reasoning_view
    if command == "/help":
        print_help(console)
        return True, False, level, reasoning_view
    if command == "/examples":
        print_examples(console)
        return True, False, level, reasoning_view
    if command == "/clear":
        history.clear()
        console.clear()
        console.print("[green]History cleared.[/green]")
        return True, False, level, reasoning_view
    if command == "/stats":
        console.print(make_stats_panel(last_stats, metadata, level))
        return True, False, level, reasoning_view
    if command == "/config":
        print_config(console, metadata, args)
        return True, False, level, reasoning_view
    if command == "/history":
        console.print(make_chat_panel(history))
        return True, False, level, reasoning_view
    if command.startswith("/level") or command.startswith("/effort"):
        parts = user_text.split()
        if len(parts) == 1:
            console.print(Panel("Choose one: " + "  ".join(LEVELS), title="Reasoning effort", border_style="cyan"))
            return True, False, level, reasoning_view
        if len(parts) != 2 or parts[1] not in LEVELS:
            console.print(Panel("Usage: /effort low|medium|high|max", title="Command hint", border_style="red"))
            return True, False, level, reasoning_view
        reasoning.set_level(parts[1])
        console.print(f"[green]Reasoning effort set to {parts[1]}[/green]")
        return True, False, parts[1], reasoning_view
    if command.startswith("/reasoning"):
        parts = user_text.split()
        if len(parts) == 1:
            console.print(Panel("Choose one: " + "  ".join(REASONING_VIEWS), title="Reasoning view", border_style="cyan"))
            return True, False, level, reasoning_view
        if len(parts) != 2 or parts[1] not in REASONING_VIEWS:
            console.print(Panel("Usage: /reasoning compact|full|off", title="Command hint", border_style="red"))
            return True, False, level, reasoning_view
        console.print(f"[green]Reasoning view set to {parts[1]}[/green]")
        return True, False, level, parts[1]
    return False, False, level, reasoning_view


def main() -> int:
    configure_console()
    args = parse_args()
    console = Console()
    adapter_path = ROOT / args.adapter if args.adapter else None
    started = time.perf_counter()
    checkpoint_path = ROOT / args.checkpoint
    config_path = ROOT / args.config
    profiles_path = ROOT / args.reasoning_profiles
    steps: list[tuple[str, str]] = [
        (f"check checkpoint {args.checkpoint} [{format_size(checkpoint_path)}]", "pending"),
        (f"check adapter {args.adapter}" if args.adapter else "adapter skipped", "pending"),
        (f"load model on {args.device}/{args.dtype}", "pending"),
        (f"load tokenizer from {args.config}", "pending"),
        (f"load reasoning profiles {args.reasoning_profiles}", "pending"),
        ("warm up runtime metadata", "pending"),
    ]

    def check_checkpoint() -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def check_adapter() -> None:
        if args.adapter and adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    run_loading_step(console, steps, 0, steps[0][0], check_checkpoint)
    if args.adapter:
        run_loading_step(console, steps, 1, steps[1][0], check_adapter)
    else:
        steps[1] = (steps[1][0], "skip")

    model, metadata = run_loading_step(
        console,
        steps,
        2,
        steps[2][0],
        lambda: load_model(
            LoadOptions(
                checkpoint_path=checkpoint_path,
                device=args.device,
                dtype=args.dtype,
                adapter_path=adapter_path,
                context_length=args.context_length,
                rope_scaling=args.rope_scaling,
            )
        ),
    )
    cfg, tokenizer = run_loading_step(console, steps, 3, steps[3][0], lambda: load_chat_tokenizer(args.config))
    metadata["tokenizer_compatibility"] = assert_tokenizer_compatible(
        metadata,
        cfg,
        ROOT,
        tokenizer=tokenizer,
        allow_mismatch=args.allow_tokenizer_mismatch,
        strict_path=args.strict_tokenizer_path,
    )
    profiles = run_loading_step(console, steps, 4, steps[4][0], lambda: load_reasoning_profiles(profiles_path))

    def warmup() -> None:
        _ = config_path.exists()
        if torch.cuda.is_available() and str(metadata.get("device", "")).startswith("cuda"):
            torch.cuda.synchronize()

    run_loading_step(console, steps, 5, steps[5][0], warmup)
    metadata["reasoning_backend"] = args.reasoning_backend
    show_loaded_summary(console, metadata, time.perf_counter() - started, args.adapter)

    generator = None
    if args.reasoning_backend == "model":
        generator = LocalModelGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.reasoning_draft_tokens,
            temperature=max(0.25, args.temperature * 0.8),
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
    reasoning = GAIReasoningRuntime(generator=generator, level=args.level, profiles=profiles)
    level = args.level
    reasoning_view = args.reasoning_view
    history: list[tuple[str, str]] = []
    last_stats = TurnStats(max_context_tokens=int(metadata.get("context_length") or 0))
    trace: ReasoningTrace | None = None

    console.clear()
    console.print(render_screen(history, trace, last_stats, metadata, level, reasoning_view, console.width))
    while True:
        user_text = Prompt.ask("[bold cyan]You[/bold cyan] [dim](type /help)[/dim]").strip()
        if not user_text:
            continue
        if user_text.startswith("/"):
            handled, should_exit, level, reasoning_view = handle_command(
                user_text, console, reasoning, history, metadata, last_stats, level, reasoning_view, args
            )
            if should_exit:
                break
            if handled:
                continue

        trace = reasoning.run(user_text, level=level)
        prompt = format_model_prompt(history, user_text, args.history_turns, trace)
        answer, last_stats = generate_with_live(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            history=history,
            trace=trace,
            metadata=metadata,
            level=level,
            reasoning_view=reasoning_view,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            console=console,
        )
        history.append((user_text, answer))
        console.clear()
        console.print(render_screen(history, trace, last_stats, metadata, level, reasoning_view, console.width))

    console.print("[green]Bye.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
