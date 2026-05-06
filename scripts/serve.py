from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
from gai1.data import ASSISTANT_LABEL, USER_LABEL
from gai1.inference import GenerationConfig, generate_text
from gai1.loading import LoadOptions, load_model
from gai1.tokenizer import BPETokenizer, ByteTokenizer


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GAI-1 with a small OpenAI-compatible local API.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--rope-scaling", default=None, choices=("none", "linear", "dynamic_ntk"))
    parser.add_argument("--rope-scaling-factor", type=float, default=None)
    parser.add_argument("--rope-original-context", type=int, default=None)
    parser.add_argument("--max-request-bytes", type=int, default=65536)
    return parser.parse_args()


def parse_stop(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError("stop must be a string or list of strings")


def chat_prompt_from_messages(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"{ASSISTANT_LABEL}: {content}")
        else:
            lines.append(f"{USER_LABEL}: {content}")
    lines.append(f"{ASSISTANT_LABEL}:")
    return "\n".join(lines)


class GAIService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config(ROOT / args.config)
        adapter_path = ROOT / args.adapter if args.adapter else None
        self.model, self.metadata = load_model(
            LoadOptions(
                checkpoint_path=ROOT / args.checkpoint,
                adapter_path=adapter_path,
                device=args.device,
                dtype=args.dtype,
                context_length=args.context_length,
                rope_scaling=args.rope_scaling,
                rope_scaling_factor=args.rope_scaling_factor,
                rope_original_context=args.rope_original_context,
            )
        )
        if self.cfg.tokenizer.kind == "byte":
            self.tokenizer = ByteTokenizer()
        else:
            self.tokenizer = BPETokenizer(ROOT / self.cfg.tokenizer.path)

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream", False):
            raise ValueError("stream=true is not supported yet")
        prompt_text = chat_prompt_from_messages(payload.get("messages"))
        stop_strings = parse_stop(payload.get("stop")) + (f"\n{USER_LABEL}:", "\nUser:", "\nYou:")
        config = GenerationConfig(
            max_new_tokens=int(payload.get("max_tokens", 128)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload["top_k"]) if payload.get("top_k") is not None else 50,
            top_p=float(payload["top_p"]) if payload.get("top_p") is not None else None,
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            stop_strings=stop_strings,
            return_full_text=False,
        )
        result = generate_text(self.model, self.tokenizer, prompt_text, config)
        model_name = str(payload.get("model", "gai-1"))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": result.usage,
        }


def make_handler(service: GAIService, max_request_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "metadata": service.metadata})
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > max_request_bytes:
                self._send(413, {"error": "request_too_large"})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    self._send(200, service.complete(payload))
                else:
                    self._send(404, {"error": "not_found"})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": str(exc)})

    return Handler


def main() -> int:
    configure_console()
    args = parse_args()
    service = GAIService(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, args.max_request_bytes))
    print(f"Serving GAI-1 on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
