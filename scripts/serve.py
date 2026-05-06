from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


class GAIService:
    def __init__(self, config_path: str, checkpoint: str) -> None:
        self.cfg = load_config(ROOT / config_path)
        self.model, self.metadata = load_model(LoadOptions(checkpoint_path=ROOT / checkpoint, device="auto", dtype="auto"))
        if self.cfg.tokenizer.kind == "byte":
            self.tokenizer = ByteTokenizer()
        else:
            self.tokenizer = BPETokenizer(ROOT / self.cfg.tokenizer.path)

    def complete(self, prompt: str, max_tokens: int = 128, temperature: float = 0.8) -> str:
        device = next(self.model.parameters()).device
        text = f"Пользователь: {prompt}\nАссистент:"
        tokens = self.tokenizer.encode(text, add_bos=True)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(out[0].tolist())


def make_handler(service: GAIService):
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
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    messages = payload.get("messages", [])
                    prompt = ""
                    if messages:
                        prompt = str(messages[-1].get("content", ""))
                    answer = service.complete(
                        prompt,
                        max_tokens=int(payload.get("max_tokens", 128)),
                        temperature=float(payload.get("temperature", 0.8)),
                    )
                    self._send(200, {"choices": [{"message": {"role": "assistant", "content": answer}}]})
                else:
                    self._send(404, {"error": "not_found"})
            except Exception as exc:
                self._send(500, {"error": str(exc)})

    return Handler


def main() -> int:
    configure_console()
    args = parse_args()
    service = GAIService(args.config, args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Serving GAI-1 on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

