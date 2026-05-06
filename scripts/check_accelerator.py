from __future__ import annotations

import sys

import torch


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console()
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("CUDA не видна. Нужен CUDA PyTorch wheel, см. scripts/setup_rtx3060_windows.ps1")
        return 1
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        print(f"gpu[{index}]={props.name}")
        print(f"  vram_total={total / 1024**3:.2f}GB")
        print(f"  vram_free={free / 1024**3:.2f}GB")
        print(f"  capability={props.major}.{props.minor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

