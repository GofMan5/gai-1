# GAI-1 TUI

Launch:

```powershell
.\run_tui.bat
```

Manual launch with a chat adapter:

```powershell
.\.venv\Scripts\python.exe .\scripts\tui.py --checkpoint .\outputs\gai1_train_gpu\last.pt --adapter .\outputs\gai1_sft_lora\adapter.pt --level high
```

Experimental 128k context launch:

```powershell
.\run_tui.bat --context-length 131072 --rope-scaling linear
```

The TUI reports `visible ctx` separately from total accumulated tokens. The
current local checkpoint was trained at a shorter context, so 128k is an
experimental load-time window until long-context tuning and evals are done.

Commands inside TUI:

- `/help` - show commands.
- `/level low|medium|high|max` - switch reasoning mode.
- `/effort low|medium|high|max` - alias for `/level`.
- `/stats` - show last runtime stats.
- `/config` - show loaded checkpoint, adapter, context, dtype, and generation settings.
- `/history` - show recent turns.
- `/examples` - show useful starter prompts.
- `/clear` - clear chat history.
- `/quit` or `/exit` - close TUI.

Panels:

- `Chat` - recent messages, turn numbers, separators, and streaming output.
- `Runtime` - grouped model/runtime stats, token counts, visible context, latency, tokens/sec, VRAM.
- `Reasoning Trace` - profile, plan, verifier passes, critiques, rollbacks.

The reasoning trace is a runtime scaffold for control and visibility. It is not hidden model weights or real private chain-of-thought.

Unknown slash commands show suggestions. Example: `/hlep` suggests `/help`.
