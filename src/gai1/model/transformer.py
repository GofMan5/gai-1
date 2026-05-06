from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from gai1.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def scaled_rope_base(base: float, seq_len: int, head_dim: int, scaling: str, factor: float, original_context: int) -> float:
    if scaling != "dynamic_ntk" or factor <= 1.0 or original_context <= 0 or seq_len <= original_context:
        return base
    exponent = head_dim / max(1, head_dim - 2)
    stretch = (factor * seq_len / original_context) - (factor - 1)
    return base * (max(1.0, stretch) ** exponent)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
    scaling: str = "none",
    factor: float = 1.0,
    original_context: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_base = scaled_rope_base(base, seq_len, head_dim, scaling, factor, original_context)
    inv_freq = 1.0 / (effective_base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    if scaling == "linear" and factor > 1.0:
        positions = positions / factor
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, base: float) -> torch.Tensor:
    _batch, _heads, seq_len, head_dim = x.shape
    cos, sin = build_rope_cache(seq_len, head_dim, base, x.device, x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.rope_base = cfg.rope_base
        self.rope_scaling = cfg.rope_scaling
        self.rope_scaling_factor = cfg.rope_scaling_factor
        self.rope_original_context = cfg.rope_original_context or cfg.block_size
        self.use_fused_qkv = self.n_kv_head == self.n_head
        if self.use_fused_qkv:
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        else:
            self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self._rope_cache_key: tuple[int, torch.device, torch.dtype, str, float, int] | None = None
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        _batch, _heads, seq_len, head_dim = x.shape
        key = (seq_len, x.device, x.dtype, self.rope_scaling, self.rope_scaling_factor, self.rope_original_context)
        if self._rope_cache_key != key or self._rope_cos is None or self._rope_sin is None:
            self._rope_cos, self._rope_sin = build_rope_cache(
                seq_len,
                head_dim,
                self.rope_base,
                x.device,
                x.dtype,
                self.rope_scaling,
                self.rope_scaling_factor,
                self.rope_original_context,
            )
            self._rope_cache_key = key
        return (x * self._rope_cos) + (rotate_half(x) * self._rope_sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        if self.use_fused_qkv:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self._rope(q)
        k = self._rope(k)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int((4 * cfg.n_embd * 2) / 3)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class SimpleMoE(nn.Module):
    """Small readable MoE. Production MoE must use expert-parallel kernels."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = min(cfg.n_experts_per_token, cfg.n_experts)
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(cfg.n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, channels = x.shape
        flat = x.reshape(batch * seq_len, channels)
        router_logits = self.gate(flat)
        router_probs = router_logits.softmax(dim=-1)
        top_weight, top_index = torch.topk(router_probs, self.top_k, dim=-1)
        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_index == expert_id
            if not mask.any():
                continue
            token_ids, choice_ids = mask.nonzero(as_tuple=True)
            expert_out = expert(flat[token_ids])
            output[token_ids] += expert_out * top_weight[token_ids, choice_ids].unsqueeze(-1)

        density = router_probs.mean(dim=0)
        load = F.one_hot(top_index, num_classes=self.n_experts).float().mean(dim=(0, 1))
        aux_loss = self.n_experts * torch.sum(density * load)
        return output.view(batch, seq_len, channels), aux_loss


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm_2 = RMSNorm(cfg.n_embd)
        self.ffn = SimpleMoE(cfg) if cfg.use_moe else SwiGLU(cfg)
        self.use_moe = cfg.use_moe

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm_1(x))
        if self.use_moe:
            y, aux_loss = self.ffn(self.norm_2(x))
        else:
            y = self.ffn(self.norm_2(x))
            aux_loss = x.new_tensor(0.0)
        return x + y, aux_loss


class GAIModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        if idx.size(1) > self.cfg.block_size:
            raise ValueError(f"Sequence length {idx.size(1)} exceeds block_size={self.cfg.block_size}")
        x = self.drop(self.token_embedding(idx))
        moe_aux = x.new_tensor(0.0)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, aux = checkpoint(block, x, use_reentrant=False)
            else:
                x, aux = block(x)
            moe_aux = moe_aux + aux
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            loss = ce_loss + (self.cfg.moe_aux_loss_weight * moe_aux)
        return logits, loss, {"moe_aux_loss": moe_aux}

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.block_size :]
            logits, _loss, _info = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def config_dict(self) -> dict[str, object]:
        return asdict(self.cfg)


def estimate_training_flops(params: int, tokens: int) -> float:
    return 6.0 * float(params) * float(tokens)


def format_param_count(params: int) -> str:
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.2f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    if params >= 1_000:
        return f"{params / 1_000:.2f}K"
    return str(params)
