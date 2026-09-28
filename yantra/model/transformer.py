"""Yantra Neural Backbone: Compact Transformer with GQA, RoPE, SwiGLU and KV Caching."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from yantra.model.config import YantraConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = max(seq_len, self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[offset : offset + seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    # cos, sin shape: (seq_len, dim) -> unsqueeze to (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: YantraConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KVCache:
    """Key-Value cache for autoregressive inference with GQA."""

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # First token(s) for this layer
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states
        else:
            # Concatenate along sequence dimension (dim=2)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.key_cache or len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def reset(self):
        self.key_cache.clear()
        self.value_cache.clear()


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with Rotary Embeddings and KV Cache support."""

    def __init__(self, config: YantraConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Calculate position offset for RoPE if KV cache is active
        past_length = kv_cache.get_seq_length(self.layer_idx) if kv_cache is not None else 0
        cos, sin = rotary_emb(hidden_states, q_len, offset=past_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Repeat KV heads for GQA if num_kv_groups > 1
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled Dot-Product Attention with causal mask
        is_causal = (attention_mask is None and q_len > 1 and past_length == 0)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.config.hidden_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class YantraDecoderLayer(nn.Module):
    """Transformer decoder block: RMSNorm -> GQA -> Residual -> RMSNorm -> SwiGLU -> Residual."""

    def __init__(self, config: YantraConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Self-Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class YantraForToolCalling(nn.Module):
    """Specialized foundation model for tool dispatching under 50 MB."""

    def __init__(self, config: Optional[YantraConfig] = None):
        super().__init__()
        self.config = config or YantraConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )

        self.layers = nn.ModuleList([
            YantraDecoderLayer(self.config, idx) for idx in range(self.config.num_hidden_layers)
        ])

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Output LM Head (optionally tied with embedding table)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Auxiliary Confidence Head (calibrated refusal / tool dispatch probability)
        if self.config.has_confidence_head:
            self.confidence_head = nn.Linear(self.config.hidden_size, 1)
        else:
            self.confidence_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        return_dict: bool = True,
        return_logits: Optional[bool] = None,
    ) -> Dict[str, Any]:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                rotary_emb=self.rotary_emb,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        # Decide whether to compute full [batch, seq_len, vocab_size] logits.
        # When return_logits=False (used by YantraTrainer), selective projection is activated to save 90%+ RAM.
        if return_logits is None:
            return_logits = True

        loss = None
        logits = None

        if return_logits:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
        elif labels is not None:
            # Selective logit projection: project hidden states strictly for tokens where labels != -100
            shift_labels = labels[..., 1:].contiguous()
            shift_hidden = hidden_states[..., :-1, :]
            active_mask = (shift_labels != -100)
            if active_mask.any():
                active_hidden = shift_hidden[active_mask]
                active_targets = shift_labels[active_mask]
                active_logits = self.lm_head(active_hidden)
                loss = F.cross_entropy(active_logits, active_targets)
            else:
                loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        confidence = None
        if self.confidence_head is not None:
            # Confidence score for the final token representation
            confidence = torch.sigmoid(self.confidence_head(hidden_states[:, -1, :]))

        if not return_dict:
            return logits, loss, confidence

        return {
            "logits": logits,
            "loss": loss,
            "confidence": confidence,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        self.eval()
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        cache = KVCache()
        curr_input_ids = input_ids
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(curr_input_ids, kv_cache=cache)
            next_token_logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            curr_input_ids = next_token  # For next iteration, feed only the new token

            if next_token.item() in stop_ids:
                break

        return generated
