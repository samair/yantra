"""Model configuration for Yantra sub-50MB architecture."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class YantraConfig:
    vocab_size: int = 8192
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 10
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    has_confidence_head: bool = True
    hidden_dropout: float = 0.0

    def calculate_parameter_count(self) -> Dict[str, Any]:
        """Calculate exact parameter breakdown and weight footprints."""
        embed_params = self.vocab_size * self.hidden_size

        # Self-Attention parameters per layer
        q_params = self.hidden_size * (self.num_attention_heads * self.head_dim)
        k_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        v_params = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        o_params = (self.num_attention_heads * self.head_dim) * self.hidden_size
        attn_params = q_params + k_params + v_params + o_params

        # MLP (SwiGLU: gate, up, down)
        mlp_params = 3 * (self.hidden_size * self.intermediate_size)

        # Norms per layer (RMSNorm: input and post_attention)
        norm_params = 2 * self.hidden_size

        layer_params = attn_params + mlp_params + norm_params
        total_layers_params = layer_params * self.num_hidden_layers

        # Final norm
        final_norm_params = self.hidden_size

        # LM Head (shared with embedding if tied)
        lm_head_params = 0 if self.tie_word_embeddings else (self.vocab_size * self.hidden_size)

        # Auxiliary confidence head (linear probe for calibrated score)
        confidence_head_params = (self.hidden_size + 1) if self.has_confidence_head else 0

        total_params = (
            embed_params
            + total_layers_params
            + final_norm_params
            + lm_head_params
            + confidence_head_params
        )

        return {
            "total_params": total_params,
            "embed_params": embed_params,
            "per_layer_params": layer_params,
            "layers_total_params": total_layers_params,
            "fp32_mb": (total_params * 4) / (1024 * 1024),
            "fp16_mb": (total_params * 2) / (1024 * 1024),
            "int8_mb": (total_params * 1) / (1024 * 1024),
            "int4_mb": (total_params * 0.5) / (1024 * 1024),
            "bit2_mb": (total_params * 0.25) / (1024 * 1024),
        }
