"""INT8 Compression, Weight Slicing, and Serialization for Yantra (< 50MB)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from yantra.model.config import YantraConfig
from yantra.model.transformer import GroupedQueryAttention, SwiGLUMLP, YantraDecoderLayer, YantraForToolCalling


class Int8Linear(nn.Module):
    """Portable per-channel symmetric INT8 quantized linear layer with dynamic dequantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Int8Linear":
        """Convert a standard nn.Linear to symmetric INT8 with per-channel scales."""
        qlinear = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.float()
            # Symmetric per-channel scale: max(abs(w)) / 127
            scale = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
            q_w = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)

            qlinear.weight_int8.copy_(q_w)
            qlinear.scale.copy_(scale)
            if linear.bias is not None:
                qlinear.bias.copy_(linear.bias)
        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically scale int8 weights during projection
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias)


def quantize_model_int8(model: YantraForToolCalling) -> YantraForToolCalling:
    """Quantize all attention and feed-forward linear layers of Yantra to INT8."""
    model.eval()

    for layer in model.layers:
        # 1. Quantize Attention projections
        layer.self_attn.q_proj = Int8Linear.from_float(layer.self_attn.q_proj)
        layer.self_attn.k_proj = Int8Linear.from_float(layer.self_attn.k_proj)
        layer.self_attn.v_proj = Int8Linear.from_float(layer.self_attn.v_proj)
        layer.self_attn.o_proj = Int8Linear.from_float(layer.self_attn.o_proj)

        # 2. Quantize SwiGLU MLP projections
        layer.mlp.gate_proj = Int8Linear.from_float(layer.mlp.gate_proj)
        layer.mlp.up_proj = Int8Linear.from_float(layer.mlp.up_proj)
        layer.mlp.down_proj = Int8Linear.from_float(layer.mlp.down_proj)

    # If LM Head is not tied to embeddings, quantize LM head as well
    if not model.config.tie_word_embeddings and isinstance(model.lm_head, nn.Linear):
        model.lm_head = Int8Linear.from_float(model.lm_head)

    return model


def export_compressed_model(
    model: YantraForToolCalling,
    output_path: Union[str, Path],
    quantize_to_int8: bool = True,
) -> Dict[str, Any]:
    """Export compressed model checkpoint and verify binary size threshold (< 50MB)."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if quantize_to_int8:
        model = quantize_model_int8(model)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "quantized": quantize_to_int8,
        },
        str(target_path),
    )

    size_bytes = target_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return {
        "output_path": str(target_path),
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "fits_under_50mb": size_mb < 50.0,
    }


def load_compressed_model(checkpoint_path: Union[str, Path], device: str = "cpu") -> YantraForToolCalling:
    """Load and re-instantiate a quantized or base Yantra checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config_dict = checkpoint.get("config", {})
    config = YantraConfig(**config_dict)

    model = YantraForToolCalling(config)
    is_quantized = checkpoint.get("quantized", False)

    if is_quantized:
        model = quantize_model_int8(model)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
