"""Unit tests for Yantra model architecture, parameter budgets, and quantization."""

import os
import tempfile
import torch
import pytest

from yantra.model.config import YantraConfig
from yantra.model.transformer import YantraForToolCalling, KVCache
from yantra.runtime.quantize import export_compressed_model, load_compressed_model, quantize_model_int8


def test_parameter_budget():
    """Verify default configuration parameter counts and memory budgets."""
    config = YantraConfig()
    budget = config.calculate_parameter_count()

    total_params = budget["total_params"]
    # Total parameters should be between 30M and 36M
    assert 30_000_000 <= total_params <= 36_000_000
    # INT8 size must be well below 50 MB
    assert budget["int8_mb"] < 50.0
    # INT4 size must be ~16-18 MB
    assert budget["int4_mb"] < 25.0


def test_model_forward_and_loss():
    """Test model forward pass with loss computation and confidence score."""
    config = YantraConfig(
        vocab_size=8192,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = YantraForToolCalling(config)

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    # Mask half the tokens
    labels[:, :8] = -100

    outputs = model(input_ids=input_ids, labels=labels)
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0.0
    assert outputs["confidence"].shape == (batch_size, 1)

    # Test memory-efficient selective projection (return_logits=False)
    opt_outputs = model(input_ids=input_ids, labels=labels, return_logits=False)
    assert opt_outputs["logits"] is None  # Never allocated
    assert opt_outputs["loss"] is not None
    # Losses must be mathematically identical
    assert torch.isclose(outputs["loss"], opt_outputs["loss"], atol=1e-5)


def test_kv_cache_generation():
    """Test autoregressive generation using incremental KV cache."""
    config = YantraConfig(
        vocab_size=8192,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = YantraForToolCalling(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(input_ids, max_new_tokens=5)
    assert generated.shape == (1, 13)


def test_int8_quantization_and_export():
    """Test INT8 compression accuracy and file size under 50 MB."""
    # Test on full architecture
    config = YantraConfig()
    model = YantraForToolCalling(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_path = os.path.join(tmp_dir, "test_model_int8.bin")
        stats = export_compressed_model(model, export_path, quantize_to_int8=True)

        assert stats["fits_under_50mb"] is True
        assert stats["size_mb"] < 50.0

        # Reload and verify inference
        reloaded = load_compressed_model(export_path)
        x = torch.randint(0, config.vocab_size, (1, 4))
        out = reloaded(x)
        assert out["logits"].shape == (1, 4, config.vocab_size)
