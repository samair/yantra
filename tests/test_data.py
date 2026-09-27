"""Unit tests for synthetic data generation and masked loss dataset."""

import pytest
import torch
from torch.utils.data import DataLoader

from yantra.data.dataset import ToolCallingCollator, ToolCallingDataset
from yantra.data.generator import SyntheticDataGenerator
from yantra.tokenizer.tokenization import YantraTokenizer


def test_synthetic_data_generation():
    """Verify synthetic dataset generator categories."""
    generator = SyntheticDataGenerator(seed=99)
    dataset = generator.generate_dataset(num_samples=40)

    assert len(dataset) == 40
    categories = {ex.category for ex in dataset}
    assert "single" in categories
    assert "multi" in categories
    assert "refusal" in categories
    assert "distractor" in categories

    for ex in dataset:
        assert isinstance(ex.tools, list)
        assert len(ex.query) > 0
        assert isinstance(ex.tool_calls, list)
        if ex.category == "refusal":
            assert len(ex.tool_calls) == 0


def test_dataset_masked_loss():
    """Verify that system prompt & query tokens are masked with -100."""
    generator = SyntheticDataGenerator(seed=12)
    examples = generator.generate_dataset(num_samples=4)
    tokenizer = YantraTokenizer.load()

    dataset = ToolCallingDataset(examples, tokenizer=tokenizer)
    collator = ToolCallingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert input_ids.shape == labels.shape
    # Ensure there are masked tokens (-100) and supervised tokens (> -100)
    for i in range(labels.shape[0]):
        row = labels[i]
        assert (row == -100).any(), "Must have masked tokens"
        assert (row != -100).any(), "Must have supervised assistant tokens"
