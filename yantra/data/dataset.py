"""PyTorch Dataset and DataCollator with Masked Cross-Entropy Loss for Tool Dispatching."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from yantra.data.generator import SyntheticExample
from yantra.tokenizer.tokenization import YantraTokenizer


class ToolCallingDataset(Dataset):
    """PyTorch Dataset for supervised tool calling with assistant token loss masking."""

    def __init__(
        self,
        data: Union[str, Path, List[Dict[str, Any]], List[SyntheticExample]],
        tokenizer: YantraTokenizer,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict[str, Any]] = []

        if isinstance(data, (str, Path)):
            with open(data, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, SyntheticExample):
                    self.examples.append({
                        "tools": item.tools,
                        "query": item.query,
                        "thought": item.thought,
                        "tool_calls": item.tool_calls,
                        "category": item.category,
                    })
                elif isinstance(item, dict):
                    self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        tools = ex["tools"]
        query = ex["query"]
        thought = ex.get("thought")
        tool_calls = ex.get("tool_calls", [])

        # 1. Format prompt and target strings
        prompt_str = self.tokenizer.format_prompt(tools, query)
        target_str = self.tokenizer.format_target(thought, tool_calls)

        # 2. Tokenize prompt and target
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=True)
        # Target tokens do not re-add BOS
        target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)

        input_ids = prompt_ids + target_ids
        # Truncate if exceeds max length
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]

        # 3. Create masked labels: prompt tokens -> -100, target tokens -> input_ids
        num_prompt_tokens = min(len(prompt_ids), len(input_ids))
        labels = [-100] * num_prompt_tokens + input_ids[num_prompt_tokens:]

        # 4. Confidence target (1.0 for valid call dispatch, 0.0 for refusal)
        is_tool_call = 1.0 if len(tool_calls) > 0 else 0.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "confidence_target": torch.tensor([is_tool_call], dtype=torch.float32),
        }


class ToolCallingCollator:
    """Collate and dynamically pad batches for Yantra training."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        confidences = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            # Pad input_ids with pad_token_id
            padded_input = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ])
            # Pad labels with -100 (ignored in loss computation)
            padded_label = torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
            # Attention mask: 1 for tokens, 0 for pad
            attention_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool),
            ])

            padded_input_ids.append(padded_input)
            padded_labels.append(padded_label)
            attention_masks.append(attention_mask)
            confidences.append(item["confidence_target"])

        return {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "confidence_target": torch.stack(confidences, dim=0),
        }
