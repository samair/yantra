"""Supervised Fine-Tuning (SFT) Trainer for Yantra Tool Dispatching Model."""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from yantra.data.dataset import ToolCallingCollator, ToolCallingDataset
from yantra.model.config import YantraConfig
from yantra.model.transformer import YantraForToolCalling
from yantra.tokenizer.tokenization import YantraTokenizer


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for training runs."""

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_epochs: int = 3
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    confidence_loss_weight: float = 0.1
    eval_every_steps: int = 50
    empty_cache_every_steps: int = 25
    save_dir: str = "checkpoints"
    device: Optional[str] = None

    def get_device(self) -> torch.device:
        if self.device:
            dev = torch.device(self.device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

        # Configure MPS memory pool on Apple Silicon to prevent aggressive RAM hoarding
        if dev.type == "mps":
            import os
            if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        return dev


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Cosine learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class YantraTrainer:
    """End-to-end trainer for Yantra model checkpoints."""

    def __init__(
        self,
        model: YantraForToolCalling,
        config: TrainingConfig,
        train_dataset: ToolCallingDataset,
        eval_dataset: Optional[ToolCallingDataset] = None,
        tokenizer: Optional[YantraTokenizer] = None,
    ):
        self.model = model
        self.config = config
        self.device = config.get_device()
        self.model.to(self.device)

        self.tokenizer = tokenizer or YantraTokenizer.load()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.collator = ToolCallingCollator(pad_token_id=self.tokenizer.pad_token_id)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collator,
            )
        else:
            self.eval_loader = None

        # Optimizer with weight decay excluding bias and layernorms
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name.lower() or "bias" in name.lower():
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optim_groups, lr=self.config.learning_rate)

        total_steps = (len(self.train_loader) // self.config.gradient_accumulation_steps) * self.config.max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=max(1, total_steps),
        )

    def train_epoch(self, epoch_idx: int) -> float:
        """Run single training epoch with gradient accumulation and masked loss."""
        self.model.train()
        total_loss = 0.0
        steps = 0
        self.optimizer.zero_grad()

        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # Log every 10%

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            conf_target = batch["confidence_target"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,  # Causal mask handled in attention
                labels=labels,
                return_logits=False,  # Memory optimization: compute loss selectively
            )

            lm_loss = outputs["loss"]
            loss = lm_loss

            # Confidence head auxiliary BCE loss with numerical clamp
            if outputs["confidence"] is not None and self.config.confidence_loss_weight > 0:
                clamped_conf = outputs["confidence"].clamp(min=1e-7, max=1.0 - 1e-7)
                conf_loss = F.binary_cross_entropy(clamped_conf, conf_target)
                loss = loss + self.config.confidence_loss_weight * conf_loss

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            steps += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Periodic cache clearing to prevent Metal memory pool bloat
            if self.config.empty_cache_every_steps > 0 and (step + 1) % self.config.empty_cache_every_steps == 0:
                if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                elif self.device.type == "cuda" and hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

            # Live step progress with flush=True
            if (step + 1) % log_interval == 0 or (step + 1) == total_batches:
                curr_loss = total_loss / max(1, steps)
                pct = int(100 * (step + 1) / total_batches)
                print(f"  [Epoch {epoch_idx}] Step {step+1}/{total_batches} ({pct}%) | Loss: {curr_loss:.4f}", flush=True)

        return total_loss / max(1, steps)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Compute evaluation metrics on validation set."""
        if not self.eval_loader:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels, return_logits=False)
            total_loss += outputs["loss"].item()
            total_batches += 1

        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.device.type == "cuda" and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))

        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }

    def train(self) -> Dict[str, Any]:
        """Run full training pipeline and save best checkpoints."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        history = []

        print(f"\nStarting {self.config.max_epochs} Epochs of Training on {self.device}...", flush=True)
        for epoch in range(1, self.config.max_epochs + 1):
            print(f"\n>>> Epoch {epoch}/{self.config.max_epochs} started...", flush=True)
            start_t = time.time()
            train_loss = self.train_epoch(epoch)
            eval_metrics = self.evaluate()
            elapsed = time.time() - start_t

            eval_loss = eval_metrics.get("eval_loss", 0.0)
            ppl = eval_metrics.get("perplexity", 0.0)
            print(f">>> Epoch {epoch}/{self.config.max_epochs} Finished in {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | PPL: {ppl:.2f}", flush=True)

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "elapsed_sec": elapsed,
                **eval_metrics,
            }
            history.append(log_dict)

        # Save final checkpoint
        checkpoint_path = os.path.join(self.config.save_dir, "yantra_latest.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config.__dict__,
            },
            checkpoint_path,
        )
        print(f"\nCheckpoint successfully saved to {checkpoint_path}", flush=True)

        return {"history": history, "checkpoint_path": checkpoint_path}
