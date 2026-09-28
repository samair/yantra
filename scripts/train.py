"""CLI Training script for Yantra Tool Dispatching Model."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yantra import (
    SyntheticDataGenerator,
    ToolCallingDataset,
    TrainingConfig,
    YantraConfig,
    YantraForToolCalling,
    YantraTokenizer,
    YantraTrainer,
    export_compressed_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train Yantra Sub-50MB Tool Dispatching Model")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic training examples to generate")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=384, help="Maximum sequence length limit")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--export_name", type=str, default="yantra_int8.bin", help="Export filename")
    args = parser.parse_args()

    print(f"=== Yantra SFT Training Pipeline ===", flush=True)
    print(f"Generating {args.samples} synthetic tool calling examples...", flush=True)
    generator = SyntheticDataGenerator(seed=42)
    dataset_records = generator.generate_dataset(num_samples=args.samples)

    split_idx = int(len(dataset_records) * 0.9)
    train_data = dataset_records[:split_idx]
    eval_data = dataset_records[split_idx:]
    print(f"Train samples: {len(train_data)}, Validation samples: {len(eval_data)}", flush=True)

    tokenizer = YantraTokenizer.load()
    train_dataset = ToolCallingDataset(train_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    eval_dataset = ToolCallingDataset(eval_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    config = YantraConfig()
    budget = config.calculate_parameter_count()
    print(f"Model parameters: {budget['total_params']:,} (~{budget['fp16_mb']:.1f} MB in FP16, ~{budget['int8_mb']:.1f} MB in INT8)", flush=True)

    model = YantraForToolCalling(config)

    train_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.epochs,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer = YantraTrainer(
        model=model,
        config=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print(f"Starting training on device: {train_config.get_device()}...", flush=True)
    train_res = trainer.train()

    print("\n=== Training Summary ===", flush=True)
    for log in train_res["history"]:
        eval_loss = log.get("eval_loss", 0.0)
        ppl = log.get("perplexity", 0.0)
        print(f"Epoch {log['epoch']}: Train Loss = {log['train_loss']:.4f}, Eval Loss = {eval_loss:.4f}, PPL = {ppl:.2f} ({log['elapsed_sec']:.1f}s)", flush=True)

    # INT8 Quantization and Export
    export_path = Path(args.save_dir) / args.export_name
    print(f"\nCompressing to INT8 per-channel weights: {export_path}...", flush=True)
    export_info = export_compressed_model(model, export_path, quantize_to_int8=True)
    print(f"Compressed file size: {export_info['size_mb']:.2f} MB", flush=True)
    print(f"Fits under 50 MB threshold: {export_info['fits_under_50mb']}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
