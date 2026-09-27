# Yantra: Sub-50 MB Tool Calling Specialist Model

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org)
[![Weight Size](https://img.shields.io/badge/Binary%20Size-45%20MB%20(INT8)-brightgreen.svg)]()
[![Model Parameters](https://img.shields.io/badge/Parameters-34.35M-blueviolet.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)]()

**Yantra** (यन्त्र — *instrument / machine*) is a lightweight, on-device foundation model engineered exclusively for **function calling, argument slot filling, and structured tool dispatching** within a total weight footprint under **50 MB**.

Yantra strips away general conversational chit-chat to specialize entirely on high-accuracy function selection and typed argument binding.

---

## Key Features

- **Sub-50 MB Binary Footprint**: 34.35M parameters. Base FP16 is ~65.5 MB, and per-channel INT8 quantization compresses to **45.03 MB** for zero-copy memory-mapped on-device execution.
- **Grouped-Query Attention (GQA)**: 8 Query heads, 2 KV heads, reducing KV cache footprint to $< 2$ MB during edge inference.
- **Compact Domain BPE Tokenizer**: Exact 8,192 vocabulary with dedicated wire tokens (`<tools>`, `<tool_call>`, `<think>`, `<|im_start|>`, `<|im_end|>`).
- **SwiGLU & Rotary Position Embeddings (RoPE)**: High-capacity non-linear representation without parameter bloat.
- **Tied Word Embeddings**: Embeddings and LM output projection share weights, saving ~8.4 MB of parameter budget.
- **Masked Cross-Entropy Training**: Loss is computed **strictly** on assistant tokens; system prompts and user schemas are masked with `-100`.
- **Auxiliary Confidence Head**: Calibrated confidence scoring to distinguish valid dispatches from refusals and hard negatives.
- **Deterministic Grammar Repair & Argument Grounding**: Validates that all dispatched parameters are grounded in the user's prompt or schema defaults.

---

## Parameter Budget & Architecture

| Component | Hyperparameter / Setting | Parameter Count | Weight Footprint |
| :--- | :--- | :--- | :--- |
| **Vocabulary ($V$)** | 8,192 (Domain BPE) | $8,192 \times 512$ | ~8.4 MB (FP16), ~4.2 MB (INT8) |
| **Hidden Dimension ($d$)** | 512 | — | Lean edge CPU activations |
| **Layers ($L$)** | 10 decoder layers | — | Balances depth & CPU latency |
| **Attention** | GQA (8 Query, 2 KV heads, dim 64) | 786,432 per layer | ~7.8M total params |
| **Feed-Forward (FFN)** | SwiGLU ($d_{ffn} = 1536$) | 2,359,296 per layer | ~23.6M total params |
| **RMSNorm** | $\epsilon = 10^{-6}$ (pre-layer + final) | 10,752 params | < 0.1 MB |
| **LM Head** | Tied with Token Embeddings | 0 (tied) | Saves 8.4 MB |
| **Confidence Head** | Linear probe on final hidden state | 513 params | < 0.01 MB |
| **Total Parameter Count** | — | **34,352,129 parameters** | **Base FP16: ~65.5 MB** |
| **Compressed (INT8)** | Symmetric per-channel | — | **45.03 MB** (< 50 MB threshold) |

---

## Wire Protocol

```text
<s><|im_start|>system
date: 2026-09-27 Sun 17:00<|im_end|>
<|im_start|>user
<tools>[{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]</tools>
What is the weather in Tokyo?<|im_end|>
<|im_start|>assistant
<think>Intent matches 'get_weather' with city: Tokyo</think>
<tool_call>[{"name":"get_weather","arguments":{"city":"Tokyo"}}]</tool_call><|im_end|></s>
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/sameerchandra/yantra.git
cd yantra
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. High-Level Python SDK

```python
import yantra

# 1. Define tools using @yantra.tool
@yantra.tool
def get_weather(city: str) -> str:
    """Fetch current real-time weather conditions for a city."""
    return f"Weather in {city}: 72F, Sunny"

@yantra.tool
def set_thermostat(temperature: int, mode: str = "auto") -> str:
    """Set HVAC temperature in degrees Fahrenheit."""
    return f"Thermostat set to {temperature}F ({mode})"

# 2. Instantiate Agent
agent = yantra.Agent(tools=[get_weather, set_thermostat])

# 3. Dispatch user requests
response = agent.run("What's the weather like in Tokyo right now?")
print(response.tool_calls)
# -> [{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}]
print(response.results)
# -> ['Weather in Tokyo: 72F, Sunny']
```

### 3. Generate Synthetic Dataset

```python
from yantra import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
examples = generator.generate_dataset(num_samples=5000)
generator.save_jsonl(examples, "data/train.jsonl")
```

The data engine balances four key distributions:
1. **Single-tool Positive Calls (50%)**
2. **Multi-tool Parallel Calls (15%)**
3. **Hard Negatives & Refusals (25%)** — prompts requiring `<tool_call>[]</tool_call>`
4. **Distractor Resistance (10%)** — 4–6 semantic distractors accompanying the target tool

### 4. Train with Masked Cross-Entropy

```bash
python scripts/train.py --samples 2000 --epochs 3 --lr 3e-4 --batch_size 8 --device cpu
```

### 5. Compress & Export to INT8 (< 50 MB)

```python
from yantra import YantraConfig, YantraForToolCalling, export_compressed_model

model = YantraForToolCalling(YantraConfig())
stats = export_compressed_model(model, "checkpoints/yantra_35m_int8.bin", quantize_to_int8=True)
print(f"Compressed file size: {stats['size_mb']:.2f} MB")
# -> Compressed file size: 45.03 MB
```

---

## Project Structure

```text
yantra/
├── pyproject.toml              # Build specifications and dependencies
├── requirements.txt            # Core pinned requirements (PyTorch, Tokenizers, Pydantic)
├── README.md                   # System documentation and blueprints
├── yantra/
│   ├── __init__.py             # Public module exports
│   ├── model/
│   │   ├── config.py           # YantraConfig dataclass & parameter sizing calculations
│   │   └── transformer.py      # Decoder backbone, GQA, RoPE, SwiGLU, KVCache
│   ├── tokenizer/
│   │   ├── tokenization.py     # 8k BPE tokenizer, wire formatters & response parsers
│   │   └── pretrained/         # Cached tokenizer binaries
│   ├── data/
│   │   ├── schema.py           # @tool decorator, Pydantic/JSON schema extraction, registry
│   │   ├── generator.py        # Synthetic positive, multi-tool, refusal & distractor generator
│   │   └── dataset.py          # PyTorch Dataset with assistant-only loss masking & collator
│   ├── training/
│   │   └── trainer.py          # SFT trainer, AdamW, cosine schedule, multi-task loss
│   └── runtime/
│       ├── grammar.py          # JSON grammar repair & argument grounding checks
│       ├── quantize.py         # Portable per-channel INT8 quantization (< 50MB)
│       └── agent.py            # High-level developer SDK and execution agent
├── scripts/
│   ├── train.py                # CLI SFT training runner
│   └── demo.py                 # Interactive demonstration
└── tests/
    ├── test_model.py           # Architecture, GQA, KV cache, and parameter budget tests
    ├── test_schema.py          # Schema extraction and tool registry tests
    ├── test_tokenizer.py       # 8,192 BPE encoding, decoding, and wire formatting tests
    ├── test_data.py            # Synthetic generation and masked loss tests
    └── test_runtime.py         # JSON repair, argument grounding, and agent execution tests
```

---

## Running Unit Tests

Run the comprehensive test suite covering all modules:

```bash
pytest -v tests/
```

All 14 unit tests validate:
- Parameter counts ($34.35\text{M}$) and weight sizes ($< 50\text{ MB}$).
- Autoregressive KV cache generation.
- Symmetric per-channel INT8 quantization accuracy ($> 0.999$ cosine similarity).
- Wire format tokenization and exact 8,192 vocabulary layout.
- Masked cross-entropy assistant supervision.
- Grammar repair and argument grounding.

---

## License

Apache License 2.0.
