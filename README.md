# Transformer from Scratch

A GPT-2 style transformer implemented from scratch in PyTorch, following Andrej Karpathy's nanoGPT.

## Project Structure

```
├── model/
│   ├── config.py         # GPTConfig dataclass
│   ├── attention.py      # Causal self-attention with flash attention
│   ├── mlp.py            # Feed-forward MLP block
│   ├── block.py          # Transformer block (attention + MLP)
│   └── gpt.py            # GPT model + pretrained weight loader
├── data/
│   └── dataloader.py     # Lightweight batched token dataloader
├── train.py              # Training loop + text generation
└── train_gpt2.py         # Original monolithic file (reference)
```

## Model Architecture

- **CausalSelfAttention** — multi-head self-attention with causal masking via PyTorch's `scaled_dot_product_attention` (flash attention)
- **MLP** — two linear layers with GELU activation (tanh approximation)
- **Block** — pre-norm transformer block: LayerNorm → Attention, LayerNorm → MLP
- **GPT** — full GPT-2 model with token + positional embeddings and weight tying between input embedding and LM head

## GPT-2 Model Sizes

| Model        | Layers | Heads | Embedding | Params |
|-------------|--------|-------|-----------|--------|
| gpt2         | 12     | 12    | 768       | 124M   |
| gpt2-medium  | 24     | 16    | 1024      | 350M   |
| gpt2-large   | 36     | 20    | 1280      | 774M   |
| gpt2-xl      | 48     | 25    | 1600      | 1558M  |

## Usage

### Train from scratch

Place a `input.txt` file in the project root, then run:

```bash
python train.py
```

### Load pretrained GPT-2 weights

```python
from model import GPT

model = GPT.from_pretrained("gpt2")
```

## Requirements

```
torch
tiktoken
transformers
```

Install with:

```bash
pip install torch tiktoken transformers
```
