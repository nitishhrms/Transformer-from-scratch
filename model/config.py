from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12        # number of transformer layers
    n_head: int = 12         # number of attention heads
    n_embd: int = 768        # embedding dimension
