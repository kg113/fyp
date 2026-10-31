import os
from dataclasses import dataclass

@dataclass
class Base:
    clip_model: str = 'openai/clip-vit-large-patch14'
    text_model: str = 'gpt2-medium'
    seed: int = 100
    num_workers: int = 2
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 60
    lr: float = 2e-4  # Changed from 5e-3 (too high) to 1e-4 for stable fine-tuning
    k: float = 0.3
    batch_size_exp: int = 5
    ep_len: int = 4
    num_layers: int = 5
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.08
    weights_dir: str = os.path.join('weights', 'large')