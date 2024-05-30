"""
The `components` package provides reusable components for building neural network architectures.

Submodules:
- `embeddings`: Contains classes for patch and positional embeddings.
- `transformer`: Contains classes for multi-head self-attention and transformer blocks.
"""

from .embeddings import PatchEmbedding, PositionalEmbedding
from .transformer import MultiHeadSelfAttention, TransformerBlock
