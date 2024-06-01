"""
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Implements the multi-head self-attention mechanism.

    Args:
        embedding_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate. Default is 0.0.
    """

    def __init__(
        self, embedding_size: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                f"Embedding size {embedding_size} is not divisible by number of attention heads {num_heads}."
            )

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.attention_map = None

        self.queries = nn.Linear(embedding_size, embedding_size)
        self.keys = nn.Linear(embedding_size, embedding_size)
        self.values = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention mechanism.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_patches + 1, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches + 1, embedding_size).
        """
        batch_size = X.shape[0]

        # Linear projections
        Q = self.queries(X)  # (batch_size, num_patches + 1, embedding_size)
        K = self.keys(X)  # (batch_size, num_patches + 1, embedding_size)
        V = self.values(X)  # (batch_size, num_patches + 1, embedding_size)

        # Split the embedding into self.num_heads different pieces
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, num_heads, num_patches + 1, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, num_heads, num_patches + 1, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (batch_size, num_heads, num_patches + 1, head_dim)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim**0.5
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(
            attention
        )  # (batch_size, num_heads, num_patches + 1, num_patches + 1)

        # Save the attention map
        self.attention_map = (
            attention  # (batch_size, num_heads, num_patches + 1, num_patches + 1)
        )

        # Compute the output
        out = torch.matmul(
            attention, V
        )  # (batch_size, num_heads, num_patches + 1, head_dim)
        out = out.permute(
            0, 2, 1, 3
        ).contiguous()  # (batch_size, num_patches + 1, num_heads, head_dim)
        out = out.view(
            batch_size, -1, self.embedding_size
        )  # (batch_size, num_patches + 1, embedding_size)

        # Final linear projection
        out = self.fc_out(out)  # (batch_size, num_patches + 1, embedding_size)

        return out


class TransformerBlock(nn.Module):
    """
    Implements a single block of the Transformer.

    Args:
        embedding_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Expansion ratio for the MLP.
        dropout (float, optional): Dropout rate. Default is 0.0.
    """

    def __init__(
        self, embedding_size: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0
    ) -> None:
        super(TransformerBlock, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadSelfAttention(embedding_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_ratio * embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_patches + 1, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches + 1, embedding_size).
        """
        # Compute multi-head self-attention with a residual connection
        X = X + self.attention(self.norm1(X))

        # MLP with residual connection
        X = X + self.mlp(self.norm2(X))

        return X
