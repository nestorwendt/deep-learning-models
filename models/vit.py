"""
"""

import torch
import torch.nn as nn
from components.embeddings import PatchEmbedding, PositionalEmbedding
from components.transformer import TransformerBlock


class ViT(nn.Module):
    """
    Implements the Vision Transformer (ViT) model.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        num_classes (int): Number of output classes.
        embedding_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer blocks.
        mlp_ratio (int): Expansion ratio for the MLP.
        in_channels (int, optional): Number of input channels. Default is 3.
        dropout (float, optional): Dropout rate. Default is 0.0.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: int,
        in_channels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.in_channels = in_channels
        self.dropout = dropout

        self.num_patches = (img_size // patch_size) ** 2

        # Initialize the patch embedding layer
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embedding_size=embedding_size,
            in_channels=in_channels,
        )

        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))

        # Learnable position embeddings
        self.position_embeddings = PositionalEmbedding(
            self.num_patches + 1, embedding_size
        )

        # Sequence of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size=embedding_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization and final fully connected layer
        self.norm = nn.LayerNorm(embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Apply patch embedding
        X = self.patch_embedding(X)  # (batch_size, num_patches, embedding_size)

        # Expand and concatenate the class token to the input
        batch_size, num_patches, embedding_size = X.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat(
            (cls_tokens, X), dim=1
        )  # (batch_size, num_patches + 1, embedding_size)

        # Add position embeddings
        X = X + self.position_embeddings(X)

        # Pass through the stack of Transformer blocks
        for transformer_block in self.transformer_blocks:
            X = transformer_block(X)  # (batch_size, num_patches + 1, embedding_size)

        # Get the classification token embedding
        X = X[:, 0]  # (batch_size, embedding_size)

        # Normalize and pass through fully connected layer
        X = self.norm(X)
        X = self.fc(X)  # (batch_size, num_classes)

        return X


def make_vit_base(
    img_size: int, patch_size: int, num_classes: int, dropout: float = 0.1
) -> ViT:
    """
    Constructs a ViT-B model.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Default is 0.1.

    Returns:
        ViT: A Vision Transformer model.
    """
    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_size=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4,
        in_channels=3,
        dropout=dropout,
    )


def make_vit_large(
    img_size: int, patch_size: int, num_classes: int, dropout: float = 0.1
) -> ViT:
    """
    Constructs a ViT-L model.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Default is 0.1.

    Returns:
        ViT: A Vision Transformer model.
    """
    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_size=1024,
        num_heads=16,
        num_layers=24,
        mlp_ratio=4,
        in_channels=3,
        dropout=dropout,
    )


def make_vit_huge(
    img_size: int, patch_size: int, num_classes: int, dropout: float = 0.1
) -> ViT:
    """
    Constructs a ViT-H model.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout rate. Default is 0.1.

    Returns:
        ViT: A Vision Transformer model.
    """
    return ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_size=1280,
        num_heads=16,
        num_layers=32,
        mlp_ratio=4,
        in_channels=3,
        dropout=dropout,
    )
