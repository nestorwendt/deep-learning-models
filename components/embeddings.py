"""
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projects each patch into an embedding space.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        embedding_size (int): Dimension of the embedding space.
        in_channels (int, optional): Number of input channels. Default is 3.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embedding_size: int,
        in_channels: int = 3,
    ) -> None:
        super(PatchEmbedding, self).__init__()

        assert (
            img_size % patch_size == 0
        ), f"Image size {img_size} is not divisible by patch size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the patch embedding.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embedding_size).
        """
        batch_size, channels, height, width = X.shape

        assert (
            height == width
        ), f"Expected square image, but got height {height} and width {width}."

        # Apply convolution to create patch embeddings
        X = self.conv(
            X
        )  # (batch_size, embedding_size, height // patch_size, width // patch_size)

        # Flatten the height and width dimensions into a single dimension to create patch tokens
        X = X.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_size)

        return X


class PositionalEmbedding(nn.Module):
    """
    Adds positional information to the patch embeddings.

    Args:
        num_patches (int): Number of patches.
        embedding_size (int): Dimension of the embedding space.
    """

    def __init__(self, num_patches: int, embedding_size: int) -> None:
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embedding_size).

        Returns:
            torch.Tensor: Output tensor with positional information added.
        """
        return x + self.pos_embed
