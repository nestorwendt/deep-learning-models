"""
This module implements a Vision Transformer (ViT) in PyTorch, which includes:
1. PatchEmbedding: Splits an image into patches and projects each patch into an embedding space.
2. MultiHeadSelfAttention: Implements the multi-head self-attention mechanism.
3. VisionTransformerBlock: Implements a single block of the Vision Transformer.
4. VisionTransformer: Combines the above components into a complete Vision Transformer model.
5. plot_attention_maps: Utility function to plot the attention maps of the Vision Transformer.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision import transforms


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

        assert (
            embedding_size % num_heads == 0
        ), f"Embedding size {embedding_size} is not divisible by number of attention heads {num_heads}."

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


class VisionTransformerBlock(nn.Module):
    """
    Implements a single block of the Vision Transformer.

    Args:
        embedding_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Expansion ratio for the MLP.
        dropout (float, optional): Dropout rate. Default is 0.0.
    """

    def __init__(
        self, embedding_size: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0
    ) -> None:
        super(VisionTransformerBlock, self).__init__()

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
        Forward pass for the Vision Transformer block.

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


class VisionTransformer(nn.Module):
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
        super(VisionTransformer, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.in_channels = in_channels
        self.dropout = dropout

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
        self.position_embeddings = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2 + 1, embedding_size)
        )

        # Sequence of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
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
        X = X + self.position_embeddings[:, : num_patches + 1, :]

        # Pass through the stack of Transformer blocks
        for transformer_block in self.transformer_blocks:
            X = transformer_block(X)  # (batch_size, num_patches + 1, embedding_size)

        # Get the classification token embedding
        X = X[:, 0]  # (batch_size, embedding_size)

        # Normalize and pass through fully connected layer
        X = self.norm(X)
        X = self.fc(X)  # (batch_size, num_classes)

        return X

    def plot_attention_maps(
        self,
        X: torch.Tensor,
        save_path: str,
        transform: Optional[transforms.Compose] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        """
        Plot the original images along with attention maps for each layer in the Vision Transformer.

        Args:
            X (torch.Tensor): Batch of input images of shape (batch_size, channels, height, width).
            save_path (str): Path to save the images.
            transform (Optional[torchvision.transforms.Compose]): A torchvision.transforms.Compose object for transforming the input images. Default is None.
            interpolation (Optional[str]): Interpolation method for displaying attention maps. Default is None.
        """
        assert (
            self.embedding_size % self.num_heads == 0
        ), f"Embedding size {self.embedding_size} is not divisible by number of attention heads {self.num_heads}."
        assert (
            self.img_size % self.patch_size == 0
        ), f"Image size {self.img_size} is not divisible by patch size {self.patch_size}."

        # Extract mean and std from transforms.Normalize
        mean, std = None, None
        if transform is not None:
            for t in transform.transforms:
                if isinstance(t, transforms.Normalize):
                    mean = torch.tensor(t.mean).unsqueeze(1).unsqueeze(2).to(X.device)
                    std = torch.tensor(t.std).unsqueeze(1).unsqueeze(2).to(X.device)
                    break

        # Extract batch size and number of patches in one dimension
        batch_size = X.size(0)
        num_patches_x = self.img_size // self.patch_size

        # Get the attention maps for the first class token only
        attention_maps = [
            block.attention.attention_map[:, :, 0, 1:].reshape(
                batch_size, -1, num_patches_x, num_patches_x
            )
            for block in self.transformer_blocks
        ]

        # Iterate over the attention layers
        for layer_i, layer_attention in enumerate(attention_maps):
            plt.figure(figsize=(self.num_heads * 2, batch_size * 2))
            plt.suptitle(f"Layer {len(self.transformer_blocks) - 1}", fontsize=16)

            num_rows = batch_size
            num_cols = self.num_heads + 1

            # Plot the original images with labels
            for batch_i, image in enumerate(X):
                # Denormalize image for plotting
                if mean is not None and std is not None:
                    image = (image * std + mean).cpu().numpy()

                plt.subplot(num_rows, num_cols, batch_i * num_cols + 1)
                plt.imshow(image.transpose(1, 2, 0))
                plt.title(f"Image {batch_i + 1}")
                plt.axis("off")

            # Plot attention maps for each head with labels
            for head_i in range(self.num_heads):
                for batch_i in range(batch_size):
                    plt.subplot(num_rows, num_cols, batch_i * num_cols + head_i + 2)
                    plt.imshow(
                        layer_attention[batch_i, head_i].cpu().detach().numpy(),
                        cmap="viridis",
                        interpolation=interpolation,
                    )
                    if batch_i == 0:  # Only label the first row
                        plt.title(f"Head {head_i + 1}")
                    plt.axis("off")

            # Adjust layout to make room for the title
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            # Save the image for the current layer
            filename = f"layer_{layer_i + 1}_attention_maps.png"
            save_file = os.path.join(save_path, filename)
            plt.savefig(save_file)
            plt.close()
