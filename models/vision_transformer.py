import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projects each patch into an embedding space.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        embedding_size (int): Dimension of the embedding space.
        in_channels (int, optional): Number of input channels. Default is 3 for RGB images.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embedding_size: int,
        in_channels: int = 3,
    ) -> None:
        super(PatchEmbedding, self).__init__()

        assert img_size % patch_size == 0, f"Image size {
            img_size} is not divisible by patch size {patch_size}."

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
            X (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, embedding_size).
        """
        B, C, H, W = X.shape
        assert H == W, f"Expected square image, but got height {
            H} and width {W}."

        # Apply convolution to create patch embeddings
        # Shape: (B, embedding_size, H//patch_size, W//patch_size)
        X = self.conv(X)

        # Flatten the height and width dimensions into a single dimension
        # Shape: (B, num_patches, embedding_size)
        X = X.flatten(2).transpose(1, 2)

        return X


class MultiHeadSelfAttention(nn.Module):
    """
    Implements the multi-head self-attention mechanism.

    Args:
        embedding_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate. Default is 0.0.
    """

    def __init__(self, embedding_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert self.head_dim * num_heads == embedding_size, f"Embedding size {
            embedding_size} is not divisible by number of heads {num_heads}."

        self.queries = nn.Linear(embedding_size, embedding_size)
        self.keys = nn.Linear(embedding_size, embedding_size)
        self.values = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-head self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches + 1, embedding_size).
        """
        batch_size = x.shape[0]

        # Linear projections
        Q = self.queries(x)  # Shape: (B, num_patches + 1, embedding_size)
        K = self.keys(x)     # Shape: (B, num_patches + 1, embedding_size)
        V = self.values(x)   # Shape: (B, num_patches + 1, embedding_size)

        # Split the embedding into self.num_heads different pieces
        # Shape: (B, num_heads, num_patches + 1, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        # Shape: (B, num_heads, num_patches + 1, head_dim)
        K = K.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        # Shape: (B, num_heads, num_patches + 1, head_dim)
        V = V.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # Shape: (B, num_heads, num_patches + 1, num_patches + 1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        # Shape: (B, num_heads, num_patches + 1, num_patches + 1)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Compute the output
        # Shape: (B, num_heads, num_patches + 1, head_dim)
        out = torch.matmul(attention, V)
        # Shape: (B, num_patches + 1, num_heads, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        # Shape: (B, num_patches + 1, embedding_size)
        out = out.view(batch_size, -1, self.embedding_size)

        # Final linear projection
        out = self.fc_out(out)  # Shape: (B, num_patches + 1, embedding_size)

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

    def __init__(self, embedding_size: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0) -> None:
        super(VisionTransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(
            embedding_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_ratio * embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, num_patches + 1, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches + 1, embedding_size).
        """
        # Residual connection for attention layer
        x = x + self.attention(self.norm1(x))
        # Residual connection for MLP layer
        x = x + self.mlp(self.norm2(x))

        return x


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
        in_channels (int, optional): Number of input channels. Default is 3 for RGB images.
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

        assert img_size % patch_size == 0, f"Image size {
            img_size} is not divisible by patch size {patch_size}."

        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            embedding_size=embedding_size,
            in_channels=in_channels,
        )

        # Learnable classification token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.position_embeddings = nn.Parameter(torch.randn(
            # Learnable position embeddings
            1, (img_size // patch_size)**2 + 1, embedding_size))

        self.transformer_blocks = nn.ModuleList([
            VisionTransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes).
        """
        # Apply patch embedding
        x = self.patch_embedding(x)

        B, N, _ = x.shape
        # Expand and concatenate the class token to the input
        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        # Add position embeddings
        x = x + self.position_embeddings[:, :N+1, :]

        # Pass through the stack of Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Take the class token only
        x = x[:, 0]
        x = self.norm(x)
        x = self.fc(x)

        return x
