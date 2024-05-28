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

        assert img_size % patch_size == 0, f"Image size {
            img_size} is not divisible by patch size {patch_size}."

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

        assert height == width, f"Expected square image, but got height {
            height} and width {width}."

        # Apply convolution to create patch embeddings
        # Output shape: (batch_size, embedding_size, height // patch_size, width // patch_size)
        X = self.conv(X)

        # Flatten the height and width dimensions into a single dimension to create patch tokens
        # Output shape: (batch_size, num_patches, embedding_size)
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

        assert embedding_size % num_heads == 0, f"Embedding size {
            embedding_size} is not divisible by number of attention heads {num_heads}."

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
        # Output shape: (batch_size, num_patches + 1, embedding_size)
        Q = self.queries(X)
        K = self.keys(X)
        V = self.values(X)

        # Split the embedding into self.num_heads different pieces
        # Output shape: (batch_size, num_heads, num_patches + 1, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # Output shape: (batch_size, num_heads, num_patches + 1, num_patches + 1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # Save the attention map
        # Output shape: (batch_size, num_heads, num_patches + 1, num_patches + 1)
        self.attention_map = attention

        # Compute the output
        # Output shape: (batch_size, num_heads, num_patches + 1, head_dim)
        out = torch.matmul(attention, V)
        # Output shape: (batch_size, num_patches + 1, num_heads, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        # Output shape: (batch_size, num_patches + 1, embedding_size)
        out = out.view(batch_size, -1, self.embedding_size)

        # Final linear projection
        # Output shape: (batch_size, num_patches + 1, embedding_size)
        out = self.fc_out(out)

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

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadSelfAttention(
            embedding_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_ratio * embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * embedding_size, embedding_size),
            nn.Dropout(dropout)
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
        self.position_embeddings = nn.Parameter(torch.randn(
            1, (img_size // patch_size) ** 2 + 1, embedding_size))

        # Sequence of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            VisionTransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

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
        # Output shape: (batch_size, num_patches, embedding_size)
        X = self.patch_embedding(X)

        # Expand and concatenate the class token to the input
        # Output shape: (batch_size, num_patches + 1, embedding_size)
        batch_size, num_patches, embedding_size = X.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)

        # Add position embeddings
        X = X + self.position_embeddings[:, :num_patches + 1, :]

        # Pass through the stack of Transformer blocks
        # Output shape: (batch_size, num_patches + 1, embedding_size)
        for transformer_block in self.transformer_blocks:
            X = transformer_block(X)

        # Get the classification token embedding
        # Output shape: (batch_size, embedding_size)
        X = X[:, 0]

        # Normalize and pass through fully connected layer
        # Output shape: (batch_size, num_classes)
        X = self.norm(X)
        X = self.fc(X)

        return X
