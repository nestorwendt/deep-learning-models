"""
This module provides functions to process images and visualize attention maps from a model.
Functions included:
    - convert_to_rgb: Converts a grayscale image to RGB.
    - normalize_image: Normalizes an image tensor to the range [0, 1].
    - get_attention_maps: Extracts attention maps from the model.
    - plot_attention_maps: Plots the attention maps for a sample batch of images.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from typing import Optional


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a grayscale image to RGB if necessary.

    Args:
        img (Image.Image): Input image.

    Returns:
        Image.Image: RGB image.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected img to be a PIL Image, but got {type(img)}")

    img = img.convert("RGB") if img.mode != "RGB" else img

    return img


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor to the range [0, 1].

    Args:
        image (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    image = (image - image.min()) / (image.max() - image.min())
    image = image.permute(1, 2, 0)

    return image


def get_attention_maps(model: nn.Module) -> torch.Tensor:
    """
    Extract the attention maps from the model after a forward pass.

    Args:
        model (nn.Module): The model to extract attention maps from.

    Returns:
        torch.Tensor: A tensor of attention maps for each image in the batch.
    """
    if not hasattr(model, "transformer_blocks"):
        raise ValueError("Model does not have transformer_blocks attribute")

    attention_maps = []
    for block in model.transformer_blocks:
        if (
            not hasattr(block.attention, "attention_map")
            or block.attention.attention_map is None
        ):
            raise ValueError(
                "Model does not have attention_map attribute or it is None"
            )
        attention_maps.append(block.attention.attention_map)

    # Stack the attention maps along a new dimension to create a tensor
    attention_maps = torch.stack(attention_maps, dim=0)
    attention_maps = attention_maps.permute(
        1, 0, 2, 3, 4
    )  # (batch_size, num_layers, num_heads, num_patches + 1, num_patches + 1)

    return attention_maps


def plot_attention_maps(
    batch: torch.Tensor,
    attention_maps: torch.Tensor,
    save_path: Optional[str] = None,
    imshow_interpolation: Optional[str] = None,
) -> None:
    """
    Plot the attention maps for a batch of images and optionally save the images.

    Args:
        batch (torch.Tensor): The batch of images.
        attention_maps (torch.Tensor): The attention maps for the batch.
        save_path (str, optional): Path to save the images. If None, the images will be shown. Defaults to None.
        imshow_interpolation (str, optional): Interpolation method for imshow. Defaults to None.
    """

    num_blocks = attention_maps.size(1)
    num_heads = attention_maps.size(2)

    # Iterate over images in the batch
    for image_idx, image in enumerate(batch):
        processed_image = normalize_image(image)

        fig, axes = plt.subplots(
            num_blocks + 1, num_heads, figsize=(num_heads * 3, (num_blocks + 1) * 3)
        )

        # Plot the original image
        if num_heads == 1:
            axes = np.expand_dims(
                axes, axis=1
            )  # Ensure axes is always 2D for consistent indexing
        axes[0, 0].imshow(processed_image)
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Original Image")
        for ax in axes[0, 1:]:
            ax.axis("off")

        # Plot the attention maps
        for block_idx in range(num_blocks):
            block_maps = attention_maps[image_idx, block_idx]
            for head_idx in range(num_heads):
                attention_map = block_maps[head_idx, 0, 1:]
                num_patches_x = int(np.sqrt(attention_map.numel()))
                attention_map = attention_map.reshape(num_patches_x, num_patches_x)

                attention_map_ax = axes[block_idx + 1, head_idx]
                attention_map_ax.imshow(
                    attention_map, cmap="viridis", interpolation="none"
                )
                attention_map_ax.axis("off")
                attention_map_ax.set_title(f"Block {block_idx} Head {head_idx}")

        plt.tight_layout()
        if save_path:
            file_name = f"image_{image_idx}.png"
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path)
        else:
            plt.show()
