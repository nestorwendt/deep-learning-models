"""
This module provides functions to process images and visualize attention maps from a model.
Functions included:
    - convert_to_rgb: Converts a grayscale image to RGB.
    - normalize_image: Normalizes an image tensor to the range [0, 1].
    - plot_attention_maps: Plots the attention maps for a sample batch of images using a given model.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


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

    return img.convert("RGB") if img.mode != "RGB" else img


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor to the range [0, 1].

    Args:
        image (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor.
    """
    image = image.permute(1, 2, 0)
    return (image - image.min()) / (image.max() - image.min())


def plot_attention_maps(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> None:
    """
    Plot the attention maps for a sample batch of images.

    Args:
        model (nn.Module): The model to use for generating attention maps.
        dataloader (torch.utils.data.DataLoader): The dataloader to use for getting a sample batch of images.
        device (torch.device): The device to use for running the model.
    """
    sample_batch, _ = next(iter(dataloader))
    sample_batch = sample_batch.to(device)
    model(sample_batch)

    num_blocks = len(model.transformer_blocks)
    num_heads = model.transformer_blocks[0].attention.attention_map.size(1)

    for image_idx, image in enumerate(sample_batch):
        image = image.detach().cpu()
        processed_image = normalize_image(image)

        fig, axes = plt.subplots(
            num_blocks + 1, num_heads, figsize=(num_heads * 3, (num_blocks + 1) * 3)
        )

        if num_heads == 1:
            axes.imshow(processed_image)
            axes.axis("off")
            axes.set_title("Original Image")
        else:
            axes[0, 0].imshow(processed_image)
            axes[0, 0].axis("off")
            axes[0, 0].set_title("Original Image")
            for ax in axes[0, 1:]:
                ax.axis("off")

        for block_idx, block in enumerate(model.transformer_blocks):
            for head_idx in range(num_heads):
                attention_map = (
                    block.attention.attention_map[image_idx, head_idx, 0, 1:]
                    .detach()
                    .cpu()
                    .numpy()
                )
                num_patches_x = int(np.sqrt(attention_map.size))
                attention_map = attention_map.reshape(num_patches_x, num_patches_x)

                attention_map_ax = axes[block_idx + 1, head_idx]
                attention_map_ax.imshow(
                    attention_map, cmap="viridis", interpolation="none"
                )
                attention_map_ax.axis("off")
                attention_map_ax.set_title(f"Block {block_idx} Head {head_idx}")

        plt.tight_layout()
        plt.show()
