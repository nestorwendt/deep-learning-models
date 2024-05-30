"""
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from typing import Optional
from torchvision import transforms
from matplotlib.figure import Figure


def convert_grayscale_to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert a grayscale image to RGB if necessary.

    Args:
        img (Image.Image): Input image.

    Returns:
        Image.Image: RGB image.
    """
    assert isinstance(
        img, Image.Image
    ), f"Expected img to be a PIL Image, but got {type(img)}"

    # Convert the image to RGB if it is not already in RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def plot_attention_maps(
    image: Image.Image,
    transform: transforms.Compose,
    model: nn.Module,
    save_path: Optional[str] = None,
    interpolation: str = "none",
) -> Optional[Figure]:
    """
    Plots attention maps for a given image using a specified model and transformation.

    Args:
        image (Image.Image): The input image to be processed.
        transform (transforms.Compose): The transformations to apply to the input image before feeding it to the model.
        model (nn.Module): The neural network model that has attention layers.
        save_path (Optional[str]): The file path to save the resulting plot. If None, the plot is displayed instead of being saved.
        interpolation (str): The interpolation method used for displaying the image. Default is 'none'.

    Returns:
        Optional[Figure]: The Matplotlib figure object containing the attention map plot.
                          If save_path is specified, returns None after saving the figure.
    """

    return None
    
    # # Save or display the plot
    # if save_path:
    #     plt.savefig(save_path)
    #     plt.close(fig)

    #     return None
    # else:
    #     plt.show()

    #     return fig
