"""
"""

import os
from datetime import datetime
import torch


def calculate_num_workers() -> int:
    """
    Calculate the number of workers based on the CPU count.

    Returns:
        int: Number of workers.
    """
    num_workers = os.cpu_count()
    num_workers = num_workers if num_workers is not None else 0

    return num_workers


def save_model(model, directory="checkpoints") -> str:
    """
    Save the model after training.

    Args:
        model (torch.nn.Module): The model to be saved.
        directory (str): The directory where the model will be saved.

    Returns:
        str: The path to the saved model file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(directory, f"{timestamp}.pth")
    os.makedirs(directory, exist_ok=True)
    torch.save(model, model_path)

    return model_path
