"""
"""

from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def get_transform(img_size: int) -> transforms.Compose:
    """
    Define the transformation pipeline for the dataset.

    Args:
        img_size (int): The target size of the image (height, width).

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def prepare_dataloaders(
    img_size: int,
    batch_size: int,
    num_workers: int,
    data_root: str = "data",
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the training and validation datasets and dataloaders.

    Args:
        img_size (int): The target size of the image (height, width).
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of worker threads for loading data.
        data_root (str, optional): The root directory of the dataset. Default is "data".
        download (bool, optional): Whether to download the dataset if not present. Default is False.

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    transform = get_transform(img_size)

    # Prepare the dataset
    train_dataset: Dataset = datasets.Imagenette(
        root=data_root, transform=transform, split="train", download=download
    )
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataset: Dataset = datasets.Imagenette(
        root=data_root, transform=transform, split="val", download=download
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader
