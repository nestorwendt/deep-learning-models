import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import datetime
from typing import Tuple

from utils.system import calculate_num_workers
from datasets import imagenet
from models.vit import ViT

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 0,
    accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """
    Function to train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Dataloader for training data.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the computation (cpu or cuda).
        epoch (int, optional): Current epoch number. Defaults to 0.
        num_epochs (int, optional): Total number of epochs. Defaults to 0.

    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch.
    """
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    with tqdm(dataloader, unit="batch", file=sys.stdout) as tepoch:
        for inputs, targets in tepoch:
            tepoch.set_description(f"Training {epoch + 1}/{num_epochs}")

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()

            if (tepoch.n + 1) % accumulation_steps == 0 or tepoch.n == len(
                dataloader
            ) - 1:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=loss.item() * accumulation_steps,
                accuracy=(100.0 * correct) / total,
            )

    avg_loss = epoch_loss / len(dataloader)
    accuracy = (100.0 * correct) / total

    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    num_epochs: int = 0,
) -> Tuple[float, float]:
    """
    Function to validate the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Dataloader for validation data.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run the computation (cpu or cuda).
        epoch (int, optional): Current epoch number. Defaults to 0.
        num_epochs (int, optional): Total number of epochs. Defaults to 0.

    Returns:
        Tuple[float, float]: Average loss and accuracy for the validation epoch.
    """
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch", file=sys.stdout) as vepoch:
            for inputs, targets in vepoch:
                vepoch.set_description(f"Validation {epoch + 1}/{num_epochs}")

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                vepoch.set_postfix(
                    val_loss=loss.item(), val_accuracy=(100.0 * val_correct) / val_total
                )

    avg_val_loss = val_loss / len(dataloader)
    val_accuracy = (100.0 * val_correct) / val_total

    return avg_val_loss, val_accuracy


def main():
    model_name = "vit-256emb-04layer-08head-08patch-16register"
    num_classes = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = calculate_num_workers()

    # Hyperparameters
    img_size = 224
    patch_size = 8
    dropout = 0.1
    num_registers = 16
    batch_size = 16
    num_epochs = 64
    learning_rate = 1e-4
    accumulation_steps = 16

    # Initialize the model
    model = ViT(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_size=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=3,
        in_channels=3,
        dropout=dropout,
        num_registers=num_registers,
    ).to(device)

    # Prepare dataloaders
    train_dataloader, val_dataloader = imagenet.prepare_dataloaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root="data",
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs,
            accumulation_steps,
        )
        val_loss, val_accuracy = validate_one_epoch(
            model, val_dataloader, criterion, device, epoch, num_epochs
        )

        # Save the model every epoch
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        description = f"{model_name}_epoch-{epoch + 1}_valacc-{val_accuracy:.2f}"
        torch.save(model, f"checkpoints/{current_time}_{description}.pth")


if __name__ == "__main__":
    main()
