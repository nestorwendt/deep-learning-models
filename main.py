import torch
from torch import nn, optim
from tqdm import tqdm

from utils.system import calculate_num_workers, save_model
from datasets import imagenette
from models.vit import make_vit_base


def main():

    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = calculate_num_workers()

    # Hyperparameters
    img_size = 224
    patch_size = 32
    dropout = 0.0
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50

    # Initialize the model
    model = make_vit_base(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    # Prepare dataloaders
    train_dataloader, val_dataloader = imagenette.prepare_dataloaders(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root="data",
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Training epoch {epoch + 1}/{num_epochs}")

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with tqdm(val_dataloader, unit="batch") as vepoch:
                for inputs, targets in vepoch:
                    vepoch.set_description(f"Validation epoch {epoch + 1}/{num_epochs}")

                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    vepoch.set_postfix(
                        val_loss=loss.item(),
                        val_accuracy=100.0 * val_correct / val_total,
                    )

    # Save the model after training
    save_model(model, "checkpoints")


if __name__ == "__main__":
    main()
