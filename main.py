import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from datetime import datetime

from models.vision_transformer import VisionTransformer

if __name__ == "__main__":

    # Hyperparameters
    img_size = 224
    patch_size = 16
    num_classes = 1000
    embedding_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    in_channels = 3
    dropout = 0.0
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() if os.cpu_count() is not None else 0

    # Initialize the Vision Transformer model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        in_channels=in_channels,
        dropout=dropout,
    ).to(device)

    # Define transformation pipeline for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )

    # Prepare the dataset
    train_dataset = datasets.ImageFolder(
        root="../data/ILSVRC2012/imagenet/train", transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = datasets.ImageFolder(
        root="../data/ILSVRC2012/imagenet/val", transform=transform)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
                tepoch.set_description(
                    f"Training epoch {epoch + 1}/{num_epochs}")

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

                tepoch.set_postfix(
                    loss=loss.item(), accuracy=100. * correct / total)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with tqdm(val_dataloader, unit="batch") as vepoch:
                for inputs, targets in vepoch:
                    vepoch.set_description(f"Validation epoch {
                                           epoch + 1}/{num_epochs}")

                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    vepoch.set_postfix(
                        val_loss=loss.item(), val_accuracy=100. * val_correct / val_total)

    # Save the model after training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("checkpoints", f"{timestamp}.pth")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model, model_path)
