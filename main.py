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
    num_classes = 10
    embedding_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    in_channels = 3
    dropout = 0.0
    batch_size = 64
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

    # Download and prepare the dataset
    train_dataset = datasets.Imagenette(
        root="data", split="train", download=False, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

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

    # Save the model after training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("checkpoints", f"{timestamp}.pth")
    torch.save(model.state_dict(), model_path)