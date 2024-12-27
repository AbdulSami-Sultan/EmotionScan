import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import os
from affectnet_dataset import AffectNetDataset
from ddamfn_model import DDAMFNPlusPlus


def train_model(root_dir, batch_size=32, epochs=10, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations (you can add more if needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = AffectNetDataset(root_dir, split="train", transform=transform)
    val_dataset = AffectNetDataset(root_dir, split="val", transform=transform)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = DDAMFNPlusPlus(num_classes=8).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate through the train dataset
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        # Log average loss for the epoch
        print(f"Epoch {epoch+1} completed. Average Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Call the training function
if __name__ == "__main__":
    root_dir = "/path/to/dataset"  # Change this to your dataset path
    train_model(root_dir)
