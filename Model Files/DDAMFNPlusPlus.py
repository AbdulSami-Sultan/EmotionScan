import torch
import torch.nn as nn
import torch.nn.functional as F

class DDAMFNPlusPlus(nn.Module):
    def __init__(self, num_classes=5):  # `num_classes` = number of emotion classes
        super(DDAMFNPlusPlus, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Attention mechanism
        attention_map = self.attention(x)
        x = x * attention_map

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (7, 7))  # Fixed output size
        x = torch.flatten(x, 1)  # Flatten for fully connected layers

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Example usage
if __name__ == "__main__":
    model = DDAMFNPlusPlus(num_classes=5)
    print(model)

    # Test with random input (batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print("Output shape:", output.shape)
