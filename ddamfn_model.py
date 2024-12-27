import torch
import torch.nn as nn
import torchvision.models as models

class DDAMFNPlusPlus(nn.Module):
    def __init__(self, num_classes=8):
        super(DDAMFNPlusPlus, self).__init__()

        # Assuming you want to use ResNet18 as the backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use appropriate weights
        # Replace the final fully connected layer with Identity (you can keep other layers)
        self.backbone.fc = nn.Identity()
        
        # Define the new fully connected layers
        # Use a dummy input to get the output size of the backbone (ResNet18)
        with torch.no_grad():
            # Dummy input for extracting the output features after backbone
            dummy_input = torch.zeros(1, 3, 224, 224)  # Example input size for ResNet18
            features_out = self.backbone(dummy_input)
            self.fc1_input_size = features_out.shape[1]  # Number of features from the backbone
        
        # Now define the new fully connected layers using the extracted input size
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Forward pass
        x = self.backbone(x)  # Pass through the backbone
        print(x.shape)
        x = self.fc1(x)       # Pass through the first fully connected layer
        x = torch.relu(x)     # Apply ReLU activation
        x = self.fc2(x)       # Pass through the second fully connected layer
        return x
