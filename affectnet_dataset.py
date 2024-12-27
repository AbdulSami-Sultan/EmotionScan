import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AffectNetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Log the dataset split being loaded
        print(f"Loading {split} data from {self.root_dir}")

        # Load image paths and corresponding labels
        for emotion_label in os.listdir(self.root_dir):
            emotion_folder = os.path.join(self.root_dir, emotion_label)
            if os.path.isdir(emotion_folder):
                for img_file in os.listdir(emotion_folder):
                    img_path = os.path.join(emotion_folder, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(int(emotion_label))  # Folder name is the label
        
        # Debug log: Number of images loaded
        print(f"Loaded {len(self.image_paths)} images with {len(self.labels)} labels from {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Debug log for accessing a particular index
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} is out of range. Dataset size is {len(self.image_paths)}")
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Log the image being loaded
        # print(f"Loading image {img_path}, label {label}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        
        if self.transform:
            image = self.transform(image)

        return image, label
