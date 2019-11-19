from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomHorizontalFlip, ToTensor, RandomCrop, Resize, CenterCrop
import torch
import os
from skimage import io, transform
from PIL import Image

class IMDBDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_paths, targets, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = img_paths
        self.root_dir = root_dir
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.samples[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.targets[idx][0], dtype=torch.long), torch.tensor(self.targets[idx][1], dtype=torch.long)
