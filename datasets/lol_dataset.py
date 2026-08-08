"""LOL-v1 dataset loader (paired low/high, positional pairing by sorted name).

Used by train.py / test.py for the LOL-v1 benchmark (paper Sec. 4.1.1).
Train mode returns (low_tensor, high_tensor); test mode returns
(low_tensor, filename). Images are loaded at full resolution (no crops).

Usage (from repo root):
    from datasets.lol_dataset import LOLDataset
    ds = LOLDataset(low_dir="data/lolv1/train/low",
                    high_dir="data/lolv1/train/high", mode="train")
"""

from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms


class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir=None, mode='train'):
        """
        Args:
            low_dir (str): directory of low-light images
            high_dir (str, optional): directory of normal-light images
                (may be omitted in test mode)
            mode (str): 'train' or 'test'; decides whether the normal-light
                image is returned
        """
        self.mode = mode
        self.low_paths = sorted([
            os.path.join(low_dir, f) for f in os.listdir(low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if self.mode == 'train':
            assert high_dir is not None, "train mode requires high_dir"
            self.high_paths = sorted([
                os.path.join(high_dir, f) for f in os.listdir(high_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            assert len(self.low_paths) == len(self.high_paths), \
                "low/high image counts differ"

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert('RGB')
        low_tensor = self.transform(low_img)

        if self.mode == 'train':
            high_img = Image.open(self.high_paths[idx]).convert('RGB')
            high_tensor = self.transform(high_img)
            return low_tensor, high_tensor
        else:
            filename = os.path.basename(self.low_paths[idx])
            return low_tensor, filename
