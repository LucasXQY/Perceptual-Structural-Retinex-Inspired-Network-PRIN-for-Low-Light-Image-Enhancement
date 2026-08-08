"""LOL-v2-Real dataset loader (pairing by trailing numeric id).

Used by train.py / test.py for the LOL-v2-Real benchmark (paper Sec. 4.1.1).
High/low files are named like normal00012 / low00012, so pairing uses the
trailing digit run as the key. Train mode returns (low_tensor, high_tensor);
test mode returns (low_tensor, filename). No resizing (images are typically
600x400).

Usage (from repo root):
    from datasets.lolv2real_dataset import LOLv2RealDataset
    ds = LOLv2RealDataset(low_dir="data/lolv2real/train/low",
                          high_dir="data/lolv2real/train/high", mode="train")
"""

from torch.utils.data import Dataset
from PIL import Image
import os
import re
from torchvision import transforms


def _is_img(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def _list_imgs(dir_path: str):
    return sorted([
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if _is_img(f)
    ])


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _pair_key(path: str) -> str:
    """
    LOLv2-Real pairing rule:
      high: normal00xxx
      low : low00xxx
    Use the trailing digit run of the stem as the key (leading zeros kept):
      normal00012 -> "00012"
      low00012    -> "00012"
    """
    s = _stem(path).lower()
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else s


class LOLv2RealDataset(Dataset):
    """
    LOLv2-Real:
      - train: (low_tensor, high_tensor)
      - test : (low_tensor, filename)
    No resizing by default (real images are typically 600x400).
    """
    def __init__(self, low_dir, high_dir=None, mode="train"):
        self.mode = mode
        self.low_paths = _list_imgs(low_dir)
        self.transform = transforms.ToTensor()

        if self.mode == "train":
            assert high_dir is not None, "train mode requires high_dir"
            high_paths = _list_imgs(high_dir)

            # Match low/high using the trailing numeric key.
            high_map = {}
            for p in high_paths:
                k = _pair_key(p)
                # A duplicate key would silently overwrite; raise instead.
                if k in high_map:
                    raise ValueError(f"Duplicate numeric key in high_dir, key={k}:\n  {high_map[k]}\n  {p}")
                high_map[k] = p

            paired_low, paired_high = [], []
            missing = []
            for lp in self.low_paths:
                k = _pair_key(lp)
                hp = high_map.get(k, None)
                if hp is None:
                    missing.append(os.path.basename(lp))
                    continue
                paired_low.append(lp)
                paired_high.append(hp)

            if len(paired_low) == 0:
                raise RuntimeError(f"No matching low/high pairs found: {low_dir} <-> {high_dir}")

            # Change this to a raise if you require every low to be paired.
            if len(missing) > 0:
                print(f"[WARN] {len(missing)} low images have no matching high (first 5): {missing[:5]}")

            self.low_paths = paired_low
            self.high_paths = paired_high

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_paths[idx]).convert("RGB")
        low_tensor = self.transform(low_img)

        if self.mode == "train":
            high_img = Image.open(self.high_paths[idx]).convert("RGB")
            high_tensor = self.transform(high_img)
            return low_tensor, high_tensor
        else:
            filename = os.path.basename(self.low_paths[idx])
            return low_tensor, filename
