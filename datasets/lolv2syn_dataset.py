"""LOL-v2-Synthetic dataset loader (pairing by lowercase filename stem).

Used by train.py / test.py for the LOL-v2-Synthetic benchmark (paper
Sec. 4.1.1). Low/high files share the same stem (extensions may differ);
pairing uses the lowercase stem as the key. Train mode returns
(low_tensor, high_tensor); test mode returns (low_tensor, filename).
No resizing; images are expected to be 384x384 (configurable via
expected_size / check_size).

Usage (from repo root):
    from datasets.lolv2syn_dataset import LOLv2SynDataset
    ds = LOLv2SynDataset(low_dir="data/lolv2syn/train/low",
                         high_dir="data/lolv2syn/train/high", mode="train")
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _is_img(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))


def _list_imgs(dir_path: str):
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if _is_img(f)
    )


def _pair_key(path: str) -> str:
    """
    Synthetic split: low/high share the same filename (extension may differ).
    Use the lowercase stem as the key to avoid case mismatches.
    e.g. r068812d7t.png -> r068812d7t
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem.lower()


class LOLv2SynDataset(Dataset):
    """
    LOLv2-Syn Dataset
      - mode='train': return (low_tensor, high_tensor)
      - mode='test' : return (low_tensor, filename)

    No resizing by default; images are expected to be 384x384
    (adjust or disable via expected_size / check_size).
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str | None = None,
        mode: str = "train",
        expected_size: tuple[int, int] = (384, 384),  # (W, H)
        check_size: bool = True,
        strict_pair: bool = True,
    ):
        assert mode in ("train", "test"), "mode must be 'train' or 'test'"
        self.mode = mode
        self.expected_size = expected_size
        self.check_size = check_size
        self.strict_pair = strict_pair

        self.low_paths = _list_imgs(low_dir)
        if len(self.low_paths) == 0:
            raise RuntimeError(f"[LOLv2SynDataset] low_dir is empty or has no images: {low_dir}")

        self.transform = transforms.ToTensor()

        if self.mode == "train":
            assert high_dir is not None, "train mode requires high_dir"
            high_paths = _list_imgs(high_dir)
            if len(high_paths) == 0:
                raise RuntimeError(f"[LOLv2SynDataset] high_dir is empty or has no images: {high_dir}")

            # Build high_map: key = stem(lowercase) -> path
            high_map = {}
            for hp in high_paths:
                k = _pair_key(hp)
                if k in high_map:
                    raise ValueError(
                        f"[LOLv2SynDataset] duplicate key in high_dir, key={k}\n"
                        f"  {high_map[k]}\n  {hp}"
                    )
                high_map[k] = hp

            # Pair low with high
            paired_low, paired_high = [], []
            missing = []
            for lp in self.low_paths:
                k = _pair_key(lp)
                hp = high_map.get(k)
                if hp is None:
                    missing.append(os.path.basename(lp))
                    continue
                paired_low.append(lp)
                paired_high.append(hp)

            if len(paired_low) == 0:
                raise RuntimeError(
                    f"[LOLv2SynDataset] no matching low/high pairs found:\n  low_dir={low_dir}\n  high_dir={high_dir}"
                )

            if len(missing) > 0:
                msg = f"[LOLv2SynDataset] {len(missing)} low images have no matching high (first 5): {missing[:5]}"
                if self.strict_pair:
                    raise RuntimeError(msg)
                else:
                    print("[WARN]", msg)

            self.low_paths = paired_low
            self.high_paths = paired_high

    def __len__(self):
        return len(self.low_paths)

    def _check_384(self, img: Image.Image, path: str, tag: str):
        if not self.check_size:
            return
        if img.size != self.expected_size:
            raise ValueError(
                f"[LOLv2SynDataset] {tag} size != {self.expected_size[0]}x{self.expected_size[1]}: "
                f"{path} size={img.size}"
            )

    def __getitem__(self, idx: int):
        low_path = self.low_paths[idx]
        low_img = Image.open(low_path).convert("RGB")
        self._check_384(low_img, low_path, "low")
        low_tensor = self.transform(low_img)

        if self.mode == "train":
            high_path = self.high_paths[idx]
            high_img = Image.open(high_path).convert("RGB")
            self._check_384(high_img, high_path, "high")
            high_tensor = self.transform(high_img)
            return low_tensor, high_tensor
        else:
            filename = os.path.basename(low_path)
            return low_tensor, filename
