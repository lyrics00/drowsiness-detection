from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


def default_image_loader(path: Path, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


class FolderImageDataset(Dataset):
    """
    Simple dataset for Kaggle-style folder layouts:

      data/processed/<split>/alert/*.jpg
      data/processed/<split>/drowsy/*.jpg

    Labels: alert=0, drowsy=1
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loader: Callable[[Path, Tuple[int, int]], np.ndarray] = default_image_loader,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.loader = loader

        self.samples: list[ImageSample] = []
        for cls_name, label in [("alert", 0), ("drowsy", 1)]:
            d = self.root / split / cls_name
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append(ImageSample(path=p, label=label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}/{split}/(alert|drowsy). "
                "Run preprocessing or check your folder layout."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self.loader(s.path, self.image_size)  # HWC float32 RGB [0,1]
        if self.transform is not None:
            img = self.transform(img)
        # Torch expects CHW
        x = np.transpose(img, (2, 0, 1)).copy()
        y = np.int64(s.label)
        return x, y

