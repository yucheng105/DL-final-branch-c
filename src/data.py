from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    dataset: str
    split: str
    generator: str


class PairedTransform:
    """Create semantic and forensic views from the same PIL image."""

    def __init__(
        self,
        semantic_size: int = 224,
        forensic_size: int | None = None,
        train: bool = False,
        augment: bool = False,
    ) -> None:
        self.semantic_size = semantic_size
        self.forensic_size = forensic_size
        self.train = train
        self.augment = augment
        self.semantic_norm = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.to_tensor = transforms.ToTensor()
        if train:
            self.semantic_resize = transforms.Compose(
                [
                    transforms.Resize(semantic_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomCrop(semantic_size, padding=4, padding_mode="reflect"),
                ]
            )
        else:
            self.semantic_resize = transforms.Compose(
                [
                    transforms.Resize(semantic_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(semantic_size),
                ]
            )

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        image = image.convert("RGB")
        if self.train and self.augment:
            image = self._augment(image)

        semantic = self.semantic_norm(self.to_tensor(self.semantic_resize(image)))
        forensic_image = image
        if self.forensic_size is not None:
            forensic_image = transforms.Resize(
                (self.forensic_size, self.forensic_size),
                interpolation=InterpolationMode.BICUBIC,
            )(forensic_image)
        forensic = self.to_tensor(forensic_image)
        return {"image_semantic": semantic, "image_forensic": forensic}

    def _augment(self, image: Image.Image) -> Image.Image:
        if torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if torch.rand(()) < 0.25:
            radius = float(torch.empty(()).uniform_(0.2, 0.8))
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image


class ForensicImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: Callable | None = None) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.path) as image:
            views = self.transform(image) if self.transform else {"image_forensic": transforms.ToTensor()(image.convert("RGB"))}
        return {
            **views,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "path": str(record.path),
            "dataset": record.dataset,
            "split": record.split,
            "generator": record.generator,
        }


def build_cifake_records(root: Path, split: str) -> list[ImageRecord]:
    split_root = root / "cifake" / split
    label_dirs = {"REAL": 0, "FAKE": 1}
    records: list[ImageRecord] = []
    for class_name, label in label_dirs.items():
        class_root = split_root / class_name
        records.extend(
            ImageRecord(path=path, label=label, dataset="cifake", split=split, generator="stable_diffusion_1_4")
            for path in sorted(class_root.rglob("*"))
            if path.suffix.lower() in IMG_EXTENSIONS
        )
    return records


def build_tiny_genimage_records(
    root: Path,
    split: str,
    generators: list[str] | None = None,
) -> list[ImageRecord]:
    gen_root = root / "tiny-genimage"
    generator_dirs = [p for p in sorted(gen_root.iterdir()) if p.is_dir()]
    if generators:
        wanted = set(generators)
        generator_dirs = [p for p in generator_dirs if p.name in wanted or _short_generator_name(p.name) in wanted]

    records: list[ImageRecord] = []
    for generator_dir in generator_dirs:
        split_root = generator_dir / split
        for class_name, label in {"nature": 0, "ai": 1}.items():
            class_root = split_root / class_name
            if not class_root.exists():
                continue
            records.extend(
                ImageRecord(
                    path=path,
                    label=label,
                    dataset="tiny-genimage",
                    split=split,
                    generator=_short_generator_name(generator_dir.name),
                )
                for path in sorted(class_root.rglob("*"))
                if path.suffix.lower() in IMG_EXTENSIONS
            )
    return records


def build_dataset(
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    transform: Callable | None,
    generators: list[str] | None = None,
) -> ForensicImageDataset:
    root = Path(dataset_root)
    if dataset_name == "cifake":
        records = build_cifake_records(root, split)
    elif dataset_name == "tiny-genimage":
        records = build_tiny_genimage_records(root, split, generators)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not records:
        raise ValueError(f"No images found for dataset={dataset_name}, split={split}, generators={generators}")
    return ForensicImageDataset(records, transform=transform)


def split_train_val(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _short_generator_name(name: str) -> str:
    return (
        name.replace("imagenet_ai_0419_", "")
        .replace("imagenet_ai_0424_", "")
        .replace("imagenet_ai_0508_", "")
        .replace("imagenet_", "")
    )
