from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class AnomalyDataset(Dataset):
    """Shared dataset wrapper for MVTec AD and VisA image folders."""

    def __init__(
        self,
        dataset,
        category,
        phase="train",
        root="data",
        image_size=256,
        limit=None,
        limit_per_class=None,
        batch_size=1,
        shuffle=False,
    ):
        self.dataset_name = dataset.lower()
        self.category = category
        self.phase = phase
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.samples, self.classes = self._build_samples()
        self.indices = self._build_indices(limit=limit, limit_per_class=limit_per_class)
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def _build_samples(self):
        if self.dataset_name == "mvtec":
            return self._build_mvtec_samples()
        if self.dataset_name == "visa":
            return self._build_visa_samples()
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _build_mvtec_samples(self):
        phase_dir = self.root / "MVTec" / self.category / self.phase
        if not phase_dir.exists():
            raise FileNotFoundError(f"MVTec path not found: {phase_dir}")

        class_dirs = sorted([path for path in phase_dir.iterdir() if path.is_dir()])
        classes = [path.name for path in class_dirs]
        samples = []
        for class_idx, class_dir in enumerate(class_dirs):
            for image_path in _iter_images(class_dir):
                samples.append((image_path, class_idx))
        return samples, classes

    def _build_visa_samples(self):
        image_dir = self.root / "Visa" / self.category / "Data" / "Images"
        if not image_dir.exists():
            raise FileNotFoundError(f"VisA path not found: {image_dir}")

        if self.phase == "train":
            class_dirs = [image_dir / "Normal"]
        elif self.phase == "test":
            class_dirs = [image_dir / "Normal", image_dir / "Anomaly"]
        else:
            raise ValueError("VisA phase must be 'train' or 'test'")

        classes = [path.name for path in class_dirs if path.exists()]
        samples = []
        for class_idx, class_dir in enumerate(class_dirs):
            if not class_dir.exists():
                continue
            for image_path in _iter_images(class_dir):
                samples.append((image_path, class_idx))
        return samples, classes

    def _build_indices(self, limit=None, limit_per_class=None):
        if limit_per_class is not None:
            indices = []
            counts = {class_idx: 0 for class_idx in range(len(self.classes))}
            for idx, (_, class_idx) in enumerate(self.samples):
                if counts[class_idx] < limit_per_class:
                    indices.append(idx)
                    counts[class_idx] += 1
            return indices

        if limit is not None:
            return list(range(min(limit, len(self.samples))))

        return list(range(len(self.samples)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label

    def get_dataset(self):
        return self

    def get_loader(self):
        return self.loader

    def get_classes(self):
        return self.classes

    def get_label_name(self, label):
        return self.classes[int(label)]

    def is_anomaly_label(self, label):
        label_name = self.get_label_name(label)
        return label_name not in {"good", "Normal"}

    def get_path(self, idx):
        idx = self.indices[idx]
        return self.samples[idx][0]


def _iter_images(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
