from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class MyData:
    def __init__(
        self,
        cls,
        phase="train",
        batch_size=1,
        shuffle=True,
        limit=None,
        limit_per_class=None,
    ):
        self.cls = cls
        self.phase = phase
        
        self.dir = f"data/MVTec/{self.cls}/{self.phase}"

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image_dataset = datasets.ImageFolder(
            root=self.dir,
            transform=self.transform
        )
        self.classes = image_dataset.classes
        self.dataset = image_dataset

        if limit_per_class is not None:
            indices = []
            counts = {class_idx: 0 for class_idx in range(len(image_dataset.classes))}
            for idx, (_, class_idx) in enumerate(image_dataset.samples):
                if counts[class_idx] < limit_per_class:
                    indices.append(idx)
                    counts[class_idx] += 1
            self.dataset = Subset(image_dataset, indices)
        elif limit is not None:
            limit = min(limit, len(image_dataset))
            self.dataset = Subset(image_dataset, range(limit))

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataset(self):
        return self.dataset

    def get_loader(self):
        return self.loader

    def get_classes(self):
        return self.classes
        
