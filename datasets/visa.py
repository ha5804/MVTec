from datasets.anomaly import AnomalyDataset


class VisADataset(AnomalyDataset):
    def __init__(
        self,
        category,
        phase="train",
        batch_size=1,
        shuffle=True,
        limit=None,
        limit_per_class=None,
        image_size=256,
    ):
        super().__init__(
            dataset="visa",
            category=category,
            phase=phase,
            image_size=image_size,
            limit=limit,
            limit_per_class=limit_per_class,
            batch_size=batch_size,
            shuffle=shuffle,
        )
