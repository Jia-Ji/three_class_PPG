import numpy as np
from torch.utils.data import Dataset
#from data.augmentations import SelectiveAugmentation, Compose


class TrainDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, transform=None):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        self.transform = transform
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        if self.transform is not None:
            x_get = self.transform(x_get)
            x_get = x_get.astype(np.float32)
        y_get = self.y[index].astype(np.int64)
        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class ValidDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        y_get = self.y[index].astype(np.int64)
        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class TestDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, ecg_path: str = None, include_ecg: bool = False):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        self.ecg = None
        self.include_ecg = False

        if ecg_path is not None:
            self.ecg = np.load(ecg_path, allow_pickle=True)
            if len(self.ecg) != len(self.x):
                raise ValueError(
                    "Length mismatch between PPG and ECG test splits."
                )

        if include_ecg:
            if self.ecg is None:
                raise ValueError("include_ecg=True but ecg_path was not provided or failed to load.")
            self.include_ecg = True
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        y_get = self.y[index].astype(np.int64)

        if self.include_ecg:
            ecg_get = self.ecg[index].astype(np.float32)
            return x_get, y_get, ecg_get

        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class BalancedTrainDataset(Dataset):
    """Dataset that balances classes by augmenting under-represented classes."""

    def __init__(
        self,
        x_path: str,
        y_path: str,
        augmentation_transforms=None,
        target_class_ratios=None,
        reference_class=None,
        classes_to_augment=None,
    ):
        """
        Args:
            x_path: Path to features.
            y_path: Path to labels.
            augmentation_transforms: List of augmentation transforms applied sequentially.
            target_class_ratios: Mapping {class_label: desired_ratio_vs_reference}.
            reference_class: The class used as the reference for desired ratios. Defaults to the
                majority class if not provided.
            classes_to_augment: Optional iterable of class labels eligible for augmentation. If
                None, all classes other than the reference class can be augmented.
        """
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        self.augmentation_transforms = augmentation_transforms or []

        self.original_length = len(self.x)
        if self.original_length != len(self.y):
            raise ValueError("Features and labels must have the same length.")

        # Prepare class information
        self.class_indices = {
            int(cls): np.where(self.y == cls)[0] for cls in np.unique(self.y)
        }

        if len(self.class_indices) == 0:
            raise ValueError("BalancedTrainDataset received an empty label array.")

        if reference_class is None:
            # Default to the majority class
            reference_class = max(self.class_indices, key=lambda k: len(self.class_indices[k]))
        reference_class = int(reference_class)

        if reference_class not in self.class_indices:
            raise ValueError(f"Reference class {reference_class} not present in labels.")

        self.reference_class = reference_class
        reference_count = len(self.class_indices[self.reference_class])

        if reference_count == 0:
            raise ValueError(f"Reference class {self.reference_class} has no samples.")

        # Normalise target class ratios
        if target_class_ratios is None:
            target_class_ratios = {cls: 1.0 for cls in self.class_indices}
        else:
            target_class_ratios = {
                int(cls): float(target_class_ratios[cls]) for cls in target_class_ratios
            }
            for cls in self.class_indices:
                target_class_ratios.setdefault(cls, 1.0)

        self.target_class_ratios = target_class_ratios

        # Determine which classes can receive augmentations
        if classes_to_augment is not None:
            classes_to_augment = {int(cls) for cls in classes_to_augment}
        else:
            classes_to_augment = set(self.class_indices.keys()) - {self.reference_class}

        self.classes_to_augment = classes_to_augment

        # Compute augmentation plan
        self.augmentations_needed = {}
        self.total_augmented_samples = 0
        for cls, indices in self.class_indices.items():
            desired_count = int(np.round(reference_count * self.target_class_ratios[cls]))
            required = max(0, desired_count - len(indices))
            if cls not in self.classes_to_augment:
                required = 0
            self.augmentations_needed[cls] = required
            self.total_augmented_samples += required

        # Prepare lookup for augmented samples
        self._augmented_class_order = []
        cumulative = 0
        self._augmented_class_boundaries = []
        for cls, extra in self.augmentations_needed.items():
            if extra <= 0:
                continue
            cumulative += extra
            self._augmented_class_order.append(cls)
            self._augmented_class_boundaries.append(cumulative)

        print("Class counts:", {cls: len(idx) for cls, idx in self.class_indices.items()})
        print("Target class ratios:", self.target_class_ratios)
        print("Augmentations needed per class:", self.augmentations_needed)

    def _apply_transforms(self, signal: np.ndarray) -> np.ndarray:
        transformed = signal
        for transform in self.augmentation_transforms:
            transformed = transform(transformed)
        return transformed.astype(np.float32)

    def __getitem__(self, index):
        if index < self.original_length:
            x_get = self.x[index].astype(np.float32)
            y_get = int(self.y[index])

            if y_get in self.classes_to_augment and len(self.augmentation_transforms) > 0:
                x_get = self._apply_transforms(x_get)

            return x_get, np.int64(y_get)

        # Augmented sample branch
        aug_index = index - self.original_length

        # Identify which class this augmented sample belongs to
        previous_boundary = 0
        selected_class = None
        within_class_index = None
        for cls, boundary in zip(self._augmented_class_order, self._augmented_class_boundaries):
            if aug_index < boundary:
                selected_class = cls
                within_class_index = aug_index - previous_boundary
                break
            previous_boundary = boundary

        if selected_class is None:
            raise IndexError("Augmented sample index out of range.")

        source_indices = self.class_indices[selected_class]
        source_index = source_indices[within_class_index % len(source_indices)]

        x_source = self.x[source_index].astype(np.float32)
        x_augmented = self._apply_transforms(x_source)

        return x_augmented, np.int64(selected_class)

    def __len__(self):
        return self.original_length + self.total_augmented_samples
