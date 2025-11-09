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
    """Dataset that balances classes by augmenting minority class (ectopic)"""
    def __init__(self, x_path: str, y_path: str, augmentation_transforms=None, target_ratio=1.0):
        """
        Args:
            x_path: Path to features
            y_path: Path to labels
            augmentation_transforms: List of augmentation transforms
            target_ratio: Target ratio of ectopic to normal (1.0 = equal)
        """
        super().__init__()
        
        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        self.augmentation_transforms = augmentation_transforms or []
        
        # Calculate class distribution
        self.normal_indices = np.where(self.y == 0)[0]
        self.ectopic_indices = np.where(self.y == 1)[0]
        
        self.n_normal = len(self.normal_indices)
        self.n_ectopic = len(self.ectopic_indices)
        
        # Calculate how many augmented ectopic samples we need
        target_ectopic = int(self.n_normal * target_ratio)
        self.n_augmentations_needed = max(0, target_ectopic - self.n_ectopic)
        
        print(f"Original: {self.n_normal} normal, {self.n_ectopic} ectopic")
        print(f"Target ratio: {target_ratio}")
        print(f"Will generate {self.n_augmentations_needed} augmented ectopic samples")
    
    def __getitem__(self, index):
        if index < len(self.x):
            # Original sample
            x_get = self.x[index].astype(np.float32)
            y_get = self.y[index].astype(np.int64)
            
            # Apply augmentation only to ectopic samples
            if y_get == 1 and len(self.augmentation_transforms) > 0:
                for transform in self.augmentation_transforms:
                    x_get = transform(x_get)
                x_get = x_get.astype(np.float32)
            
            return x_get, y_get
        else:
            # Augmented ectopic sample
            aug_index = index - len(self.x)
            original_index = self.ectopic_indices[aug_index % len(self.ectopic_indices)]
            
            x_get = self.x[original_index].astype(np.float32)
            y_get = self.y[original_index].astype(np.int64)
            
            # Apply all augmentations
            for transform in self.augmentation_transforms:
                x_get = transform(x_get)
            x_get = x_get.astype(np.float32)
            
            return x_get, y_get
    
    def __len__(self):
        return len(self.x) + self.n_augmentations_needed
