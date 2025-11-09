import torch
from omegaconf import OmegaConf
from data import TrainDataset, ValidDataset, TestDataset, BalancedTrainDataset
from data.augmentations import AmplitudeScaling, BaselineWander, AdditiveGaussianNoise, RandomDropouts, MotionArtifacts, TimeScaling, Compose

def create_train_data_loader(cfg):
    train_datasets = TrainDataset(**cfg.path.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets, shuffle=True, **cfg.loader
    )

    valid_datasets = ValidDataset(**cfg.path.valid)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_datasets, shuffle=False, **cfg.loader
    )

    return train_dataloader, valid_dataloader

def create_test_data_loader(cfg):
    test_path_cfg = cfg.path.test

    dataset_kwargs = {
        "x_path": test_path_cfg.x_path,
        "y_path": test_path_cfg.y_path,
    }

    ecg_path = getattr(test_path_cfg, "ecg_path", None)
    if ecg_path is not None:
        dataset_kwargs["ecg_path"] = ecg_path
        dataset_kwargs["include_ecg"] = getattr(test_path_cfg, "include_ecg", True)

    test_datasets = TestDataset(**dataset_kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        test_datasets, shuffle=False, **cfg.loader
    )

    return test_dataloader

def _build_ectopic_augmentations(aug_cfg):
    """Build augmentation pipeline for selected target classes."""
    if not getattr(aug_cfg, "enable", False):
        return []
    
    transforms = []
    
    if aug_cfg.amplitude_scaling.enable:
        transforms.append(AmplitudeScaling(**aug_cfg.amplitude_scaling.params))
    
    if aug_cfg.baseline_wander.enable:
        transforms.append(BaselineWander(**aug_cfg.baseline_wander.params))
    
    if aug_cfg.additive_gaussian_noise.enable:
        transforms.append(AdditiveGaussianNoise(**aug_cfg.additive_gaussian_noise.params))
    
    if aug_cfg.random_dropouts.enable:
        transforms.append(RandomDropouts(**aug_cfg.random_dropouts.params))
    
    if aug_cfg.motion_artifacts.enable:
        transforms.append(MotionArtifacts(**aug_cfg.motion_artifacts.params))
    
    if aug_cfg.time_scaling.enable:
        transforms.append(TimeScaling(**aug_cfg.time_scaling.params))
    
    return transforms

def create_balanced_train_data_loader(cfg):
    """Create balanced training data loader with class-specific augmentation."""

    augmentation_transforms = []
    target_class_ratios = None
    reference_class = None
    classes_to_augment = None

    if hasattr(cfg, "augmentations"):
        augmentation_transforms = _build_ectopic_augmentations(cfg.augmentations)

        if hasattr(cfg.augmentations, "target_class_ratios"):
            target_class_ratios = OmegaConf.to_container(
                cfg.augmentations.target_class_ratios, resolve=True
            )

        if hasattr(cfg.augmentations, "reference_class"):
            reference_class = cfg.augmentations.reference_class

        if hasattr(cfg.augmentations, "classes_to_augment"):
            classes_to_augment = list(cfg.augmentations.classes_to_augment)

    # Create balanced dataset
    train_datasets = BalancedTrainDataset(
        **cfg.path.train,
        augmentation_transforms=augmentation_transforms,
        target_class_ratios=target_class_ratios,
        reference_class=reference_class,
        classes_to_augment=classes_to_augment,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets, shuffle=True, **cfg.loader
    )

    valid_datasets = ValidDataset(**cfg.path.valid)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_datasets, shuffle=False, **cfg.loader
    )

    return train_dataloader, valid_dataloader