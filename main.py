import hydra
import hydra.utils
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import os
import shutil
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger


from utils import create_train_data_loader, create_test_data_loader, create_balanced_train_data_loader
from models.model_adapt import EctopicsClassifier
from data.augmentations import set_augmentations_seed



@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    
    # Set seed FIRST before anything else
    seed = 1024
    pl.seed_everything(seed, workers=True)
    
    # Set augmentation RNG seed for reproducibility
    set_augmentations_seed(seed)
    
    # Additional deterministic settings for full reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading ...", flush=True)
    # train_loader, valid_loader = create_train_data_loader(cfg.data)
    train_loader, valid_loader = create_balanced_train_data_loader(cfg.data)
    test_loader = create_test_data_loader(cfg.data)
    print("Done!", flush=True)

    total_training_steps = len(train_loader) * cfg.trainer.parameters.max_epochs

    model = EctopicsClassifier(**cfg.model, total_training_steps = total_training_steps, training_config = cfg)

    checkpoint_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.trainer.callbacks.early_stop)
    callbacks = [checkpoint_callback, early_stop_callback]

    logger = TensorBoardLogger(**cfg.trainer.callbacks.logger)
    # Align CSV logger with TensorBoard logger directory and version
    csv_logger = CSVLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=logger.version,
    )

    trainer = pl.Trainer(**cfg.trainer.parameters, callbacks=callbacks, logger=[logger, csv_logger])

    ckpt_path = None
    if cfg.experiment.resume_ckpt:
        ckpt_path = cfg.experiment.ckpt_path

    if cfg.experiment.train:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path,
        )

    if cfg.experiment.test:
        # Use best checkpoint if available; otherwise test current model
        test_ckpt = "best" if cfg.trainer.callbacks.model_checkpoint.save_top_k and cfg.trainer.callbacks.model_checkpoint.monitor else None
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=test_ckpt)
        
    # Save main_log.txt to logs directory after testing
    original_cwd = hydra.utils.get_original_cwd()
    main_log_path = os.path.join(original_cwd, "main_log.txt")
    
    if os.path.exists(main_log_path):
        # Get the logger version directory
        version = logger.version if logger.version is not None else ""
        if version:
            log_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
        else:
            log_dir = os.path.join(logger.save_dir, logger.name)
        
        os.makedirs(log_dir, exist_ok=True)
        dest_path = os.path.join(log_dir, "main_log.txt")
        shutil.copy2(main_log_path, dest_path)
        print(f"Saved main_log.txt to {dest_path}", flush=True)
    else:
        print(f"Warning: main_log.txt not found in current or original directory. Skipping copy to logs directory.", flush=True)



if __name__ == '__main__':
    main()
    print("Training Done!")
