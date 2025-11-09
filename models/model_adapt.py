import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Any, List, Tuple
from transformers import get_scheduler
import numpy as np
from omegaconf import DictConfig, OmegaConf
import io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from typing import List, Tuple, Union

from .resnet import resnet18_1D, resnet34_1D
from .loss_function import get_loss_function
import yaml

class CompeleteModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.__initialize_modules(config)
    

    def __initialize_modules(self, config: DictConfig):
        self.feat_extracter = resnet18_1D(**config.hyperparameters.feat_extracter)
        # self.feat_extracter = resnet34_1D(**config.hyperparameters.feat_extracter)
        self.classifier = nn.Linear(config.hyperparameters.feat_extracter.feat_dim,
                                config.hyperparameters.classifier.num_classes)
    
    def forward(self, x:Tensor):
        feat = self.feat_extracter(x)
        logits = self.classifier(feat)
        
        return logits
    

class EctopicsClassifier(pl.LightningModule):
    def __init__(self, 
                 task: str="binary", 
                 num_classes: int=2, 
                 lr: float=0.0001, 
                 weight_decay: float=0.0001,
                 loss_name: str="cross_entropy",
                 use_lr_scheduler: bool=True,
                 lr_warmup_ratio: float = 0.1,
                 device: str="cuda",
                 total_training_steps: int=1000,
                 config: DictConfig=None,
                 training_config=None,
                 **kwargs):
        super().__init__()

        # save training config to logs
        if training_config is not None:
            if hasattr(training_config, 'keys'):  # DictConfig or dict
                config_dict = OmegaConf.to_container(training_config, resolve=True)
            else:
                # If it’s still a string path (for backward compatibility)
                with open(training_config, "r") as f:
                    config_dict = yaml.safe_load(f)
            self.save_hyperparameters(config_dict)
        else:
            self.save_hyperparameters()

        self.task = task
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_ratio = lr_warmup_ratio
    

        if device == "cuda" and torch.cuda.is_available():
            self.device_type = torch.device("cuda")
        else:
            self.device_type = torch.device("cpu")
        
        self.total_steps = total_training_steps
        self.config = config

        self.metrics_lst = []
        for metric in self.config.metrics:
            if self.config.metrics[metric]:
                self.metrics_lst.append(metric)

        print("Loss Function: ", self.loss_name, flush=True)
        print("Metrics: ", self.config.metrics, flush=True)

        self.model = CompeleteModel(self.config)
        
        if self.loss_name == 'bce':
            self.loss_fn = get_loss_function(self.loss_name)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")
        
        self.metrics = nn.ModuleDict({
            "metrics_train": nn.ModuleDict({}),
            "metrics_valid": nn.ModuleDict({}),
            "metrics_test": nn.ModuleDict({})
        })

        for phase in ["train", "valid", "test"]:
            for metric in self.config.metrics:
                if metric == "accuracy":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.Accuracy(
                        self.task, num_classes=self.num_classes, average="none"
                    )
                elif metric == "cf_matrix":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.ConfusionMatrix(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "f1":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.F1Score(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "specificity":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.Specificity(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "AUC":
                    self.metrics["metrics_" + phase][metric] = torchmetrics.AUROC(
                        self.task, num_classes=self.num_classes
                    )
                elif metric == "sensitivity":
                    if self.task == "binary":
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Recall(task="binary")
                    else:
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Recall(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        )
                elif metric == "ppv":
                    if self.task == "binary":
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Precision(task="binary")
                    else:
                        self.metrics["metrics_" + phase][metric] = torchmetrics.Precision(
                            task="multiclass", num_classes=self.num_classes, average="macro"
                        )

        
        self.step_losses = {"train": [], "valid": [], "test": []}
        
        # Configure misclassified samples collection from training config
        if self.config is not None and hasattr(self.config, 'keys'):
            # Access misclassified_samples config using OmegaConf
            try:
                misclassified_cfg = self.config.misclassified_samples
                self.collect_misclassified_samples = OmegaConf.select(misclassified_cfg, 'enable', default=False)
                self.max_misclassified_to_store = OmegaConf.select(misclassified_cfg, 'max_samples', default=100)
            except (AttributeError, KeyError):
                # Default values if config section doesn't exist
                self.collect_misclassified_samples = False
                self.max_misclassified_to_store = 100
        else:
            # Default values if config not available
            self.collect_misclassified_samples = False
            self.max_misclassified_to_store = 100
        
        # Store misclassified samples for visualization
        self.misclassified_samples = {"test": []}
        
        if self.collect_misclassified_samples:
            print(f"Misclassified samples collection: ENABLED (max {self.max_misclassified_to_store} samples)", flush=True)
        else:
            print("Misclassified samples collection: DISABLED", flush=True)

    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.use_lr_scheduler:
            scheduler = {
                "scheduler": get_scheduler(
                    "polynomial",
                    optimizer,
                    num_warmup_steps=round(self.warmup_ratio * self.total_steps),
                    num_training_steps=self.total_steps,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

        return optimizer

    def forward(self, x: Tensor):
        return self.model(x)

    def plot_confusion_matrix(self, matrix):
        # Row-wise normalization (each row sums to 1) so colors reflect per-class proportions
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.to(dtype=torch.float32)
            row_sums = matrix.sum(dim=1, keepdim=True)
            # Avoid division by zero: only divide rows with positive sum
            matrix = torch.where(row_sums > 0, matrix / row_sums, matrix)
            data = matrix.detach().cpu().numpy()
        else:
            data = np.asarray(matrix, dtype=np.float32)
            row_sums = data.sum(axis=1, keepdims=True)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums>0)

        fig, ax = plt.subplots()
        cax = ax.matshow(data, vmin=0.0, vmax=1.0, cmap='Blues')
        fig.colorbar(cax)
    
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close(fig)  # ensure to close the figure to free memory
        buf.seek(0)
    
        image = Image.open(buf)
        image_tensor = ToTensor()(image)
    
        return image_tensor

    def visualize_misclassified_samples(self, phase: str):
        """Visualize misclassified samples and save to file"""
        if not self.misclassified_samples[phase]:
            return
        
        samples = self.misclassified_samples[phase]
        n_samples = len(samples)
        
        # Determine whether ECG traces are available
        has_ecg = any('ecg' in sample for sample in samples)

        # Create a grid of subplots (max 20 samples per visualization)
        n_plots = min(n_samples, 20)

        if has_ecg:
            fig, axes = plt.subplots(n_plots, 2, figsize=(18, 3.5 * n_plots))
            if n_plots == 1:
                axes = axes.reshape(1, 2)
        else:
            n_cols = 4
            n_rows = (n_plots + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
        
        label_names = ['Normal', 'PVC']
        
        def _to_1d(signal_array):
            if isinstance(signal_array, torch.Tensor):
                signal_array = signal_array.detach().cpu().numpy()
            if signal_array.ndim == 1:
                return signal_array
            if signal_array.ndim == 2:
                if signal_array.shape[0] == 1:
                    return signal_array[0]
                if signal_array.shape[1] == 1:
                    return signal_array[:, 0]
                return signal_array[0]
            return signal_array.reshape(-1)

        for i in range(n_plots):
            sample = samples[i]
            signal = _to_1d(sample['input'])

            if has_ecg:
                ppg_ax = axes[i, 0]
                ppg_ax.plot(signal, linewidth=1.5)
                ppg_ax.set_title(
                    f'True: {label_names[sample["true_label"]]}, '
                    f'Pred: {label_names[sample["prediction"]]}'
                    f'\nProb: [{sample["probabilities"][0]:.3f}, {sample["probabilities"][1]:.3f}]',
                    fontsize=9
                )
                ppg_ax.grid(True, alpha=0.3)
                ppg_ax.set_xlabel('Time')
                ppg_ax.set_ylabel('PPG Amplitude')

                ecg_data = sample.get('ecg')
                if ecg_data is not None:
                    ecg_signal = _to_1d(ecg_data)
                    ecg_ax = axes[i, 1]
                    ecg_ax.plot(ecg_signal, color='tab:red', linewidth=1.2)
                    ecg_ax.set_title('Corresponding ECG', fontsize=9)
                    ecg_ax.grid(True, alpha=0.3)
                    ecg_ax.set_xlabel('Time')
                    ecg_ax.set_ylabel('ECG Amplitude')
                else:
                    axes[i, 1].axis('off')
            else:
                ax = axes[i]
                ax.plot(signal, linewidth=1.5)
                ax.set_title(
                    f'True: {label_names[sample["true_label"]]}, '
                    f'Pred: {label_names[sample["prediction"]]}'
                    f'\nProb: [{sample["probabilities"][0]:.3f}, {sample["probabilities"][1]:.3f}]',
                    fontsize=9
                )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')

        if not has_ecg:
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to file and TensorBoard
        import os
        if hasattr(self, 'logger') and self.logger is not None:
            version = self.logger.version if self.logger.version is not None else ""
            if version:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name, f"version_{version}")
            else:
                save_dir = os.path.join(self.logger.save_dir, self.logger.name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{phase}_misclassified_samples.png")
            
            # Save to file
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved {n_samples} misclassified {phase} samples to {save_path}", flush=True)
            
            # Also log to TensorBoard if available
            if hasattr(self.logger, 'experiment') and self.logger.experiment is not None:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                image = Image.open(buf)
                image_tensor = ToTensor()(image)
                self.logger.experiment.add_image(f"{phase}_misclassified_samples", image_tensor, global_step=self.current_epoch)
        else:
            # Fallback: save to current directory
            save_path = f"{phase}_misclassified_samples.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved {n_samples} misclassified {phase} samples to {save_path}", flush=True)
        
        plt.close(fig)

    def update_metrics(self, outputs, targets, phase: str = "train"):
        # model_device =  next(self.model.parameters()).device
        for k in self.config.metrics:
            # metric_device = self.metrics["metrics_" + phase][k].device
            self.metrics["metrics_" + phase][k].update(outputs, targets)

    def reset_metrics(self, phase: str = "train"):
        for k in self.config.metrics:
            self.metrics["metrics_" + phase][k].reset()
    
    def log_all(self, items: List[Tuple[str, Union[float, torch.Tensor]]], phase: str = "train", prog_bar: bool = True, sync_dist_group: bool = False):
        for key, value in items:
            if value is not None:
                # Check if value is a float
                if isinstance(value, float):
                    self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                # Check if value is a tensor
                elif isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:  # Scalar tensor
                        self.log(f"{phase}_{key}", value, prog_bar=prog_bar, sync_dist_group=sync_dist_group)
                    elif len(value.shape) == 2:  # 2D tensor, assume confusion matrix and log as image
                        image_tensor = self.plot_confusion_matrix(value)
                        self.logger.experiment.add_image(f"{phase}_{key}", image_tensor, global_step=self.current_epoch)

    def training_step(self, batch, batch_idx):
        
        # print("training steps beigins...")
        x, targets = batch
        output_logits = self(x)
        # print(type(output_logits[0]))
        # print(targets.dtype)
        # print(targets.shape)
        preds = torch.argmax(output_logits, dim=1)


        if self.loss_name == "bce":
            loss = self.loss_fn(targets, output_logits, self.device)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")
        
        self.update_metrics(preds, targets, "train")
          
        self.step_losses["train"].append(loss.item())
        return {"loss": loss}

    def on_train_epoch_end(self):
        """End of the training epoch"""
        avg_loss = sum(self.step_losses["train"]) / len(self.step_losses["train"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "train"]["accuracy"].compute()

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "train"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "train"]["f1"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "train"]["specificity"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "train"]["AUC"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "train"]["sensitivity"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "train"]["ppv"].compute()
        
        self.log_all(
                items=[
                    ("loss", avg_loss),
                    ("accuracy", acc),
                    ("specificity", spec),
                    ("AUC", auc),
                    ("sensitivity", sensitivity),
                    ("ppv", ppv),
                    ("confusion_matrix", matrix),
                    ("f1", f1),
                ],
                phase="train",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        self.reset_metrics("train")
        self.step_losses["train"].clear()

    def validation_step(self, batch):
        
        x, targets = batch
        output_logits = self(x)
        preds = torch.argmax(output_logits, dim=1)
        loss = F.cross_entropy(output_logits, targets)

        self.update_metrics(preds, targets, "valid")
        self.step_losses["valid"].append(loss)

        # # Collect misclassified samples if enabled
        # if self.collect_misclassified_samples:
        #     misclassified_mask = (preds != targets)
        #     if misclassified_mask.any() and len(self.misclassified_samples["valid"]) < self.max_misclassified_to_store:
        #         misclassified_indices = torch.where(misclassified_mask)[0]
        #         for idx in misclassified_indices:
        #             if len(self.misclassified_samples["valid"]) >= self.max_misclassified_to_store:
        #                 break
        #             self.misclassified_samples["valid"].append({
        #                 'input': x[idx].detach().cpu().numpy(),
        #                 'prediction': preds[idx].item(),
        #                 'true_label': targets[idx].item(),
        #                 'probabilities': F.softmax(output_logits[idx], dim=0).detach().cpu().numpy()
        #             })

        return {"val_loss": loss}
    
    # def on_validation_epoch_start(self):
    #     """Clear misclassified samples at the start of validation epoch"""
    #     if self.collect_misclassified_samples:
    #         self.misclassified_samples["valid"].clear()
    
    def on_validation_epoch_end(self):
        """End of the training epoch"""
        avg_loss = sum(self.step_losses["valid"]) / len(self.step_losses["valid"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "valid"]["accuracy"].compute()

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "valid"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "valid"]["f1"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "valid"]["specificity"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "valid"]["AUC"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "valid"]["sensitivity"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "valid"]["ppv"].compute()
        
        self.log_all(
                items=[
                    ("loss", avg_loss),
                    ("accuracy", acc),
                    ("specificity", spec),
                    ("AUC", auc),
                    ("sensitivity", sensitivity),
                    ("ppv", ppv),
                    ("confusion_matrix", matrix),
                    ("f1", f1),
                ],
                phase="valid",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        self.reset_metrics("valid")
        self.step_losses["valid"].clear()
        
        # # Visualize misclassified samples if enabled and any exist
        # if self.collect_misclassified_samples and self.misclassified_samples["valid"]:
        #     self.visualize_misclassified_samples("valid")
    
    def test_step(self, batch):
        
        ecg = None

        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, targets, ecg = batch
            elif len(batch) == 2:
                x, targets = batch
            else:
                raise ValueError("Unexpected batch structure received in test_step.")
        elif isinstance(batch, dict):
            x = batch.get("x") or batch.get("ppg")
            targets = batch.get("y") or batch.get("labels")
            ecg = batch.get("ecg")
        else:
            raise ValueError("Unsupported batch type received in test_step.")

        output_logits = self(x)
        preds = torch.argmax(output_logits, dim=1)
        loss = F.cross_entropy(output_logits, targets)

        self.update_metrics(preds, targets, "test")
        self.step_losses["test"].append(loss)

        # Collect misclassified samples if enabled
        if self.collect_misclassified_samples:
            misclassified_mask = (preds != targets)
            if misclassified_mask.any() and len(self.misclassified_samples["test"]) < self.max_misclassified_to_store:
                misclassified_indices = torch.where(misclassified_mask)[0]
                for idx in misclassified_indices:
                    if len(self.misclassified_samples["test"]) >= self.max_misclassified_to_store:
                        break
                    sample_payload = {
                        'input': x[idx].detach().cpu().numpy(),
                        'prediction': preds[idx].item(),
                        'true_label': targets[idx].item(),
                        'probabilities': F.softmax(output_logits[idx], dim=0).detach().cpu().numpy()
                    }

                    if ecg is not None:
                        sample_payload['ecg'] = ecg[idx].detach().cpu().numpy()

                    self.misclassified_samples["test"].append(sample_payload)

        return {'test_loss': loss}

    def on_test_epoch_start(self):
        """Clear misclassified samples at the start of test epoch"""
        if self.collect_misclassified_samples:
            self.misclassified_samples["test"].clear()
    
    def on_test_epoch_end(self):
        """End of the training epoch"""
        avg_loss = sum(self.step_losses["test"]) / len(self.step_losses["test"])

        acc = matrix = f1 = spec = auc = sensitivity = ppv = None

        if "accuracy" in self.metrics_lst:
            acc = self.metrics["metrics_" + "test"]["accuracy"].compute()

        if "cf_matrix" in self.metrics_lst:
            matrix = self.metrics["metrics_" + "test"]["cf_matrix"].compute()

        if "f1" in self.metrics_lst:
            f1 = self.metrics["metrics_" + "test"]["f1"].compute()
        
        if "specificity" in self.metrics_lst:
            spec = self.metrics["metrics_" + "test"]["specificity"].compute()

        if "AUC"  in self.metrics_lst:
            auc = self.metrics["metrics_" + "test"]["AUC"].compute()

        if "sensitivity" in self.metrics_lst:
            sensitivity = self.metrics["metrics_" + "test"]["sensitivity"].compute()

        if "ppv" in self.metrics_lst:
            ppv = self.metrics["metrics_" + "test"]["ppv"].compute()
        
        self.log_all(
                items=[
                    ("loss", avg_loss),
                    ("accuracy", acc),
                    ("specificity", spec),
                    ("AUC", auc),
                    ("sensitivity", sensitivity),
                    ("ppv", ppv),
                    ("confusion_matrix", matrix),
                    ("f1", f1),
                ],
                phase="test",
                prog_bar=True,
                sync_dist_group=False,
            )
        
        self.reset_metrics("test")
        self.step_losses["test"].clear()
        
        # Visualize misclassified samples if enabled and any exist
        if self.collect_misclassified_samples and self.misclassified_samples["test"]:
            self.visualize_misclassified_samples("test")

    def predict_step(self, batch):
        x, targets = batch
        output_logits = self(x)
        preds = torch.argmax(output_logits, dim=1)
        return preds


    
    













    
