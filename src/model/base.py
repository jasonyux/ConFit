from abc import ABC, abstractmethod
import pytorch_lightning as pl
import wandb

class BaseModel(pl.LightningModule, ABC):
    """shared functions across all models in this project
    """
    metrics = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _log_metrics(self, preds, labels, mode='train'):
        preds = preds.detach()
        labels = labels.to(preds.device)

        for name, metric in self.metrics[mode].items():
            metric = metric.to(preds.device)
            m = metric(preds, labels)

            # defines summary metrics for wandb
            if self.trainer.global_step == 0 and wandb.run is not None:
                if m.dim() == 0:
                    wandb.define_metric(f"train/{name}_epoch", summary='max')
                else:
                    for i in range(len(m)):
                        wandb.define_metric(f"train/class{i}_{name}_epoch", summary='max')
                        wandb.define_metric(f"val/class{i}_{name}_epoch", summary='max')
            
            # log metrics
            if m.dim() == 0:
                self.log(
                    f"{mode}/{name}",
                    m.item(),
                    on_step=True, on_epoch=False, sync_dist=True,
                    batch_size=len(labels)
                )
            else:
                for i, score in enumerate(m):
                    self.log(
                        f"{mode}/{name}_class_{i}",
                        score,
                        on_step=True, on_epoch=False, # on_epoch=True will use torch.mean, which is not correct
                        sync_dist=True,
                        batch_size=len(labels)
                    )
        return

    def on_train_epoch_end(self) -> None:
        if 'train' not in self.metrics:
            return
        
        for name, metric in self.metrics['train'].items():
            # log metrics
            m = metric.compute()
            if m.dim() == 0:
                self.log(
                    f"train/{name}_epoch",
                    m.item(), 
                    on_step=False, on_epoch=True,
                    sync_dist=True
                )
            else:
                for i, score in enumerate(m):
                    self.log(
                        f"train/class{i}_{name}_epoch",
                        score,
                        on_step=False, on_epoch=True,
                        sync_dist=True
                    )
            metric.reset() # need to manually reset
        return

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if 'val' not in self.metrics:
            return
        
        for name, metric in self.metrics['val'].items():
            # log metrics
            m = metric.compute()
            if m.dim() == 0:
                self.log(
                    f"val/{name}_epoch",
                    m.item(), 
                    on_step=False, on_epoch=True,
                    sync_dist=True
                )
            else:
                for i, score in enumerate(m):
                    if m.dim() == 0:
                        self.log(
                            f"val/{name}_epoch",
                            m.item(), 
                            on_step=False, on_epoch=True,
                            sync_dist=True
                        )
                    else:
                        for i, score in enumerate(m):
                            self.log(
                                f"val/class{i}_{name}_epoch",
                                score,
                                on_step=False, on_epoch=True,
                                sync_dist=True
                            )
            metric.reset()
        return