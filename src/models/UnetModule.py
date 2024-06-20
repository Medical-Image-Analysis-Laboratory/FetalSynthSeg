from typing import Any, Dict, Tuple
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, Dice
import monai


class Unet(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        lr: float = None
    ) -> None:
        """Initialize a UnetModule.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss function
        self.criterion = monai.losses.DiceCELoss(to_onehot_y=True,
                                                 include_background=False,
                                                 softmax=True,)

        # metric objects for calculating and averaging accuracy across batches
        self.train_dsc = Dice(ignore_index=0, num_classes=8,)
        self.val_dsc_synth = Dice(ignore_index=0, num_classes=8,)
        self.val_dsc = Dice(ignore_index=0, num_classes=8,)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss_synth = MeanMetric()
        self.val_loss = MeanMetric()
        # for tracking best so far validation accuracy
        self.val_dsc_best = MaxMetric()

        self.test_loss_synth = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_dsc = Dice(ignore_index=0, num_classes=8,)
        self.test_dsc_synth = Dice(ignore_index=0, num_classes=8,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, names = batch#['image'], batch['label']
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        # names = batch['names']
        return {'preds': preds, 'names': names, 'labels': y, 'image': x}

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_dsc.reset()
        self.val_dsc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, names = batch#['image'], batch['label']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_dsc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/dsc", self.train_dsc, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        if dataloader_idx == 0:
            # update and log metrics
            self.val_loss(loss)
            self.val_dsc(preds, targets)
            self.log("val/loss", self.val_loss, on_step=True,
                     on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("val/dsc", self.val_dsc, on_step=True,
                     on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        else:
            # update and log metrics
            self.val_loss_synth(loss)
            self.val_dsc_synth(preds, targets)
            self.log("val/synth_loss", self.val_loss_synth,
                     on_step=True, on_epoch=True, prog_bar=True,
                     add_dataloader_idx=False)
            self.log("val/synth_dsc", self.val_dsc_synth,
                     on_step=True, on_epoch=True, prog_bar=True,
                     add_dataloader_idx=False)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        if dataloader_idx == 0:
            # update and log metrics
            self.test_loss(loss)
            self.test_dsc(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False,
                     on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test/dsc", self.test_dsc, on_step=True,
                     on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        else:
            # update and log metrics
            self.test_loss_synth(loss)
            self.test_dsc_synth(preds, targets)
            self.log("test/synth_loss", self.test_loss_synth,
                     on_step=False, on_epoch=True, prog_bar=True,
                     add_dataloader_idx=False)
            self.log("test/synth_dsc", self.test_dsc_synth,
                     on_step=True, on_epoch=True, prog_bar=True,
                     add_dataloader_idx=False)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_dsc.compute()  # get current val acc
        self.val_dsc_best(acc)  # update best so far val acc
        # log `val_dsc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/dsc_best", self.val_dsc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters(),
                                           lr=self.hparams.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Unet(None, None, None, None)