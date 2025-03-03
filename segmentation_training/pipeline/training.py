import lightning as L
import torch
import segmentation_models_pytorch as smp



class SegmentationModel(L.LightningModule):
    def __init__(self, architecture, encoder, lr, validation_metric="f1_score"):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = smp.create_model(
            arch=architecture, encoder_name=encoder, in_channels=4, classes=1, activation="sigmoid"
        )
        
        self.loss_fn = smp.losses.DiceLoss(mode="binary")
        self.validation_metric = getattr(smp.metrics, validation_metric)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs.contiguous(), labels.long())

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs.contiguous(), labels.long())

        hard_res = (outputs.contiguous() > 0.5).to(torch.int)[:, 0]
        tp, fp, fn, tn = smp.metrics.get_stats(hard_res, labels, mode="binary")
        score = self.validation_metric(tp, fp, fn, tn, reduction="micro")

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"val_{self.hparams.validation_metric}", score, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
