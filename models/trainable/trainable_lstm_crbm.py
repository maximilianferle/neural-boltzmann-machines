import torch
from models.base.lstm_crbm import LSTMCRBM
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainableLSTMCRBM(LightningModule):
    def __init__(self,
                 n_features: int,
                 lstm_units: int,
                 crbm_hidden: int,

                 mc_steps: int = 32,
                 lstm_weight_decay: float = .2,
                 bias_weight_decay: float = .5,
                 precision_weight_decay: float = .5,
                 weights_weight_decay: float = 1.,
                 learning_rate: float = 5e-5,
                 ):
        super().__init__()
        self.mc_steps = mc_steps
        self.lstm_weight_decay = lstm_weight_decay
        self.bias_weight_decay = bias_weight_decay
        self.precision_weight_decay = precision_weight_decay
        self.weights_weight_decay = weights_weight_decay
        self.learning_rate = learning_rate

        self.model = LSTMCRBM(n_features=n_features,
                              lstm_units=lstm_units,
                              crbm_hidden=crbm_hidden, )
        self.save_hyperparameters()

    def training_step(self, batch, batch_index):
        """Train on CD loss."""
        y, x = batch
        losses = self.model.compute_loss(y, x, mc_steps=self.mc_steps)
        self.log("CD_Loss", losses["CD_loss"], prog_bar=True)
        self.log("MSE_Bias_Loss", losses["MSE_bias_loss"], prog_bar=True)
        self.log("MSE_Var_Loss", losses["MSE_variance_loss"], prog_bar=True)
        # return losses["CD_loss"]
        return losses["CD_loss"]

    def validation_step(self, batch, batch_index):
        """Validate"""
        y, x = batch
        losses = self.model.compute_loss(y, x, mc_steps=self.mc_steps, denoise=True)
        self.log("val_MSE", losses["MSE_pred"], prog_bar=True)
        self.log("val_cosine_similarity", losses["cosine_similarity"], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.lstm.parameters(),
                    "weight_decay": self.lstm_weight_decay,
                },
                {
                    "params": self.model.crbm.bias_net.parameters(),
                    "weight_decay": self.bias_weight_decay,
                },
                {
                    "params": self.model.crbm.precision_net.parameters(),
                    "weight_decay": self.precision_weight_decay,
                },
                {
                    "params": self.model.crbm.weights_net.parameters(),
                    "weight_decay": self.weights_weight_decay,
                },
            ],
            lr=self.learning_rate,
        )
        scheduler = ReduceLROnPlateau(optimizer=optimizer)
        return [optimizer]
