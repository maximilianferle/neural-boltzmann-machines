import torch
from models.base.lstm import LSTMPredictor
from lightning import LightningModule
from torch.nn.functional import mse_loss, cosine_similarity


class TrainableLSTM(LightningModule):
    def __init__(self,
                 n_features: int,
                 lstm_units: int,
                 lstm_weight_decay: float = .2,
                 learning_rate: float = 1e-3,
                 ):
        super().__init__()
        self.lstm_weight_decay = lstm_weight_decay

        self.model = LSTMPredictor(n_features=n_features,
                                   lstm_units=lstm_units, )
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def training_step(self, batch, batch_index):
        y, x = batch
        y_pred = self.model(x)
        mse = mse_loss(y, y_pred)
        self.log("MSE", mse, prog_bar=True)
        return mse

    def validation_step(self, batch, batch_index):
        y, x = batch
        y_pred = self.model(x)
        mse = mse_loss(y, y_pred)
        cos_sim = cosine_similarity(y, y_pred).mean()
        self.log("val_MSE", mse, prog_bar=True)
        self.log("val_cosine_similarity", cos_sim, prog_bar=True)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.parameters(),
                    "weight_decay": self.lstm_weight_decay,
                }
            ],
            lr=self.learning_rate,
        )
        return [optimizer]
