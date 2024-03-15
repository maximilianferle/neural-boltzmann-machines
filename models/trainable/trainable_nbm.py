import torch

from lightning import LightningModule
from models.base.nbm import NBM
import math


class TrainableNBM(LightningModule):
    """
    Converts underlying Neural Boltzmann Machine class to a Lightning Module
    """

    def __init__(
            self,
            image_shape,
            num_classes,
            latent_size,
            visible_unit_type,
            mc_steps,
            bias_net_weight_decay,
            precision_net_weight_decay,
            weights_net_weight_decay,
            learning_rate,
            logger_dir,
    ):
        """
        Args:
            image_shape: Shape of images (tuple of int x int)
            num_classes: Number of classes in dataset (int)
            latent_size: Size of the latent space (int)
            visible_unit_type: type of visible units of (str)
            mc_steps: Number of MCMC sampling steps (int)
            bias_net_weight_decay: L2 penalty to apply to bias network (float)
            precision_net_weight_decay: L2 penalty to apply to precision network (float)
            weights_net_weight_decay: L2 penalty to apply to weights network (float)
            learning_rate: learning rate (float)
            logger_dir: directory where outputs should get logged (str)
        """
        super().__init__()

        self.image_shape = image_shape
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.mc_steps = mc_steps
        self.bias_net_weight_decay = bias_net_weight_decay
        self.precision_net_weight_decay = precision_net_weight_decay
        self.weights_net_weight_decay = weights_net_weight_decay
        self.learning_rate = learning_rate
        self.logger_dir = logger_dir

        self.model = NBM(
            nx=num_classes,
            ny=math.prod(image_shape),
            nh=latent_size,
            visible_unit_type=visible_unit_type
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_index):
        """Train on CD loss."""
        y, x = batch
        losses = self.model.compute_loss(y, x, mc_steps=self.mc_steps)
        self.log("CD_Loss", losses["CD_loss"], prog_bar=True)
        self.log("MSE_Bias_Loss", losses["MSE_bias_loss"], prog_bar=True)
        self.log("MSE_Var_Loss", losses["MSE_variance_loss"], prog_bar=True)
        self.log("MSE_pred", losses["MSE_pred"], prog_bar=True)
        return losses["CD_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.bias_net.parameters(),
                    "weight_decay": self.bias_net_weight_decay,
                },
                {
                    "params": self.model.precision_net.parameters(),
                    "weight_decay": self.precision_net_weight_decay,
                },
                {
                    "params": self.model.weights_net.parameters(),
                    "weight_decay": self.weights_net_weight_decay,
                },
            ],
            lr=self.learning_rate,
        )
        return [optimizer]
