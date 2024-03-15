from models.trainable.trainable_lstm_crbm import TrainableLSTMCRBM
from data.datasets import SequenceDataModule
from lightning import Trainer, seed_everything
from data.data_preparation import make_sequence_data, get_filled_df
from torchinfo import summary


def run(
        max_epochs=250,
        batch_size=32,
        seed=1,
):
    # Setup the trainer
    seed_everything(seed)
    trainer = Trainer(
        # accelerator=accelerator(),
        accelerator="cpu",
        max_epochs=max_epochs,
    )

    datamodule = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()), batch_size=batch_size)

    # Construct our model
    model = TrainableLSTMCRBM(n_features=10,
                              lstm_units=32,
                              crbm_hidden=16,
                              lstm_weight_decay=.1,
                              mc_steps=128,
                              weights_weight_decay=.2,
                              bias_weight_decay=.1,
                              precision_weight_decay=.1,
                              learning_rate=1e-3,)
    summary(model.model)

    # Train it!
    trainer.fit(model, datamodule)

    return trainer, model, datamodule, model.model


if __name__ == "__main__":
    lightning_trainer, lightning_model, lightning_datamodule, model = run()
