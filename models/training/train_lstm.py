from models.trainable.trainable_lstm import TrainableLSTM
from data.datasets import SequenceDataModule
from lightning import Trainer, seed_everything
from data.data_preparation import make_sequence_data, get_filled_df


def run(
        max_epochs=50,
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
    model = TrainableLSTM(n_features=10, lstm_units=32,)

    # Train it!
    trainer.fit(model, datamodule)

    return trainer, model, datamodule, model.model


if __name__ == "__main__":
    lightning_trainer, lightning_model, lightning_datamodule, model = run()
