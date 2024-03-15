import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.nn.utils.rnn import pack_sequence

from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from lightning import LightningDataModule

import numpy as np
from sklearn.preprocessing import PowerTransformer

from data.data_preparation import get_filled_df, make_sequence_data
from abc import abstractmethod


class IsingTransform:
    def __call__(self, x, noise_scale=1e-3):
        """
        Turn grayscale images between 0 and 1 into binary variables of +/- 1.

        Args:
            x: input (tensors)
            noise_scale (float): scalar to set noise scale.
                Between 1e-3 and 1e-2 leads to same training convergence.
                Needs to be non-zero due to degenerate variance of some pixels in unmodified MNIST.

        Returns:  output (tensors)

        """
        x_clamped = torch.clamp(x, noise_scale, 1 - noise_scale)
        return 2 * torch.bernoulli(x_clamped) - 1


class GaussianTransform:
    def __call__(self, x, noise_scale=1e-3):
        """
        Turn grayscale images between 0 and 1 into centered Gaussian variables.

        Args:
            x (tensor): input
            noise_scale (float): scalar to set noise scale.
                Between 1e-3 and 1e-2 leads to same training convergence.
                Needs to be non-zero due to degenerate variance of some pixels in unmodified MNIST.

        Returns: output (tensors)

        """
        center = 2.0 * x - 1.0
        noise = noise_scale * torch.randn_like(x)

        return center + noise


class TransformTargetsToOneHot:
    def __init__(self, num_classes):
        """
        Transforms targets to one_hot variables.

        Args:
            num_classes (int): number of classes
        """
        self.num_classes = num_classes

    def __call__(self, x):
        """
        Args:
            x (tensors): input

        Returns: output (tensors)

        """
        return (
            F.one_hot(torch.tensor(x).view(-1, 1), self.num_classes).squeeze().float()
        )


class DataModule(LightningDataModule):
    def __init__(
            self,
            data_dir,
            batch_size=32,
            num_workers=1,
            output_type="gaussian",
            dataset_name="MNIST",
    ):
        """
        Create DataModule.

        Args:
            data_dir (str): directory to download data
            batch_size (int) : size of a batch in a dataloader
            num_workers (int) : number of workers in dataloader
            output_type (str) : whether outputs should be gaussian or ising
            dataset_name (str) : which dataset to create
        """

        super().__init__()

        self.num_classes = 10
        self.dims = (1, 28, 28)
        self._val_size = self.num_classes * 500

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if output_type == "gaussian":
            output_transform = GaussianTransform()
        elif output_type == "ising":
            output_transform = IsingTransform()
        else:
            raise ValueError(f"Unrecognized output type {output_type}")
        self.output_type = output_type

        if dataset_name == "MNIST":
            dataset = MNIST
        elif dataset_name == "FashionMNIST":
            dataset = FashionMNIST
        else:
            raise ValueError(f"Unrecognized dataset name {dataset_name}")
        self.dataset = dataset

        self.transform = transforms.Compose([transforms.ToTensor(), output_transform])
        self.target_transform = TransformTargetsToOneHot(self.num_classes)

        # These are datasets that will get created in self.setup
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # Download the data
        self.dataset(self.data_dir, train=True, download=True)
        self.dataset(self.data_dir, train=False, download=True)

    def setup(self, stage):
        train_val = self.dataset(
            self.data_dir,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        train_size = len(train_val) - self._val_size

        self.train, self.val = random_split(train_val, [train_size, self._val_size])

        self.test = self.dataset(
            self.data_dir,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class _BaseDataModule(LightningDataModule):
    """
    Base data provider class for (LSTM)NBM training.
    """

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self._split_data()
        self._fit_power_transformer()

    @staticmethod
    def __raise():
        raise NotImplementedError("Base Class not meant to be instanced. Please Subclass.")

    @abstractmethod
    def _split_data(self, *args, **kwargs):
        self.__raise()

    @abstractmethod
    def _fit_power_transformer(self, *args, **kwargs):
        self.__raise()

    @abstractmethod
    def _make_dataset(self, *args, **kwargs):
        self.__raise()

    def setup(self, stage=None):

        # Prepare datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = self._make_dataset(self._train)
            self.val_dataset = self._make_dataset(self._val)

        if stage == 'test' or stage is None:
            self.test_dataset = self._make_dataset(self._test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def normalize(self, input_sequence):
        concatenated_sequence, sequence_indices = self._concatenate_sequences(input_sequence)
        transformed_sequence = self.power_transformer.transform(concatenated_sequence)
        split_sequences = self._split_transformed_sequences(transformed_sequence, sequence_indices)
        return split_sequences

    def denormalize(self, input_sequence):
        concatenated_sequence, sequence_indices = self._concatenate_sequences(input_sequence)
        transformed_sequence = self.power_transformer.inverse_transform(concatenated_sequence)
        split_sequences = self._split_transformed_sequences(transformed_sequence, sequence_indices)
        return split_sequences

    @staticmethod
    def _concatenate_sequences(input_sequence):
        concatenated_sequence = np.concatenate(input_sequence, axis=0)
        sequence_indices = np.cumsum([0] + [len(seq) for seq in input_sequence])
        return concatenated_sequence, sequence_indices

    @staticmethod
    def _split_transformed_sequences(input_sequence, sequence_indices):
        split_sequences = [input_sequence[start:end] for start, end in zip(sequence_indices[:-1], sequence_indices[1:])]
        return split_sequences

    def train(self, numpy: bool = False):
        return self._make_dataset(self._train, numpy=numpy)

    def val(self, numpy: bool = False):
        return self._make_dataset(self._val, numpy=numpy)

    def test(self, numpy: bool = False):
        return self._make_dataset(self._test, numpy=numpy)


class _NextStepOverDataModule(_BaseDataModule):
    def __init__(self,
                 sequence_list: list = None,
                 batch_size: int = 32):
        self.sequence_list = make_sequence_data(get_filled_df()) if sequence_list is None else sequence_list
        super().__init__(batch_size=batch_size)

    def _split_data(self):
        n_samples = len(self.sequence_list)
        self._train = self.sequence_list[:int(.7 * n_samples)]  # 70% training
        self._val = self.sequence_list[int(.7 * n_samples):int(.9 * n_samples)]  # 20% validation
        self._test = self.sequence_list[int(.9 * n_samples):]  # 10% testing

    def _fit_power_transformer(self):
        # Fit the PowerTransformer on the concatenated training data
        self.power_transformer = PowerTransformer()
        normalization_data, _ = self._concatenate_sequences(self._train)
        self.power_transformer.fit(normalization_data)


class WindowDataModule(_NextStepOverDataModule):
    """
    Data module class providing sliding window samples for NBM training.
    """

    def _make_dataset(self, data, numpy: bool = False):
        data = self.normalize(data)
        inputs, targets = [], []
        for sequence in data:
            n_samples = sequence.shape[0]

            if n_samples < 4:
                continue

            for i, j in zip(range(n_samples), range(3, n_samples)):
                inputs.append(sequence[i:j, :].ravel())
                targets.append(sequence[j, :])

        if numpy:
            return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        else:
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            return TensorDataset(targets_tensor, inputs_tensor)


class SequenceDataset(Dataset):
    """
    Convenience class to be used as a wrapper for providing whole-sequence samples.
    """

    def __init__(self, targets, sequences):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.targets[idx], self.sequences[idx]


class SequenceDataModule(_NextStepOverDataModule):
    """
    Data module class providing whole sequence samples for LSTMNBM training.
    """

    def _make_dataset(self, data, numpy: bool = False):
        data = self.normalize(data)
        sequences, targets = [], []
        for sequence in data:
            n_samples = sequence.shape[0]

            if n_samples < 2:
                continue

            for i in range(1, n_samples):
                sequences.append(torch.tensor(sequence[:i, :], dtype=torch.float32))
                targets.append(torch.tensor(sequence[i, :], dtype=torch.float32))

        return SequenceDataset(targets, sequences)

    @staticmethod
    def collate_fn(batch):
        # Unzip batch
        targets, sequences = zip(*batch)
        sequences, targets = zip(*sorted(zip(sequences, targets), key=lambda x: len(x[0]), reverse=True))

        # Pack the padded sequences
        packed_sequences = pack_sequence(sequences, enforce_sorted=True)
        targets = torch.cat([target.unsqueeze(0) for target in targets], dim=0)
        return targets, packed_sequences

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


class SequenceGradientDataModule(SequenceDataModule):
    """
    Sublcass for proving relative gradient-step sequences as opposed to absolute value sequences for LSTMNBM training.
    """

    def _make_dataset(self, data, numpy: bool = False):
        data = self.normalize(data)
        sequences, targets = [], []
        for sequence in data:
            n_samples = sequence.shape[0]

            if n_samples < 2:
                continue

            sequence_grad = sequence[1:, :] - sequence[:-1, :]
            for i in range(1, n_samples):
                sequences.append(torch.tensor(sequence[:i, :], dtype=torch.float32))
                targets.append(torch.tensor(sequence_grad[i - 1, :], dtype=torch.float32))

        return SequenceDataset(targets, sequences)
