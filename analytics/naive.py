import matplotlib.pyplot as plt
from data.data_preparation import make_sequence_data, get_filled_df
from data.datasets import SequenceDataModule
import numpy as np
from scipy.stats import linregress
from torch.nn.functional import mse_loss, cosine_similarity
# from sklearn.metrics.pairwise import cosine_similarity
from data.datasets import SequenceDataset
from pathlib import Path
from models.trainable.trainable_lstm_crbm import TrainableLSTMCRBM
import torch
from torch.nn.utils.rnn import unpack_sequence

__ckpt_path = Path(
    __file__).parent.parent / "models/training/lightning_logs/version_26/checkpoints/epoch=153-step=26180.ckpt"
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def get_naive_preds():
    sg = SequenceDataModule()

    data = sg.normalize(sg._val)
    preds, targets, seq_idx = [], [], []

    for sequence in data:
        n_samples = sequence.shape[0]

        if n_samples < 2:
            continue

        seq_idx.append(len(sequence))

        for i in range(1, n_samples):
            preds.append(sequence[i - 1, :])
            targets.append(sequence[i, :], )

    return np.array(targets), np.array(preds), seq_idx


def get_model_preds():
    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.val_dataset
    targets, packed_sequences = sg.collate_fn(val_data)
    sequences = unpack_sequence(packed_sequences)
    pass

    true_grads = torch.stack([targets[i, :] - sequences[i][-1, :] for i in range(1414)], dim=1)

    sample_outputs = torch.stack([model.sample(packed_sequences) for i in range(1000)], dim=1).mean(dim=1)
    pred_grads = torch.stack([sample_outputs[i, :] - sequences[i][-1, :] for i in range(1414)], dim=1)
    pass

    features = ["Hemoglobin",
                "Calcium",
                "Creatinine",
                "LDH",
                "Albumin",
                r"$\beta$-2-Microglobulin",
                "M-Protein",
                r"SFL-$\lambda$",
                r"SFL-$\kappa$",
                "Leukocytes", ]
    fig, ax = plt.subplots(nrows=2, ncols=5)

    for i in range(10):
        a, b = true_grads[i, :], pred_grads[i, :]
        res = linregress(a, b)
        row = i % 5
        col = i // 5
        ax_temp = ax[col, row]
        ax_temp.scatter(a, b, c=colors[i], marker=".", alpha=.5)
        # ax_temp.plot([a.min(), a.max()], [res.slope * a.min() + res.intercept, res.slope * a.max() + res.intercept], ls="--", c="k")
        ax_temp.plot([a.min(), a.max()], [a.min(), a.max()], ls="--", c="k")
        ax_temp.set_title(features[i])
        # ax_temp.set_yticklabels([])
        # ax_temp.set_xticklabels([])
        # ax_temp.text(.05, .95, f'$m = {round(res.slope, 2)}$\n$R² = {round(res.rvalue, 2)}$', ha='left', va='top',
        #              transform=ax_temp.transAxes)
        ax_temp.text(.05, .95, f'$r = {round(res.rvalue, 2)}$', ha='left', va='top',
                     transform=ax_temp.transAxes)

    fig.add_subplot(111, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Virtual Twin", fontsize=14, labelpad=30)
    plt.xlabel("Patient", fontsize=14, labelpad=30)
    plt.title(r"$\frac{d}{dt}\vec{x}$", fontsize=18, pad=36, weight="bold")

    plt.show()


    return sample_outputs.numpy()


def plot_feature_lines():
    targets, sample_outputs, *_ = get_naive_preds()

    features = ["Hemoglobin",
                "Calcium",
                "Creatinine",
                "LDH",
                "Albumin",
                r"$\beta$-2-Microglobulin",
                "M-Protein",
                r"SFL-$\lambda$",
                r"SFL-$\kappa$",
                "Leukocytes", ]
    fig, ax = plt.subplots(nrows=2, ncols=5)

    for i in range(10):
        a, b = targets[:, i], sample_outputs[:, i]
        res = linregress(a, b)
        row = i % 5
        col = i // 5
        ax_temp = ax[col, row]
        ax_temp.scatter(a, b, c=colors[i], marker=".", alpha=.5)
        # ax_temp.plot([a.min(), a.max()], [res.slope * a.min() + res.intercept, res.slope * a.max() + res.intercept], ls="--", c="k")
        ax_temp.plot([a.min(), a.max()], [a.min(), a.max()], ls="--", c="k")
        ax_temp.set_title(features[i])
        # ax_temp.set_yticklabels([])
        # ax_temp.set_xticklabels([])
        # ax_temp.text(.05, .95, f'$m = {round(res.slope, 2)}$\n$R² = {round(res.rvalue, 2)}$', ha='left', va='top',
        #              transform=ax_temp.transAxes)
        ax_temp.text(.05, .95, f'$r² = {round(res.rvalue, 2)}$', ha='left', va='top',
                     transform=ax_temp.transAxes)

    fig.add_subplot(111, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Virtual Twin", fontsize=14, labelpad=30)
    plt.xlabel("Patient", fontsize=14, labelpad=30)

    plt.show()


def look_at_naive_grads():
    targets, naive_preds, seq_idx = get_naive_preds()
    model_preds = get_model_preds()
    naive_sim = cosine_similarity(targets, naive_preds)
    model_sim = cosine_similarity(targets, model_preds)
    pass


if __name__ == "__main__":
    # plot_feature_lines()
    # look_at_naive_grads()
    get_model_preds()
