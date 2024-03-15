import numpy as np

from models.trainable.trainable_lstm_crbm import TrainableLSTMCRBM
from models.base.lstm_crbm import LSTMCRBM
from pathlib import Path
from data.datasets import SequenceDataModule
from data.data_preparation import get_filled_df, make_sequence_data
from torch.nn.functional import mse_loss, cosine_similarity
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import unpack_sequence
from scipy.stats import linregress

__ckpt_path = Path(__file__).parent.parent / "training/lightning_logs/version_26/checkpoints/epoch=153-step=26180.ckpt"
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def main():
    features = ["Hb",
                "Ca",
                "Cr",
                "LDH",
                "Alb",
                r"$\beta$2m",
                "M-Pr",
                r"SFL-$\lambda$",
                r"SFL-$\kappa$",
                "Lk", ]

    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model
    pass

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.test_dataset
    targets, packed_sequences = sg.collate_fn(val_data)

    pred = model.sample(packed_sequences, denoise=True)
    # mse = mse_loss(pred, targets)
    # cos_sim = cosine_similarity(pred, targets).mean()

    target_means = targets.std(axis=0)  # TODO
    pred_means = pred.std(axis=0)  # TODO

    r_mean_corr = torch.corrcoef(
        torch.cat([targets.std(axis=0, keepdim=True), pred.std(axis=0, keepdim=True)], dim=0))[0, 1]

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].scatter(target_means, pred_means, c=colors)
    for i, label in enumerate(features):
        ax[0].text(target_means[i], pred_means[i], label, ha='left', va='top', )
    ax[0].plot([target_means.min(), target_means.max()], [target_means.min(), target_means.max()], ls="--", c="k")
    ax[0].text(.05, .95, f'$r = {round(float(r_mean_corr), 2)}$', ha='left', va='top',
               transform=ax[0].transAxes)

    ax[0].set_title("Second-Order Moments")

    # plt.show()

    feature_corr_true = torch.corrcoef(targets.T)
    feature_corr_pred = torch.corrcoef(pred.T)

    indices = torch.triu_indices(10, 10, offset=1)
    triu_true, triu_pred = feature_corr_true[indices[0], indices[1]], feature_corr_pred[indices[0], indices[1]]
    r = torch.corrcoef(torch.stack([triu_true, triu_pred], dim=0))[0, 1]
    ax[1].scatter(triu_true, triu_pred, c=triu_true, cmap='PRGn', edgecolor='k', vmax=.6, vmin=-.6)
    ax[1].plot([triu_true.min(), triu_true.max()], [triu_true.min(), triu_true.max()], ls="--", c="k")
    ax[1].text(.05, .95, f'$r = {round(float(r), 2)}$', ha='left', va='top', transform=ax[1].transAxes)

    ax[1].set_title("Feature Correlations")

    fig.add_subplot(111, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Virtual Twin", fontsize=12, labelpad=30)
    plt.xlabel("Patient", fontsize=12, labelpad=30)
    plt.tight_layout()
    plt.show()

    r_feature_corr = torch.corrcoef(
        torch.stack([feature_corr_true[indices[0], indices[1]], feature_corr_pred[indices[0], indices[1]]], axis=0))
    pass


def show_dists():
    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.test_dataset
    targets, packed_sequences = sg.collate_fn(val_data)
    pass

    sample_dists = [model.sample(packed_sequences) for i in range(1000)]
    pass

    sample_dists = torch.stack(sample_dists, dim=1)
    sample_dists_mean = sample_dists.mean(dim=1)
    sample_dists_std = sample_dists.std(dim=1).mean(dim=0)
    mse = mse_loss(sample_dists_mean, targets)
    cos_sim = cosine_similarity(sample_dists_mean, targets).mean()
    pass


def mse_of_len():
    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.val_dataset
    targets, packed_sequences = sg.collate_fn(val_data)

    sample_outputs = torch.stack([model.sample(packed_sequences) for i in range(1000)], dim=1).mean(dim=1)
    pass

    unpacked_sequences = unpack_sequence(packed_sequences=packed_sequences)
    lens = torch.tensor([len(seq) for seq in unpacked_sequences])

    # steps = torch.arange(1, lens.max(), 3)
    # mse = [mse_loss(targets[((lens >= i) & (lens < j))], sample_outputs[((lens >= i) & (lens < j))]) for i, j in
    #        zip(steps[:-1], steps[1:])]
    #
    # plt.plot(torch.arange(2, lens.max(), 3), mse)
    # targets_subset, predictions_subset = targets[lens <= 12], sample_outputs[lens <= 12]

    mse = [mse_loss(targets[lens == i], sample_outputs[lens == i]) for i in torch.arange(1, 13)]
    plt.bar(torch.arange(1, 13), mse)
    plt.xticks(np.arange(1, 13), np.arange(1, 13) * 3)
    plt.xlabel("Months of prior observation")
    plt.ylabel(r"Prediction MSE [$\sigma$²]")
    plt.ylim(.25, None)

    plt.show()


def plot_feature_lines():
    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.val_dataset
    targets, packed_sequences = sg.collate_fn(val_data)

    sample_outputs = torch.stack([model.sample(packed_sequences) for i in range(1000)], dim=1).mean(dim=1)

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
        ax_temp.text(.05, .95, f'$r = {round(res.rvalue, 2)}$', ha='left', va='top',
                     transform=ax_temp.transAxes)

    fig.add_subplot(111, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Virtual Twin", fontsize=14, labelpad=30)
    plt.xlabel("Patient", fontsize=14, labelpad=30)

    plt.show()


def plot_feature_corr_mat():
    features = ["Hb",
                "Ca",
                "Cr",
                "LDH",
                "Alb",
                r"$\beta$2m",
                "M-Pr",
                r"SFL-$\lambda$",
                r"SFL-$\kappa$",
                "Lk", ]
    lightning_module = TrainableLSTMCRBM.load_from_checkpoint(__ckpt_path)
    model = lightning_module.model

    sg = SequenceDataModule(sequence_list=make_sequence_data(get_filled_df()))
    sg.setup()
    val_data = sg.val_dataset
    targets, packed_sequences = sg.collate_fn(val_data)

    sample_outputs = torch.stack([model.sample(packed_sequences) for i in range(1000)], dim=1).mean(dim=1)

    corr_mat_true = torch.corrcoef(targets.T)
    corr_mat_pred = torch.corrcoef(sample_outputs.T)
    pass

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].imshow(corr_mat_true, cmap='PRGn', vmax=.75, vmin=-.75)
    ax[0].set_title("Patient")
    ax[0].set_yticks(np.arange(10), features)
    ax[0].set_xticks(np.arange(10), features, rotation=90)
    ax[1].imshow(corr_mat_pred, cmap='PRGn', vmax=.75, vmin=-.75)
    ax[1].set_title("Virtual Twin")
    ax[1].set_xticks(np.arange(10), features, rotation=90)

    fig.add_subplot(111, frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("Feature Correlations")
    plt.show()


if __name__ == "__main__":
    main()
    # show_dists()
    # mse_of_len()
    # plot_feature_lines()
    # plot_feature_corr_mat()
