import json
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from .data_loader import Corpus


def visualize_batch(
    data_corpus: Corpus,
    example_batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
    example_preds=None,
    max_viz_samples=8,
    torch=True,
) -> str:
    viz_strings = []
    batch_size = example_batch[0].size(1)
    for i in range(min(max_viz_samples, batch_size)):
        viz_strings.append(f"Example {i}:")
        if torch:
            input_x = example_batch[0][:, i].cpu()
            target_y = example_batch[1][:, i].cpu()
        else:
            input_x = example_batch[0][:, i]
            target_y = example_batch[1][:, i]
        if example_preds is not None:
            if torch:
                pred_y = example_preds[:, i].cpu()
            else:
                pred_y = example_preds[:, i]
        decoded_x = "".join(data_corpus.vocab.get_char_symbols(input_x))
        decoded_y = "".join(data_corpus.vocab.get_char_symbols(target_y))
        viz_strings.append(f"Inputs:  {decoded_x}")
        viz_strings.append(f"Targets: {decoded_y}")
        if example_preds is not None:
            decoded_pred_y = "".join(data_corpus.vocab.get_char_symbols(pred_y))
            viz_strings.append(f"Predictions:   {decoded_pred_y}")

    return "\n".join(viz_strings)


def _compute_interspike_intervals(
    spike_record: np.ndarray,
    delta_time: float = 1.0,
) -> np.ndarray:
    assert spike_record.ndim == 2
    spike_times = [
        delta_time * np.flatnonzero(spike_record[:, i])
        for i in range(spike_record.shape[1])
    ]
    isis = [np.diff(times) for times in spike_times]
    isis = [x for x in isis if len(x) > 0]
    return np.concatenate(isis) if len(isis) > 0 else np.array([])


def _compute_pairwise_correlations(spike_record: np.ndarray) -> np.ndarray:
    assert spike_record.ndim == 2

    # remove silent / constant neurons to avoid NaN correlations
    spike_record = spike_record[:, spike_record.std(axis=0) > 0]
    if spike_record.shape[1] < 2:
        return np.array([])

    corr = np.corrcoef(spike_record.T.astype(np.float64))
    idx_i, idx_j = np.triu_indices(corr.shape[0], k=1)
    return corr[idx_i, idx_j]


def _insert_zero_crossings(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign_change = y[:-1] * y[1:] < 0
    if not np.any(sign_change):
        return x, y

    idx = np.flatnonzero(sign_change)
    alpha = -y[idx] / (y[idx + 1] - y[idx])
    x_cross = x[idx] + alpha * (x[idx + 1] - x[idx])

    x = np.insert(x, idx + 1, x_cross)
    y = np.insert(y, idx + 1, 0.0)
    return x, y


def _plot_neuron_trace(
    ax,
    signal: np.ndarray,
    neuron_idx: int,
    add_legend: bool = False,
):
    x = np.arange(len(signal), dtype=np.float64)
    x, signal = _insert_zero_crossings(x, signal)

    ax.plot(
        x,
        np.where(signal <= 0.0, signal, np.nan),
        color="#0173B2",
        lw=1.0,
        alpha=0.9,
        label="sub-threshold" if add_legend else None,
    )
    ax.plot(
        x,
        np.where(signal >= 0.0, signal, np.nan),
        color="#029E73",
        lw=1.0,
        alpha=0.9,
        label="activity" if add_legend else None,
    )
    ax.axhline(
        0.0,
        color="#DE8F05",
        linestyle="--",
        lw=0.9,
        alpha=0.85,
    )
    ax.text(
        0.995,
        0.92,
        f"Neuron #{neuron_idx}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    if add_legend:
        ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)


def calculate_and_plot_spiking_stats(
    layer_activity: np.ndarray,
    neuron_outputs: np.ndarray,
    neuron_biases: np.ndarray,
    max_interval: int = 200,
    path: Optional[str] = None,
    plot_num_steps: Optional[int] = None,
    plot_num_neurons: int = 1024,
    trace_neuron_indices: tuple[int, ...] = (1, 2, 3),
    stats_burn_in: int = 100,
    delta_time: float = 1.0,
    eps: float = 1e-6,
):
    layer_activity = np.asarray(layer_activity)
    neuron_outputs = np.asarray(neuron_outputs)
    neuron_biases = np.asarray(neuron_biases).squeeze()

    assert layer_activity.ndim == 2
    assert neuron_outputs.ndim == 2
    assert neuron_biases.ndim == 1
    assert layer_activity.shape == neuron_outputs.shape
    assert layer_activity.shape[1] == neuron_biases.shape[0]

    # ---------------- data prep ----------------

    num_steps, num_neurons = layer_activity.shape
    plot_num_steps = min(num_steps, plot_num_steps or num_steps)
    plot_num_neurons = min(num_neurons, plot_num_neurons)

    activity = layer_activity[-plot_num_steps:, :plot_num_neurons]
    neuron_outputs = neuron_outputs[-plot_num_steps:, :plot_num_neurons]
    neuron_biases = neuron_biases[:plot_num_neurons]

    spike_record = np.asarray(activity > 0.0, dtype=bool)

    burn_in = min(stats_burn_in, max(0, spike_record.shape[0] - 1))
    stat_spike_record = spike_record[burn_in:]

    # ---------------- stats ----------------

    spike_counts = stat_spike_record.sum(axis=0)
    activity_levels = spike_counts / (stat_spike_record.shape[0] * delta_time + eps)
    active_fraction = 100.0 * spike_record.mean(axis=1)

    isis = _compute_interspike_intervals(stat_spike_record, delta_time=delta_time)
    if len(isis) > 0:
        isi_unique, isi_counts = np.unique(isis, return_counts=True)
        isi_prob = isi_counts / (len(isis) + eps)
        isis_mean = float(np.mean(isis))
        isis_std = float(np.std(isis))
        isis_cv = float(isis_std / (isis_mean + eps))
    else:
        isi_unique = np.array([0.0])
        isi_prob = np.array([0.0])
        isis_mean = float("nan")
        isis_std = float("nan")
        isis_cv = float("nan")

    pairwise_corr = _compute_pairwise_correlations(stat_spike_record)
    if len(pairwise_corr) > 0:
        pairwise_corr_mean = float(np.mean(pairwise_corr))
        pairwise_corr_std = float(np.std(pairwise_corr))
    else:
        pairwise_corr_mean = float("nan")
        pairwise_corr_std = float("nan")

    stats = {
        "num_steps": int(stat_spike_record.shape[0]),
        "num_neurons": int(stat_spike_record.shape[1]),
        "num_intervals": int(len(isis)),
        "activity_mean": float(np.mean(activity_levels)),
        "activity_std": float(np.std(activity_levels)),
        "num_silent_neurons": int(np.sum(spike_counts == 0)),
        "fano_factor": float(spike_counts.var() / (spike_counts.mean() + eps)),
        "isis_mean": isis_mean,
        "isis_std": isis_std,
        "isis_cv": isis_cv,
        "pairwise_corr_mean": pairwise_corr_mean,
        "pairwise_corr_std": pairwise_corr_std,
    }

    # ---------------- figure scaffold ----------------

    unit_height = 0.95
    trace_units = 3
    raster_main_units = 4
    active_fraction_units = 1
    stats_units = 3.5

    fig_height = unit_height * (
        trace_units + raster_main_units + active_fraction_units + stats_units
    )

    fig = plt.figure(figsize=(12, fig_height))
    fig_left, fig_right, fig_top, fig_bottom = 0.075, 0.975, 0.935, 0.06

    outer_gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[
            trace_units,
            raster_main_units + active_fraction_units,
            stats_units,
        ],
        left=fig_left,
        right=fig_right,
        top=fig_top,
        bottom=fig_bottom,
        hspace=0.30,
    )

    raster_row_gs = outer_gs[1].subgridspec(
        2,
        1,
        height_ratios=[raster_main_units, active_fraction_units],
        hspace=0.05,
    )
    raster_ax = fig.add_subplot(raster_row_gs[0, 0])
    active_ax = fig.add_subplot(raster_row_gs[1, 0], sharex=raster_ax)

    # ---------------- raster ----------------

    vmax = max(float(np.max(activity)), eps)
    raster_ax.imshow(
        activity.T,
        cmap=matplotlib.colormaps["binary"],
        norm=PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax),
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    raster_ax.set_ylabel("Neuron")
    plt.setp(raster_ax.get_xticklabels(), visible=False)

    # ---------------- active fraction ----------------

    active_ax.plot(active_fraction, color="#666666", lw=1.0)
    active_ax.set_xlim(0, plot_num_steps - 1)
    active_ax.set_ylim(0.0, max(float(active_fraction.max()) * 1.15, 0.1))
    active_ax.set_ylabel("% Active")
    active_ax.set_xlabel("Time Bin", labelpad=-8)
    active_ax.text(
        0.995,
        0.95,
        f"mean: {active_fraction.mean():.2f}",
        transform=active_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    # ---------------- neuron traces ----------------

    trace_neuron_indices = [
        idx for idx in trace_neuron_indices if 0 <= idx < plot_num_neurons
    ]
    if len(trace_neuron_indices) == 0:
        trace_neuron_indices = [0]

    trace_gs = outer_gs[0].subgridspec(
        len(trace_neuron_indices),
        1,
        hspace=0.2,
    )

    trace_axes = []
    trace_signals = []
    for neuron_idx in trace_neuron_indices:
        signal = neuron_outputs[:, neuron_idx] + neuron_biases[neuron_idx]
        trace_signals.append(signal)

    max_abs_trace = max(float(np.max(np.abs(signal))) for signal in trace_signals)
    max_abs_trace = max(max_abs_trace, eps)

    first_trace_ax = None
    for i, (neuron_idx, signal) in enumerate(zip(trace_neuron_indices, trace_signals)):
        if first_trace_ax is None:
            curr_ax = fig.add_subplot(trace_gs[i, 0], sharex=raster_ax)
            first_trace_ax = curr_ax
        else:
            curr_ax = fig.add_subplot(
                trace_gs[i, 0],
                sharex=first_trace_ax,
                sharey=first_trace_ax,
            )

        trace_axes.append(curr_ax)

        _plot_neuron_trace(
            curr_ax,
            signal,
            neuron_idx=neuron_idx,
            add_legend=(i == 0),
        )

        curr_ax.set_xlim(0, plot_num_steps - 1)
        curr_ax.set_ylim(-1.05 * max_abs_trace, 1.05 * max_abs_trace)
        curr_ax.margins(x=0.0)

        if i < len(trace_neuron_indices) - 1:
            curr_ax.tick_params(bottom=False, labelbottom=False)
        else:
            curr_ax.set_xlabel("Time Bin", labelpad=-8)

        if i == len(trace_neuron_indices) // 2:
            curr_ax.set_ylabel("Value")

    # ---------------- statistics row ----------------

    stats_gs = outer_gs[2].subgridspec(1, 3, wspace=0.32)
    ax_activity = fig.add_subplot(stats_gs[0, 0])
    ax_isi = fig.add_subplot(stats_gs[0, 1])
    ax_corr = fig.add_subplot(stats_gs[0, 2])

    # activity histogram
    activity_levels_percent = 100.0 * activity_levels
    activity_mean_percent = 100.0 * stats["activity_mean"]
    activity_std_percent = 100.0 * stats["activity_std"]

    ax_activity.hist(activity_levels_percent, bins=60)
    ax_activity.set_xlabel("Neuron % Active")
    ax_activity.set_ylabel("Count")
    ax_activity.text(
        0.95,
        0.95,
        "\n".join(
            [
                f"mean: {activity_mean_percent:.2f}",
                f"std:  {activity_std_percent:.2f}",
                f"silent: {stats['num_silent_neurons']}",
            ]
        ),
        transform=ax_activity.transAxes,
        ha="right",
        va="top",
    )

    # ISI distribution
    ax_isi.plot(isi_unique, isi_prob)
    ax_isi.set_yscale("log")
    ax_isi.set_xlim(0, max_interval)
    ax_isi.set_xlabel("Inter Spike Interval (ISI)")
    ax_isi.set_ylabel("Prob (Log)")
    ax_isi.text(
        0.95,
        0.95,
        "\n".join(
            [
                f"mean: {stats['isis_mean']:.2f}",
                f"std:  {stats['isis_std']:.2f}",
                f"cv:   {stats['isis_cv']:.2f}",
                f"fano: {stats['fano_factor']:.2f}",
            ]
        ),
        transform=ax_isi.transAxes,
        ha="right",
        va="top",
    )

    # pairwise-correlation histogram
    if len(pairwise_corr) > 0:
        ax_corr.hist(pairwise_corr, bins=60)
        ax_corr.axvline(0.0, color="grey", lw=0.5, alpha=0.4)
        ax_corr.set_xlabel(r"Pairwise Corr. $C_{ij}$")
        ax_corr.set_ylabel("Count")
        ax_corr.text(
            0.95,
            0.95,
            f"mean: {stats['pairwise_corr_mean']:.3f}\n"
            f"std:  {stats['pairwise_corr_std']:.3f}",
            transform=ax_corr.transAxes,
            ha="right",
            va="top",
        )

        def k_formatter(y, _pos):
            if y == 0:
                return "0"
            if abs(y) >= 1000:
                return f"{y / 1000:.0f}k"
            return f"{int(y)}"

        ax_corr.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    else:
        ax_corr.text(
            0.5,
            0.5,
            "no valid pairs",
            transform=ax_corr.transAxes,
            ha="center",
            va="center",
        )
        ax_corr.set_xlabel(r"Pairwise Corr. $C_{ij}$")
        ax_corr.set_ylabel("Count")

    # ---------------- row headings ----------------

    heading_gap = 0.010
    heading_fontsize = 13
    content_x = 0.5 * (fig_left + fig_right)

    row_traces_top = max(ax.get_position().y1 for ax in trace_axes)
    row_raster_top = raster_ax.get_position().y1
    row_stats_top = max(ax.get_position().y1 for ax in [ax_activity, ax_isi, ax_corr])

    fig.text(
        content_x,
        row_traces_top + heading_gap,
        "Neuron Activity",
        ha="center",
        va="bottom",
        fontsize=heading_fontsize,
    )
    fig.text(
        content_x,
        row_raster_top + heading_gap,
        "Network Activity",
        ha="center",
        va="bottom",
        fontsize=heading_fontsize,
    )
    fig.text(
        content_x,
        row_stats_top + heading_gap,
        "Network Statistics",
        ha="center",
        va="bottom",
        fontsize=heading_fontsize,
    )

    # ---------------- panel labels ----------------

    def place_panel_label(ax, label):
        pos = ax.get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.012,
            label,
            fontsize=16,
            color="dimgrey",
            va="bottom",
            ha="right",
        )

    place_panel_label(trace_axes[0], "a)")
    place_panel_label(raster_ax, "b)")
    place_panel_label(ax_activity, "c)")
    place_panel_label(ax_isi, "d)")
    place_panel_label(ax_corr, "e)")

    # ---------------- save ----------------

    if path is None:
        print(json.dumps(stats, indent=2))
        plt.show()
    else:
        plt.savefig(
            path + ".pdf",
            format="pdf",
            bbox_inches="tight",
            metadata={"CreationDate": None},
        )
        with open(path + ".json", "w") as f:
            json.dump(stats, f, indent=2)
        plt.close(fig)

    return stats
