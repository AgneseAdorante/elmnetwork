from typing import Optional

import jax
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def visualize_network_inference(
    inputs,
    labels,
    logits,
    recordings,
    path: Optional[str] = None,
    bin_size: Optional[float] = None,
    adding: bool = False,
    max_viz_samples: int = 8,
):
    batch_size = min(len(inputs), len(labels), max_viz_samples)

    # Recordings contain all network layers. The last one is the output layer,
    # which is already visualized from logits, so only plot previous layers as hidden.
    hidden_recordings = recordings[:-1]
    num_hidden_layers = len(hidden_recordings)
    total_subplots = num_hidden_layers + 2

    num_rows, num_cols = (batch_size + 1) // 2, 2

    fig = plt.figure(figsize=(16, 4 * num_rows))
    outer_grid = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    for i in range(batch_size):
        x = inputs[i]
        y = int(np.asarray(labels[i]).item())

        inner_grid = gridspec.GridSpecFromSubplotSpec(
            total_subplots,
            1,
            subplot_spec=outer_grid[i],
            height_ratios=[0.5] + [1] * (total_subplots - 1),
        )

        curr_axes = []

        # ---------------- output ----------------
        ax = fig.add_subplot(inner_grid[0])
        curr_axes.append(ax)

        output_data = np.asarray(jax.nn.softmax(logits[i], axis=-1))
        y_pred = int(np.argmax(output_data[-1]))

        ax.imshow(
            output_data.T,
            cmap="binary",
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(f"Label vs. Prediction: {y} vs. {y_pred}")
        ax.set_ylabel("Output")
        ax.set_xticklabels([])
        ax.get_xaxis().set_visible(False)

        # ---------------- hidden ----------------
        for j in range(num_hidden_layers - 1, -1, -1):
            ax = fig.add_subplot(inner_grid[num_hidden_layers - j])
            curr_axes.append(ax)

            ax.imshow(
                np.asarray(hidden_recordings[j][0]["activity"][i]).T,
                cmap="binary",
                origin="lower",
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_ylabel(f"Hidden {j + 1}" if num_hidden_layers > 1 else "Hidden")
            ax.set_xticklabels([])
            ax.get_xaxis().set_visible(False)

        # ---------------- input ----------------
        ax = fig.add_subplot(inner_grid[-1])
        curr_axes.append(ax)

        input_data = np.asarray(x, dtype=np.float32)
        input_vmax = max(2.0, float(np.max(input_data)))

        ax.imshow(
            input_data.T,
            cmap="binary",
            vmin=0.0,
            vmax=input_vmax,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_ylabel("Input")
        ax.set_xlabel("Time Bin")

        if bin_size is not None:
            ax.annotate(
                f"bin size = {bin_size}ms",
                xy=(0.96, 0.84),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 2,
                },
                zorder=10,
            )

        if adding:
            split_idx = input_data.shape[0] // 2
            for curr_ax in curr_axes:
                curr_ax.axvline(
                    x=split_idx,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.85,
                )

    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path + ".pdf")
        plt.close(fig)
