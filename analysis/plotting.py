"""Plotting utilities for result analysis and figure export."""
import os
import numpy as np
import matplotlib.pyplot as plt


def save_figure(do_save, filename, figure_folder="Figures", fmt="svg"):
    """Save the current matplotlib figure if do_save is True."""
    if do_save:
        os.makedirs(figure_folder, exist_ok=True)
        filepath = os.path.join(figure_folder, f"{filename}.{fmt}")
        plt.savefig(filepath, format=fmt, bbox_inches='tight')
        print(f"Figure saved: {filepath}")


def plot_training_loss(train_loss, val_loss, checkpoint_epoch, title="", save=False, fig_dir="Figures"):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss, linewidth=0.5, label='Train Loss')
    val_epochs = [(i + 1) * checkpoint_epoch for i in range(len(val_loss))]
    ax.plot(val_epochs, val_loss, 'o-', markersize=3, label='Val Loss')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training Loss: {title}')
    ax.legend()
    ax.grid(True)
    save_figure(save, f"loss_{title}", fig_dir)
    plt.show()


def plot_response_comparison(t, y_pred, y_true, node_labels=None, node_colors=None,
                              resp_type="acc", title="", save=False, fig_dir="Figures"):
    """Plot predicted vs true response time histories for each node.

    Args:
        node_colors: list of colors per node (e.g. 'blue' for instrumented,
                     'green' for reconstructed). Falls back to 'b' for all.
        resp_type: 'acc' or 'dsp' — controls y-axis label and figure filename.
    """
    ylabel_map = {"acc": "Acc [m/s²]", "dsp": "Disp [m]"}
    ylabel = ylabel_map.get(resp_type, resp_type)

    n_nodes = y_pred.shape[0]
    fig, axes = plt.subplots(n_nodes, 1, figsize=(10, 3 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]
    for i in range(n_nodes):
        label = node_labels[i] if node_labels else f'Node {i + 1}'
        pred_color = node_colors[i] if node_colors else 'b'
        axes[i].plot(t, y_true[i], 'k--', linewidth=0.8, label='Reference')
        axes[i].plot(t, y_pred[i], color=pred_color, linewidth=0.5, alpha=0.7,
                     label='Predicted')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(label)
        axes[i].legend(loc='upper right')
        axes[i].grid(True)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title)
    plt.tight_layout()
    save_figure(save, f"{resp_type}_{title}", fig_dir)
    plt.show()


def plot_mode_shapes(modeshapes_pred, modeshapes_true=None,
                      title="", save=False, fig_dir="Figures"):
    """Plot predicted (and optionally true) mode shapes."""
    n_modes = modeshapes_pred.shape[0]
    n_dof = modeshapes_pred.shape[1]
    floors = list(range(n_dof + 1))

    fig, axes = plt.subplots(1, n_modes, figsize=(3 * n_modes, 6), sharey=True)
    if n_modes == 1:
        axes = [axes]
    for i in range(n_modes):
        pred = np.insert(modeshapes_pred[i], 0, 0.0)
        axes[i].plot(pred, floors, 'b-o', label='Predicted')
        if modeshapes_true is not None:
            true = np.insert(modeshapes_true[i], 0, 0.0)
            axes[i].plot(true, floors, 'k--s', label='Reference')
        axes[i].set_title(f'Mode {i + 1}')
        axes[i].set_xlabel('Mode Shape')
        axes[i].grid(True)
        axes[i].legend()
    axes[0].set_ylabel('Floor')
    fig.suptitle(title)
    plt.tight_layout()
    save_figure(save, f"modeshape_{title}", fig_dir)
    plt.show()
