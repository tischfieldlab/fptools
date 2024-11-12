from matplotlib.ticker import Locator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from matplotlib.axes import Axes
from matplotlib.text import Text
import pandas as pd
import scipy
import scipy.interpolate
import seaborn as sns
from typing import Optional, Union
from matplotlib.lines import Line2D

from fptools.io import Session, Signal, SessionCollection


def plot_signal(
    signal: Signal,
    ax: Optional[Axes] = None,
    show_indv: bool = True,
    color: ColorType = "k",
    indv_c: ColorType = "b",
    indv_alpha: float = 0.1,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.DataFrame(signal.signal.T)
    df.index = signal.time
    df = df.melt(ignore_index=False)

    if show_indv and signal.nobs:
        for i in range(signal.signal.shape[0]):
            sns.lineplot(data=None, x=signal.time, y=signal.signal[i, :], alpha=indv_alpha, ax=ax, color=indv_c)

    sns.lineplot(data=df, x=df.index, y="value", ax=ax, color=color)

    ax.set_xlabel("Time, Reletive to Event (s)")
    ax.set_ylabel(f"{signal.name} ({signal.units})")

    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    if len(xticklabels) == 0:
        xticklabels = [Text(text=f"{xt}") for xt in xticks]
    for k, v in signal.marks.items():
        ax.axvline(v, c="gray", ls="--")
        try:
            xt = np.where(xticks == float(v))[0][0]
            xticklabels[xt] = Text(text=k)
        except:
            xticks = np.append(xticks, float(v))
            xticklabels.append(Text(text=k))
            order = np.argsort(xticks)
            xticks = xticks[order]
            xticklabels = [xticklabels[i] for i in order]
            pass
    ax.set_xticks(xticks, [t.get_text() for t in xticklabels])

    return ax


def sig_catplot(
    sessions: SessionCollection,
    signal: str,
    col: Union[str, None] = None,
    col_order: Union[list[str], None] = None,
    row: Union[str, None] = None,
    row_order: Union[list[str], None] = None,
    palette=None,
    hue: Union[str, None] = None,
    hue_order: Union[list[str], None] = None,
    show_indv: bool = False,
    indv_alpha: float = 0.1,
):

    metadata = sessions.metadata

    if col is not None:
        if col_order is None:
            if pd.api.types.is_categorical_dtype(metadata[col]):
                plot_cols = metadata[col].cat.categories.values
            else:
                plot_cols = sorted(metadata[col].unique())
        else:
            avail_cols = list(metadata[col].unique())
            plot_cols = [c for c in col_order if c in avail_cols]
    else:
        plot_cols = [None]

    if row is not None:
        if row_order is None:
            if pd.api.types.is_categorical_dtype(metadata[row]):
                plot_rows = metadata[row].cat.categories.values
            else:
                plot_rows = sorted(metadata[row].unique())
        else:
            avail_rows = list(metadata[row].unique())
            plot_rows = [r for r in row_order if r in avail_rows]
    else:
        plot_rows = [None]

    if hue is not None and hue_order is None:
        if pd.api.types.is_categorical_dtype(metadata[hue]):
            hue_order = metadata[hue].cat.categories.values
        else:
            hue_order = sorted(metadata[hue].unique())

    if hue_order is not None and palette is None:
        palette = sns.color_palette("colorblind", n_colors=len(hue_order))

    fig, axs = plt.subplots(
        len(plot_rows), len(plot_cols), figsize=(len(plot_cols) * 6, len(plot_rows) * 6), sharey=True, sharex=True, squeeze=False
    )

    for row_i, cur_row in enumerate(plot_rows):
        if cur_row is not None:
            row_criteria = metadata[row] == cur_row
            row_title = f"{row} = {cur_row}"
        else:
            row_criteria = np.ones(len(metadata.index), dtype=bool)
            row_title = None

        for col_i, cur_col in enumerate(plot_cols):
            if cur_col is not None:
                col_criteria = metadata[col] == cur_col
                col_title = f"{col} = {cur_col}"
            else:
                col_criteria = np.ones(len(metadata.index), dtype=bool)
                col_title = None

            ax = axs[row_i, col_i]

            title = f"{signal}"
            if col_title is not None or row_title is not None:
                title += " at " + " & ".join([t for t in [col_title, row_title] if t is not None])
            ax.set_title(title)

            if hue is None:
                try:
                    sig = sessions.select(row_criteria, col_criteria).aggregate_signals(signal)
                    plot_signal(sig, ax=ax, show_indv=show_indv, color=palette[0], indv_c=palette[0], indv_alpha=indv_alpha)
                except:
                    pass

            elif hue_order is not None:
                legend_items = []
                legend_labels = []
                for hi, curr_hue in enumerate(hue_order):
                    try:
                        sess_subset = sessions.select(row_criteria, col_criteria, metadata[hue] == curr_hue)
                        if len(sess_subset) > 0:
                            sig = sess_subset.aggregate_signals(signal)
                            plot_signal(sig, ax=ax, show_indv=show_indv, color=palette[hi], indv_c=palette[hi])

                        legend_items.append(Line2D([0], [0], color=palette[hi]))
                        legend_labels.append(f"{curr_hue}, n={len(sess_subset)}")

                    except:
                        raise

                ax.legend(legend_items, legend_labels, loc="upper right")
                # sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))


def plot_heatmap(signal: Signal, ax=None, cmap="viridis", vmin=None, vmax=None):

    if ax is None:
        fig, ax = plt.subplots()

    cbar_kwargs = {"label": f"{signal.name} ({signal.units})"}

    sns.heatmap(data=np.atleast_2d(signal.signal), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws=cbar_kwargs)

    xticks = [0, signal.nsamples]
    xticklabels = [f"{signal.time[0]:0.0f}", f"{signal.time[-1]:0.0f}"]
    for m, t in signal.marks.items():
        i = signal.tindex(t)
        ax.axvline(i, c="w", ls="--")
        xticks.append(i)
        xticklabels.append(f"{m}")
    order = np.argsort(xticks)
    xticks = [xticks[i] for i in order]
    xticklabels = [xticklabels[i] for i in order]
    ax.set_xticks(xticks, labels=xticklabels, rotation=0)

    ax.set_xlabel("Time, Reletive to Event (sec)")

    return ax
