from collections.abc import Mapping
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Literal, Optional, Union

from fptools.io import Signal, SessionCollection


def plot_signal(
    signal: Signal,
    ax: Optional[Axes] = None,
    show_indv: bool = True,
    color: ColorType = "k",
    indv_c: ColorType = "b",
    indv_alpha: float = 0.1,
    indv_kwargs: Optional[dict] = None,
    agg_kwargs: Optional[dict] = None,
) -> Axes:
    """Plot a signal as a lineplot.

    Args:
        signal: the signal to be plotted
        ax: optional axes to plot on. If not provided, a new figure with a single axes will be created
        show_indv: if True, plot individual traces, otherwise only plot aggregate traces
        color: color of the aggregated trace
        indv_c: color of individual traces
        indv_alpha: alpha transparency for individual traces
        indv_kwargs: kwargs to pass to `seaborn.lineplot()` for individual traces
        agg_kwargs: kwarge to pass to `seaborn.lineplot()` for aggregate traces

    Returns:
        Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    df = pd.DataFrame(signal.signal.T)
    df.index = signal.time
    df = df.melt(ignore_index=False)

    _indv_kwargs = {
        "alpha": indv_alpha,
        "color": indv_c,
    }
    if indv_kwargs is not None:
        _indv_kwargs.update(indv_kwargs)

    if show_indv and signal.nobs > 1:
        for i in range(signal.signal.shape[0]):
            sns.lineplot(data=None, x=signal.time, y=signal.signal[i, :], ax=ax, **_indv_kwargs)

    _agg_kwargs = {"color": color}
    if agg_kwargs is not None:
        _agg_kwargs.update(agg_kwargs)

    sns.lineplot(data=df, x=df.index, y="value", ax=ax, **_agg_kwargs)

    ax.set_xlabel("Time, Reletive to Event (s)")
    ax.set_ylabel(f"{signal.name} ({signal.units})")

    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()
    if len(xticklabels) == 0:
        xticklabels = [Text(text=f"{xt}") for xt in xticks]
    for k, v in signal.marks.items():
        # only annotate marks if they are within the time domain
        if v >= signal.time.min() and v <= signal.time.max():
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
    signal: Union[str, list[str]],
    col: Union[Literal["signal"], str, None] = None,
    col_order: Union[list[str], None] = None,
    row: Union[Literal["signal"], str, None] = None,
    row_order: Union[list[str], None] = None,
    palette: Optional[Union[str, list, dict[Any, str]]] = None,
    hue: Union[str, None] = None,
    hue_order: Union[list[str], None] = None,
    show_indv: bool = False,
    indv_alpha: float = 0.1,
    height: float = 6,
    aspect: float = 1.5,
    sharex: bool = True,
    sharey: bool = True,
    agg_method: str = "mean",
    indv_kwargs: Optional[dict] = None,
    agg_kwargs: Optional[dict] = None,
) -> tuple[Figure, np.ndarray]:
    """Plot signals, similar to `seaborn.catplot()`.

    You may provide more than one signal name to the `signal` parameter. In this case, you must also specify one
    facet (e.x. `col`, `row`, `hue`) as the string "signal".

    Args:
        sessions: sessions to plot data from
        signal: name of the signal(s) to be plotted.
        col: metadata column on which to form plot columns, or if multiple signal names are given to `signal`, you may specify "signal" here
        col_order: explicit ordering for columns
        row: metadata column on which to form plot rows, or if multiple signal names are given to `signal`, you may specify "signal" here
        row_order: explicit ordering for rows
        palette: palette to use for hue mapping. A dict[value, color], or something that sns.color_palette() understands
        hue: metadata column on which to group and color, or if multiple signal names are given to `signal`, you may specify "signal" here
        hue_order: explicit ordering for hues
        show_indv: if True, show individual session traces
        indv_alpha: alpha transparency level for individual session traces, in range (0, 1)
        height: height of each facet
        aspect: Aspect ratio of each facet, so that aspect * height gives the width of each facet
        sharex: If true, the facets will share x axes.
        sharey: If true, the facets will share y axes.
        agg_method: method to use for aggregation (see `SessionCollection.aggregate_signals()` for more details)

    Returns:
        Figure and array of axes
    """
    metadata = sessions.metadata

    _signals: list[str] = []
    if isinstance(signal, str):
        _signals = [signal]
    else:
        _signals = list(signal)

    if len(_signals) > 1:
        # multiple signals provided, need to check a few things...
        # 1) at least one facet should be "signal"
        # 2) at most one facet should be "signal"
        num_sig_facets = [(fname, fval) for fname, fval in [("col", col), ("row", row), ("hue", hue)] if fval == "signal"]
        if len(num_sig_facets) <= 0:
            raise ValueError('When providing multiple values to parameter `signal`, at least one facet must be set to "signal"!')
        if len(num_sig_facets) > 1:
            raise ValueError('When providing multiple values to parameter `signal`, at most one facet may be set to "signal"!')

    plot_cols: list[Any]
    if col is not None:
        if col == "signal":
            plot_cols = [s for s in _signals]
        elif col_order is None:
            if pd.api.types.is_categorical_dtype(metadata[col]):
                plot_cols = list(metadata[col].cat.categories.values)
            else:
                plot_cols = sorted(metadata[col].unique())
        else:
            avail_cols = list(metadata[col].unique())
            plot_cols = [c for c in col_order if c in avail_cols]
    else:
        plot_cols = [None]

    plot_rows: list[Any]
    if row is not None:
        if row == "signal":
            plot_rows = [s for s in _signals]
        elif row_order is None:
            if pd.api.types.is_categorical_dtype(metadata[row]):
                plot_rows = list(metadata[row].cat.categories.values)
            else:
                plot_rows = sorted(metadata[row].unique())
        else:
            avail_rows = list(metadata[row].unique())
            plot_rows = [r for r in row_order if r in avail_rows]
    else:
        plot_rows = [None]

    if hue is not None and hue_order is None:
        if hue == "signal":
            hue_order = [s for s in _signals]
        elif pd.api.types.is_categorical_dtype(metadata[hue]):
            hue_order = metadata[hue].cat.categories.values
        else:
            hue_order = sorted(metadata[hue].unique())

    use_palette: list[str] = []
    if hue_order is not None:
        if palette is None:
            # default palette
            _palette = sns.color_palette("colorblind", n_colors=len(hue_order))
            use_palette = [_palette[i] for i in range(len(hue_order))]
        elif isinstance(palette, Mapping):
            # we got a dict-like of categories -> colors
            use_palette = [palette[item] for item in hue_order]
        else:
            # list or string, it's seaborn's problem now
            _palette = sns.color_palette(palette, n_colors=len(hue_order))
            use_palette = [_palette[i] for i in range(len(hue_order))]

    fig, axs = plt.subplots(
        len(plot_rows),
        len(plot_cols),
        figsize=(len(plot_cols) * (height * aspect), len(plot_rows) * height),
        sharey=sharey,
        sharex=sharex,
        squeeze=False,
    )

    sig_to_plot = _signals[0]
    for row_i, cur_row in enumerate(plot_rows):
        if cur_row is not None:
            if row == "signal":
                row_criteria = np.ones(len(metadata.index), dtype=bool)
                row_title = None
                sig_to_plot = cur_row
            else:
                row_criteria = metadata[row] == cur_row
                row_title = f"{row} = {cur_row}"
        else:
            row_criteria = np.ones(len(metadata.index), dtype=bool)
            row_title = None

        for col_i, cur_col in enumerate(plot_cols):
            if cur_col is not None:
                if col == "signal":
                    col_criteria = np.ones(len(metadata.index), dtype=bool)
                    col_title = None
                    sig_to_plot = cur_col
                else:
                    col_criteria = metadata[col] == cur_col
                    col_title = f"{col} = {cur_col}"
            else:
                col_criteria = np.ones(len(metadata.index), dtype=bool)
                col_title = None

            ax = axs[row_i, col_i]
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelbottom=True)

            if hue == "signal":
                title = ""
                if col_title is not None or row_title is not None:
                    title += " & ".join([t for t in [col_title, row_title] if t is not None])
                ax.set_title(title)
            else:
                title = f"{sig_to_plot}"
                if col_title is not None or row_title is not None:
                    title += " at " + " & ".join([t for t in [col_title, row_title] if t is not None])
                ax.set_title(title)

            if hue is None:
                try:
                    sig = sessions.select(row_criteria, col_criteria).aggregate_signals(sig_to_plot, method=agg_method)
                    plot_signal(
                        sig,
                        ax=ax,
                        show_indv=show_indv,
                        color=use_palette[0],
                        indv_c=use_palette[0],
                        indv_alpha=indv_alpha,
                        indv_kwargs=indv_kwargs,
                        agg_kwargs=agg_kwargs,
                    )
                except:
                    pass

            elif hue_order is not None:
                legend_items = []
                legend_labels = []
                for hi, curr_hue in enumerate(hue_order):
                    try:
                        if hue == "signal":
                            sess_subset = sessions.select(row_criteria, col_criteria)
                            sig_to_plot = curr_hue
                        else:
                            sess_subset = sessions.select(row_criteria, col_criteria, metadata[hue] == curr_hue)

                        if len(sess_subset) > 0:
                            sig = sess_subset.aggregate_signals(sig_to_plot, method=agg_method)
                            plot_signal(
                                sig,
                                ax=ax,
                                show_indv=show_indv,
                                color=use_palette[hi],
                                indv_c=use_palette[hi],
                                indv_kwargs=indv_kwargs,
                                agg_kwargs=agg_kwargs,
                            )

                        legend_items.append(Line2D([0], [0], color=use_palette[hi]))
                        legend_labels.append(f"{curr_hue}, n={len(sess_subset)}")

                    except:
                        raise

                ax.legend(legend_items, legend_labels, loc="upper right")
                # sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
    return fig, axs


def plot_heatmap(
    signal: Signal, ax: Optional[Axes] = None, cmap="viridis", vmin: Optional[float] = None, vmax: Optional[float] = None
) -> Axes:
    """Plot a signal as a heatmap.

    Args:
        signal: the signal to be plotted
        ax: optional axes to plot on. If not provided, a new figure with a single axes will be created
        cmap: colormap to use for plotting
        vmin: minimum value mapping to colormap start. If None, will use the data minimum value
        vmax: maximum value mapping to colormap end. If None, will use the data maximum value

    Returns:
        Axes
    """
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
