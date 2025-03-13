from typing import Literal, Optional
from matplotlib.axes import Axes
import numpy as np
import seaborn as sns

from fptools.io import Session
from fptools.viz import plot_signal
from ..lib import mad
from ..common import ProcessorThatPlots, SignalList


class Zscore(ProcessorThatPlots):
    """A `Processor` that calculates signal z-scores."""

    def __init__(
        self, signals: SignalList, mode: Literal["zscore", "modified_zscore"] = "zscore", baseline: Optional[tuple[float, float]] = None
    ):
        """Initialize this Processor.

        Args:
            signals: list of signal names to be downsampled
            mode: the type of z-score to calculate, either "zscore", which uses traditional z-scoring, or "modified_zscore", which uses the median absolute deviation
            baseline: the time window to use for the baseline calculation, in seconds. If None, the entire signal is used.
        """
        self.signals = signals
        self.mode = mode
        self.baseline = baseline

    def __call__(self, session: Session) -> Session:
        """Effect this Processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            if self.baseline is not None:
                baseline_data = sig.signal[..., sig.tindex(self.baseline[0]) : sig.tindex(self.baseline[1])]
            else:
                baseline_data = sig.signal

            if self.mode == "zscore":
                mean = baseline_data.mean(axis=-1, keepdims=True)
                std = baseline_data.std(axis=-1, keepdims=True)
                sig.signal = (sig.signal - mean) / std
                sig.units = "Z-score"

            elif self.mode == "modified_zscore":
                median = np.median(baseline_data, axis=-1, keepdims=True)
                mad_val = mad(baseline_data)
                sig.signal = (0.6745 * (sig.signal - median)) / mad_val
                sig.units = "mZ-score"

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this processing step. Will show the computed zscore signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[signame]
            plot_signal(sig, ax=ax, show_indv=True, color=palette[i], indv_c=palette[i], agg_kwargs={"label": sig.name})
        ax.set_title("Calculated z-scored Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
