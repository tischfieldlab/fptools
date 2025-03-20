from typing import Literal, Optional
from matplotlib.axes import Axes
import numpy as np
import seaborn as sns

from fptools.io import Session
from fptools.viz import plot_signal
from ..lib import mad, zscore, madscore, modified_zscore
from ..common import ProcessorThatPlots, SignalList


class Zscore(ProcessorThatPlots):
    """A `Processor` that calculates signal z-scores."""

    def __init__(
        self,
        signals: SignalList,
        mode: Literal["zscore", "madscore", "modified_zscore"] = "zscore",
        baseline: Optional[tuple[float, float]] = None,
        outlier_removal: Optional[Literal["zscore", "madscore", "modified_zscore"]] = None,
        outlier_threshold: Optional[float] = None,
    ):
        """Initialize this Processor.

        Args:
            signals: list of signal names to be downsampled
            mode: the type of z-score to calculate, either "zscore" (which uses traditional z-scoring), "madscore" (which uses median-absolute-deviation scores) or "modified_zscore" (which uses scaled median absolute deviation scores)
            baseline: the time window to use for the baseline calculation, in seconds. If None, the entire signal is used.
            outlier_removal: If not None, method to use for outlier removal within the baseline period (if specified) or otherwise the entire signal. Options are "zscore", "madscore" or "modified_zscore". If None, no outlier removal is performed.
            outlier_threshold: threshold for outlier removal. If the outlier removal method is "zscore", this is the number of standard deviations from the mean to consider an outlier. If the method is "madscore", this is the number of median absolute deviations from the median to consider an outlier. If the method is "modified_zsocre", this is the number of scaled median absolute deviations from the median to consider an outlier.
        """
        self.signals = signals
        self.mode = mode
        self.baseline = baseline
        self.outlier_removal = outlier_removal
        self.outlier_threshold = outlier_threshold

    def __call__(self, session: Session) -> Session:
        """Effect this Processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            # subset data to the baseline window
            if self.baseline is not None:
                baseline_data = sig.signal[..., sig.tindex(self.baseline[0]) : sig.tindex(self.baseline[1])]

            else:
                baseline_data = sig.signal

            # perform outlier removal if requested on the baseline data
            if self.outlier_removal is not None:
                if self.outlier_threshold is None:
                    raise ValueError("If outlier_removal is specified, outlier_threshold must also be specified")

                if self.outlier_removal == "zscore":
                    baseline_data = baseline_data[np.abs(zscore(baseline_data)) < self.outlier_threshold]

                elif self.outlier_removal == "madscore":
                    baseline_data = baseline_data[np.abs(madscore(baseline_data)) < self.outlier_threshold]

                elif self.outlier_removal == "modified_zscore":
                    baseline_data = baseline_data[np.abs(modified_zscore(baseline_data)) < self.outlier_threshold]

                else:
                    raise ValueError("Invalid outlier_removal method")

            # calculate the final scores
            if self.mode == "zscore":
                sig.signal = zscore(
                    sig.signal, mu=baseline_data.mean(axis=-1, keepdims=True), sigma=baseline_data.std(axis=-1, keepdims=True)
                )
                sig.units = "Z-score"

            elif self.mode == "madscore":
                sig.signal = madscore(sig.signal, mu=np.median(baseline_data, axis=-1, keepdims=True), sigma=mad(baseline_data))
                sig.units = "MAD-score"

            elif self.mode == "modified_zscore":
                sig.signal = modified_zscore(sig.signal, mu=np.median(baseline_data, axis=-1, keepdims=True), sigma=mad(baseline_data))
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
        ax.set_title(f"Calculated {self.mode} Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
