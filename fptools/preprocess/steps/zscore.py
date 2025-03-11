from matplotlib.axes import Axes
import seaborn as sns
from scipy import stats

from fptools.io import Session
from ..common import ProcessorThatPlots, SignalList


class Zscore(ProcessorThatPlots):
    """A `Preprocessor` that calculates signal z-scores."""

    def __init__(self, signals: SignalList):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be downsampled
            frequency: critical frequency used for lowpass filter
        """
        self.signals = signals

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            sig.signal = stats.zscore(sig.signal)
            sig.units = "SDs"

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the computed zscore signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=f"corrected {sig.name}", c=palette[i], linestyle="-")
        ax.set_title("Calculated zscore Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
