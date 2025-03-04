from typing import Literal, Union

from matplotlib.axes import Axes
import seaborn as sns
from scipy import stats

from fptools.io import Session, Signal
from ..lib import lowpass_filter
from ..common import PreprocessorStep, SignalList


class Dff(PreprocessorStep):
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
        for signame in self._resolve_signal_names(session, self.signals):
            sig = session.signals[signame]

            sig.signal = stats.zscore(sig.signal)
            sig.units = "AU"

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the computed zscore signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        signals = self._resolve_signal_names(session, self.signals)
        palette = sns.color_palette("colorblind", n_colors=len(signals))
        for i, signame in enumerate(signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=f"corrected {sig.name}", c=palette[i], linestyle="-")
        ax.set_title("Calculated zscore Signal")
        ax.legend()
