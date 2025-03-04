from typing import Literal, Union

from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session, Signal
from ..lib import detrend_double_exponential, lowpass_filter
from ..common import PreprocessorStep, SignalList


class DblExpFit(PreprocessorStep):
    """A `Preprocessor` that fits a double exponential function."""

    def __init__(self, signals: SignalList, frequency: float = 0.01):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be downsampled
            frequency: critical frequency used for lowpass filter
        """
        self.signals = signals
        self.frequency = frequency

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self._resolve_signal_names(session, self.signals):
            sig = session.signals[signame]

            dxp_sig = sig.copy(f"{signame}_dxp")
            _, dxp_sig.signal = detrend_double_exponential(sig.time, sig.signal)
            session.add_signal(dxp_sig)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the double exponential fit.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        signals = self._resolve_signal_names(session, self.signals)
        palette = sns.color_palette("colorblind", n_colors=len(signals))
        for i, signame in enumerate(signals):
            sig = session.signals[f"{signame}_dxp"]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i], linestyle="--")
        ax.set_title("Double Exponential Fit")
        ax.legend()
