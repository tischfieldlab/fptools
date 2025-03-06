from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..lib import lowpass_filter
from ..common import PreprocessorStep, SignalList


class Lowpass(PreprocessorStep):
    """A `Preprocessor` that generates a lowpass filtered signal."""

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
        for signame in self.signals:
            sig = session.signals[signame]

            lowpass_sig = sig.copy(f"{signame}_lowpass")
            lowpass_sig.signal = lowpass_filter(sig.signal, sig.fs, self.frequency)
            session.add_signal(lowpass_sig)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the lowpass filtered signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[f"{signame}_lowpass"]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i], linestyle="--")
        ax.set_title("Lowpass Filtered Signal")
        ax.legend()
