from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..lib import downsample, t2fs
from ..common import ProcessorThatPlots, SignalList


class Downsample(ProcessorThatPlots):
    """A `Preprocessor` that downsamples signals."""

    def __init__(self, signals: SignalList, window: int = 10, factor: int = 10):
        """Initialize this preprocessor.

        Downsampling performs a moving average convolution, with window size `window`, and then samples every `factor` steps.

        Args:
            signals: list of signal names to be downsampled
            window: size of the window for the moving average calculation
            factor: step size for taking samples
        """
        self.signals = signals
        self.window = window
        self.factor = factor

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            sig.signal, sig.time = downsample(sig.signal, sig.time, window=self.window, factor=self.factor)
            sig.fs = t2fs(sig.time)
        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of the preprocessing step. Will show the downsampled signals.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i])
        ax.set_title("Downsampled Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
