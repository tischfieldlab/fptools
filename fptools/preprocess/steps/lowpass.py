from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from fptools.viz import plot_signal
from ..lib import lowpass_filter
from ..common import ProcessorThatPlots, SignalList


class Lowpass(ProcessorThatPlots):
    """A `Processor` that generates a lowpass filtered signal."""

    def __init__(self, signals: SignalList, frequency: float = 0.01):
        """Initialize this processor.

        Args:
            signals: list of signal names to be downsampled
            frequency: critical frequency used for lowpass filter
        """
        self.signals = signals
        self.frequency = frequency

    def __call__(self, session: Session) -> Session:
        """Effect this processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the processing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            lowpass_sig = sig.copy(f"{signame}_lowpass")
            lowpass_sig.signal = lowpass_filter(sig.signal, sig.fs, self.frequency)
            session.add_signal(lowpass_sig)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this processing step. Will show the lowpass filtered signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[f"{signame}_lowpass"]
            plot_signal(sig, ax=ax, show_indv=True, color=palette[i], indv_c=palette[i], agg_kwargs={"label": sig.name})
        ax.set_title("Lowpass Filtered Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
