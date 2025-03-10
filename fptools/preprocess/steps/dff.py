from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..common import ProcessorThatPlots, PairedSignalList


class Dff(ProcessorThatPlots):
    """A `Preprocessor` that calculates signal dF/F."""

    def __init__(self, signals: PairedSignalList, center: bool = True):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be downsampled
            center: if true, the signal is centered around the control signal, before the ratio is calculated. If False, only the ratio is calculated.
        """
        self.signals = signals
        self.center = center

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for sig1, sig2 in self.signals:
            exp = session.signals[sig1]
            ctr = session.signals[sig2]

            if self.center:
                exp.signal = (((exp - ctr) / ctr) * 100).signal
            else:
                exp.signal = ((exp / ctr) * 100).signal
            exp.units = "ΔF/F"

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the computed dF/F signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, (signame, _) in enumerate(self.signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i], linestyle="-")
        ax.set_title("Calculated dF/F Signal")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
