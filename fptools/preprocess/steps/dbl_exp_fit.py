from typing import Literal, Union

from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..lib import detrend_double_exponential
from ..common import ProcessorThatPlots, SignalList


class DblExpFit(ProcessorThatPlots):
    """A `Preprocessor` that fits a double exponential function."""

    def __init__(self, signals: SignalList, apply: bool = True):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be fitted
            apply: if True, detrend the signal using the double exponential fit. If False, only calculate the fit.
        """
        self.signals = signals
        self.apply = apply

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            # create a new signal to hold the double exponential fit
            dxp_sig = sig.copy(f"{signame}_dxpfit")
            session.add_signal(dxp_sig)

            # calculate the fit and the detrended signal
            detrended, dxp_sig.signal = detrend_double_exponential(sig.time, sig.signal)

            # if the user requested, apply detrending to the signal
            if self.apply:
                sig.signal = detrended

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the double exponential fit.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[f"{signame}_dxp"]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i], linestyle="--")

            if self.apply:
                sig = session.signals[signame]
                ax.plot(sig.time, sig.signal, label=f"detrended {sig.name}", c=palette[i], linestyle="-")
        ax.set_title("Double Exponential Fit")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
