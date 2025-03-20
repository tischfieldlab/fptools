from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from fptools.viz import plot_signal
from ..lib import detrend_double_exponential
from ..common import ProcessorThatPlots, SignalList


class DblExpFit(ProcessorThatPlots):
    """A `Processor` that fits a double exponential function."""

    def __init__(self, signals: SignalList, apply: bool = True):
        """Initialize this processor.

        Args:
            signals: list of signal names to be fitted
            apply: if True, detrend the signal using the double exponential fit. If False, only calculate the fit.
        """
        self.signals = signals
        self.apply = apply

    def __call__(self, session: Session) -> Session:
        """Effect this processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the processing step applied
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
        """Plot the effects of this processing step. Will show the double exponential fit.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[f"{signame}_dxpfit"]
            plot_signal(sig, ax=ax, show_indv=True, color=palette[i], indv_c=palette[i], agg_kwargs={"label": sig.name})

            if self.apply:
                sig = session.signals[signame]
                plot_signal(
                    sig,
                    ax=ax,
                    show_indv=True,
                    color=sns.desaturate(palette[i], 0.3),
                    indv_c=sns.desaturate(palette[i], 0.3),
                    agg_kwargs={"label": f"detrended {sig.name}"},
                )

        ax.set_title("Double Exponential Fit")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
