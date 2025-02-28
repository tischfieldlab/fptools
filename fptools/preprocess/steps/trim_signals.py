from typing import Literal, Union

from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..lib import trim
from ..common import PreprocessorStep, SignalList


class TrimSignals(PreprocessorStep):
    """A `Preprocessor` that trims signals."""

    def __init__(self, signals: SignalList, extent: Union[Literal["auto"], float, tuple[float, float]] = "auto"):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be trimmed
            extent: specification for trimming. "auto" uses the offset stored in `scalars['Fi1i']`, a single float trims that amount of time (in seconds) from the beginning, a tuple of two floats specifies the amount of time (in seconds) from the beginning and end to trim, respectively.
        """
        self.signals = signals
        self.extent = extent

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self._resolve_signal_names(session, self.signals):
            sig = session.signals[signame]
            if self.extent == "auto":
                trim_args = {"begin": int(session.scalars["Fi1i"][0] * sig.fs)}
            elif isinstance(self.extent, float):
                trim_args = {"begin": int(self.extent * sig.fs)}
            elif len(self.extent) == 2:
                trim_args = {"begin": int(self.extent[0] * sig.fs), "end": int(self.extent[1] * sig.fs)}

            sig.signal, sig.time = trim(sig.signal, sig.time, **trim_args)
        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show trimmed signals.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        signals = self._resolve_signal_names(session, self.signals)
        palette = sns.color_palette("colorblind", n_colors=len(signals))
        for i, signame in enumerate(signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i])
        ax.set_title("Trimmed Signal")
        ax.legend()
