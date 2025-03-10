from typing import Literal, Union

from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from ..lib import trim
from ..common import ProcessorThatPlots, SignalList


class TrimSignals(ProcessorThatPlots):
    """A `Preprocessor` that trims signals."""

    def __init__(self, signals: SignalList, begin: Union[None, Literal["auto"], int, float] = None, end: Union[None, int, float] = None):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be trimmed
            begin: if not None, trim that amount of time (in seconds) from the beginning of the signal. If "auto", use the offset stored in `block.scalars.Fi1i.ts` for trimming
            end: if not None, trim that amount of time (in seconds) from the end of the signal.
        """
        self.signals = signals
        self.begin = begin
        self.end = end

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            sig = session.signals[signame]

            begin: Union[int, None]
            if self.begin == "auto":
                begin = int(session.scalars["Fi1i"][0] * sig.fs)
            elif isinstance(self.begin, (float, int)):
                begin = int(self.begin * sig.fs)
            elif self.begin is None:
                begin = None
            else:
                raise ValueError(f"Invalid value for begin: {self.begin}")

            end: Union[int, None]
            if isinstance(self.end, (float, int)):
                end = int(self.end * sig.fs)
            elif self.end is None:
                end = None
            else:
                raise ValueError(f"Invalid value for end: {self.end}")

            sig.signal, sig.time = trim(sig.signal, sig.time, begin=begin, end=end)
        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show trimmed signals.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, signame in enumerate(self.signals):
            sig = session.signals[signame]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i])
        ax.set_title("Trimmed Signal")
        ax.legend()
