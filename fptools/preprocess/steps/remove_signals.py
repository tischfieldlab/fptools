from typing import Optional
from matplotlib.axes import Axes

from fptools.io import Session
from ..common import ProcessorThatPlots, SignalList


class RemoveSignals(ProcessorThatPlots):
    """A `Preprocessor` that allows you to rename things."""

    def __init__(self, signals: SignalList):
        """Initialize this preprocessor.

        Args:
            signals: signals to be removed
        """
        self.signals = signals

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for signame in self.signals:
            session.remove_signal(signame)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the computed dF/F signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        headers = ["Signal Name"]
        rows = []
        for signame in self.signals:
            rows.append([signame])

        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="upper center")
        # set header cells to have bold text
        for row, col in [(0, i) for i in range(len(headers))]:
            table_cell = table[row, col]
            table_cell.set_text_props(fontweight="bold")

        ax.axis("off")  # turn off axis, not needed since this is a table
        ax.set_title("Removed Signals")
