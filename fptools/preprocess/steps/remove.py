from typing import Optional
from matplotlib.axes import Axes

from fptools.io import Session
from ..common import ProcessorThatPlots


class Remove(ProcessorThatPlots):
    """A `Processor` that allows you to remove things."""

    def __init__(self, signals: Optional[list[str]] = None, epocs: Optional[list[str]] = None, scalars: Optional[list[str]] = None):
        """Initialize this processor.

        Args:
            signals: list of signal names to be removed
            epocs: epoc names to be removed
            scalars: scalar names to be removed
        """
        self.signals = signals
        self.epocs = epocs
        self.scalars = scalars

    def __call__(self, session: Session) -> Session:
        """Effect this processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the processing step applied
        """
        if self.signals is not None:
            for signame in self.signals:
                session.remove_signal(signame)

        if self.epocs is not None:
            for epocname in self.epocs:
                session.epocs.pop(epocname)

        if self.scalars is not None:
            for scalarname in self.scalars:
                session.scalars.pop(scalarname)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this processing step. Will show the computed dF/F signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        headers = ["Type", "Name"]
        rows = []
        if self.signals is not None:
            for v in self.signals:
                rows.append(["signal", v])

        if self.epocs is not None:
            for v in self.epocs:
                rows.append(["epoc", v])

        if self.scalars is not None:
            for v in self.scalars:
                rows.append(["scalar", v])

        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="upper center")
        # set header cells to have bold text
        for row, col in [(0, i) for i in range(len(headers))]:
            table_cell = table[row, col]
            table_cell.set_text_props(fontweight="bold")

        ax.axis("off")  # turn off axis, not needed since this is a table
        ax.set_title("Removed Signals")
