from typing import Optional
from matplotlib.axes import Axes

from fptools.io import Session
from ..common import ProcessorThatPlots


class Rename(ProcessorThatPlots):
    """A `Processor` that allows you to rename things."""

    def __init__(
        self, signals: Optional[dict[str, str]] = None, epocs: Optional[dict[str, str]] = None, scalars: Optional[dict[str, str]] = None
    ):
        """Initialize this processor.

        Args:
            signals: dictionary of signal names to be renamed
            epocs: dictionary of epoc names to be renamed
            scalars: dictionary of scalar names to be renamed
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
            for k, v in self.signals.items():
                session.rename_signal(k, v)

        if self.epocs is not None:
            for k, v in self.epocs.items():
                session.rename_epoc(k, v)

        if self.scalars is not None:
            for k, v in self.scalars.items():
                session.rename_scalar(k, v)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this processing step. Will show the computed dF/F signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        headers = ["Type", "Original", "New"]
        rows = []
        if self.signals is not None:
            for k, v in self.signals.items():
                rows.append(["signal", k, v])

        if self.epocs is not None:
            for k, v in self.epocs.items():
                rows.append(["epoc", k, v])

        if self.scalars is not None:
            for k, v in self.scalars.items():
                rows.append(["scalar", k, v])

        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="upper center")
        # set header cells to have bold text
        for row, col in [(0, i) for i in range(len(headers))]:
            table_cell = table[row, col]
            table_cell.set_text_props(fontweight="bold")

        ax.axis("off")  # turn off axis, not needed since this is a table
        ax.set_title("Renamed Items")
