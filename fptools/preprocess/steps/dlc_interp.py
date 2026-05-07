from matplotlib.axes import Axes
from fptools.io.session import Session
from fptools.preprocess.common import ProcessorThatPlots


class DLCInterpolation(ProcessorThatPlots):
    """A `Processor` that interpolates missing frames in dlc data."""

    def __init__(
        self,
    ):
        """Initialize this Processor."""
        pass

    def __call__(self, session: Session) -> Session:
        pass

    def plot(self, session: Session, ax: Axes):
        pass
