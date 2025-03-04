from abc import ABC, abstractmethod
import os
from typing import Literal, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from fptools.io import Session


SignalList = Union[Literal["all"], list["str"]]
PairedSignalList = list[tuple[str, str]]


class Preprocessor(ABC):
    """Abstract Preprocessor.

    Implementors should implement the `__call__` method.
    """

    @abstractmethod
    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        raise NotImplementedError()


class PreprocessorStep(Preprocessor):
    """Abstract Preprocessor.

    Implementors should implement the `__call__` and `plot` methods.
    """

    @abstractmethod
    def plot(self, session: Session, ax: Axes) -> None:
        """Plot the effects of this preprocessing step.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        raise NotImplementedError()

    def _resolve_signal_names(self, session: Session, signals: SignalList) -> list[str]:
        """Resolve signal names, including special monikers, i.e. "all"."""
        if signals == "all":
            return list(session.signals.keys())
        else:
            return [s for s in signals if s in session.signals.keys()]


class Pipeline(Preprocessor):
    def __init__(self, steps: Optional[list[Preprocessor]] = None):
        """Initialize this pipeline.

        Args:
            steps: list of preprocessors to run on a given Session
        """
        self.steps: list[Preprocessor]
        if steps is None:
            self.steps = []
        else:
            self.steps = steps
        self.plot: bool = True
        self.plot_dir: str = "."

    def __call__(self, session: Session) -> Session:
        """Run this pipeline on a Session.

        Args:
            session: the session to operate on

        Returns:
            Session with Preprocessors applied
        """
        try:
            if self.plot:
                # allocate a plot:
                # - each step receives one row to plot on
                fig, axs = plt.subplots(len(self.steps), 1, figsize=(24, 6 * len(self.steps)))

            for i, step in enumerate(self.steps):
                session = step(session)
                if self.plot and hasattr(step, "plot"):
                    step.plot(session, axs[i])

            return session
        except:
            raise
        finally:
            if self.plot:
                fig.savefig(os.path.join(self.plot_dir, f"{session.name}.png"), dpi=600)
                fig.savefig(os.path.join(self.plot_dir, f"{session.name}.pdf"))
                plt.close(fig)
