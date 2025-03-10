from abc import ABC, abstractmethod
import os
from typing import Literal, Optional, Protocol, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from fptools.io import Session


SignalList = list["str"]
PairedSignalList = list[tuple[str, str]]


def _flatten_paired_signals(signals: PairedSignalList) -> list[str]:
    return [s for pair in signals for s in pair]


def _remap_signals(signals: SignalList, rename_map: dict[str, str]) -> list[str]:
    return [rename_map.get(s, s) for s in signals]


def _remap_paired_signals(signals: PairedSignalList, rename_map: dict[str, str]) -> PairedSignalList:
    return [(rename_map.get(s1, s1), rename_map.get(s2, s2)) for s1, s2 in signals]


class Processor(ABC):
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


class ProcessorThatPlots(Processor):
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


class Pipeline(Processor):
    """A pipeline of Preprocessors."""

    def __init__(self, steps: Optional[list[Processor]] = None, plot: bool = True, plot_dir: Optional[str] = None):
        """Initialize this pipeline.

        Args:
            steps: list of preprocessors to run on a given Session
            plot: whether to plot the results of each step
            plot_dir: directory to save plots to, if None, will save to current working directory
        """
        self.steps: list[Processor]
        if steps is None:
            self.steps = []
        else:
            self.steps = steps

        self.plot: bool = plot
        self.plot_dir: str = plot_dir or os.getcwd()

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
                if self.plot:
                    if hasattr(step, "plot"):
                        step.plot(session, axs[i])
                    else:
                        if hasattr(step, "__class__"):
                            step_name = step.__class__.__name__  # this handles the case where the step is a class
                        elif hasattr(step, "__name__"):
                            step_name = step.__name__  # this handles the case where the step is a function
                        elif hasattr(step, "func"):
                            step_name = step.func.__name__  # this handles the case where the step is a partial
                        else:
                            step_name = "Unknown"  # this handles the case where we cannot figure out a reasonable name for the step

                        message = f'Step #{i+1} "{step_name}" has no plot method, skipping plotting for this step.'
                        axs[i].text(0.5, 0.5, message, ha="center", va="center", transform=axs[i].transAxes)
                        axs[i].axis("off")

            return session
        except:
            raise
        finally:
            if self.plot:
                fig.savefig(os.path.join(self.plot_dir, f"{session.name}.png"), dpi=300)
                fig.savefig(os.path.join(self.plot_dir, f"{session.name}.pdf"))
                plt.close(fig)
