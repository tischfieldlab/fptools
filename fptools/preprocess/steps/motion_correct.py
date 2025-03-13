from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from fptools.preprocess.lib import estimate_motion
from ..common import ProcessorThatPlots, PairedSignalList


class MotionCorrect(ProcessorThatPlots):
    """A `Processor` that estimates and corrects for motion artifacts."""

    def __init__(self, signals: PairedSignalList):
        """Initialize this processor.

        Args:
            signals: list of signal names to be downsampled
        """
        self.signals = signals

    def __call__(self, session: Session) -> Session:
        """Effect this processing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the processing step applied
        """
        for sig1, sig2 in self.signals:
            exp = session.signals[sig1]
            ctr = session.signals[sig2]

            # create a signal to hold the motion estimate data
            est_motion = exp.copy(f"{exp.name}_motion_est")
            est_motion.units = "AU"
            session.add_signal(est_motion)

            # calculate the motion estimate and corrected signal
            exp.signal, est_motion.signal = estimate_motion(exp.signal, ctr.signal)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this processing step. Will show the motion estimate and the corrected signal.

        Args:
            session: the session being operated upon
            ax: matplotlib Axes for plotting onto
        """
        palette = sns.color_palette("colorblind", n_colors=len(self.signals))
        for i, (signame, _) in enumerate(self.signals):
            sig = session.signals[signame]
            mot = session.signals[f"{signame}_motion_est"]
            ax.plot(sig.time, sig.signal, label=sig.name, c=palette[i], linestyle="-")
            ax.plot(mot.time, mot.signal, label=mot.name, c=sns.desaturate(palette[i], 0.3), linestyle="-")
        ax.set_title("Motion Estimation and Correction")
        ax.legend(loc="upper left")
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
