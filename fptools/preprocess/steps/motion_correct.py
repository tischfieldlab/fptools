from matplotlib.axes import Axes
import seaborn as sns

from fptools.io import Session
from fptools.preprocess.lib import estimate_motion
from ..common import PreprocessorStep, PairedSignalList


class MotionCorrect(PreprocessorStep):
    """A `Preprocessor` that estimates and corrects for motion artifacts."""

    def __init__(self, signals: PairedSignalList):
        """Initialize this preprocessor.

        Args:
            signals: list of signal names to be downsampled
            frequency: critical frequency used for lowpass filter
        """
        self.signals = signals

    def __call__(self, session: Session) -> Session:
        """Effect this preprocessing step.

        Args:
            session: the session to operate upon

        Returns:
            Session with the preprocessing step applied
        """
        for sig1, sig2 in self.signals:
            exp = session.signals[sig1]
            ctr = session.signals[sig2]
            est_motion = exp.copy(f"{exp.name}_motion_est")
            est_motion.units = "AU"

            exp.signal, est_motion.signal = estimate_motion(exp.signal, ctr.signal)
            session.add_signal(est_motion)

        return session

    def plot(self, session: Session, ax: Axes):
        """Plot the effects of this preprocessing step. Will show the motion estimate and the corrected signal.

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
        ax.legend()
