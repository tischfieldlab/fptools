
import numpy as np

from fptools.io import Session, Signal
from fptools.measure import measure_peaks


def test_measure_peaks(tdt_preprocessed_sessions):
    measure_peaks(tdt_preprocessed_sessions, signal="Dopamine")

def test_measure_peaks_dynamic(tdt_preprocessed_sessions): 
    def height_filter(session: Session, signal: Signal, trial: int, trail_data: np.ndarray):
        return trail_data.mean() + (3.5 * trail_data.std())

    measure_peaks(tdt_preprocessed_sessions, signal="Dopamine", include_detection_params=True, height=height_filter)
