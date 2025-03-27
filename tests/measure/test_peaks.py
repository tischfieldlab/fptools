
import numpy as np

from fptools.io import Session, Signal
from fptools.measure import measure_peaks, detect_naive_peaks, collect_signals


def test_measure_peaks(tdt_preprocessed_sessions):
    measure_peaks(tdt_preprocessed_sessions, signal="Dopamine")

def test_measure_peaks_dynamic(tdt_preprocessed_sessions): 
    def height_filter(session: Session, signal: Signal, trial: int, trail_data: np.ndarray):
        return trail_data.mean() + (3.5 * trail_data.std())

    measure_peaks(tdt_preprocessed_sessions, signal="Dopamine", include_detection_params=True, height=height_filter)

def test_detect_naive_peaks(tdt_preprocessed_sessions):
    def collect_sigs(session:Session):
        session.add_signal(collect_signals(session, signal="Dopamine", event="RNP"))
    tdt_preprocessed_sessions.apply(collect_sigs)
    detect_naive_peaks(tdt_preprocessed_sessions, signal="Dopamine@RNP", window=(0.0, 1.0))
