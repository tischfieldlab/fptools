from typing import Literal, Union
from fptools.io import SessionCollection
import scipy
import numpy as np
from sklearn import metrics
import pandas as pd

from fptools.viz.signal_collector import collect_signals


FieldList = Union[Literal["all"], list[str]]


def measure_snr_overall(sessions: SessionCollection, signals: Union[str, list[str]], include_meta: FieldList = "all") -> pd.DataFrame:

    sigs_to_measure = []
    if isinstance(signals, str):
        sigs_to_measure.append(signals)
    else:
        sigs_to_measure.extend(signals)

    data = []
    for session in sessions:
        # determine metadata fileds to include
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        for sig_name in sigs_to_measure:
            sig = session.signals[sig_name]
            data.append({
                **meta,
                'signal': sig_name,
                'snr': np.mean(sig.signal)**2 / np.std(sig.signal)**2,
            })
    return pd.DataFrame(data)


def measure_snr_event(sessions: SessionCollection, signals: Union[str, list[str]], noise_range: tuple[float, float], signal_range: tuple[float, float], event: str, include_meta: FieldList = "all") -> pd.DataFrame:
    
    sigs_to_measure = []
    if isinstance(signals, str):
        sigs_to_measure.append(signals)
    else:
        sigs_to_measure.extend(signals)

    data = []
    for session in sessions:
        # determine metadata fileds to include
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        for sig_name in ['Dopamine', 'Isosbestic']:
            n = collect_signals(session, event, sig_name, pre=noise_range[0], post=noise_range[1])
            s = collect_signals(session, event, sig_name, pre=signal_range[0], post=signal_range[1])
            data.append({
                **meta,
                'signal': sig_name,
                'snr': np.median((s.signal.max(axis=1) - s.signal.min(axis=1))**2 / n.signal.std(axis=1)**2)
            })
    return pd.DataFrame(data)


def measure_peaks(sessions: SessionCollection, signal: str, include_meta: FieldList = "all", **kwargs) -> pd.DataFrame:
    """Measure peaks within a signal.

    Args:
        sessions: collection of sessions to work on
        signal: name of the signal to measure
        include_meta: metadata fields to include in the final output. Special string "all" will include all metadata fields
        **kwargs: additional kwargs to pass to `scipy.signal.find_peaks()`

    Returns:
        pandas `DataFrame` with peak measurements.
    """
    detection_params = {"prominence": (None, None), "distance": 10000, "height": (None, None)}
    detection_params.update(**kwargs)
    peak_data = []
    for session in sessions:
        # determine metadata fileds to include
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        t = session.signals[signal].time
        sig = np.atleast_2d(session.signals[signal].signal)
        for i in range(sig.shape[0]):
            peaks, props = scipy.signal.find_peaks(sig[i, :], **detection_params)
            if len(peaks) > 1:
                print("found more than one peak!!")

            peak_data.append(
                {
                    **meta,
                    "trial": i,
                    "peak_i": peaks[0],
                    "peak_t": t[peaks[0]],
                    "height": props["peak_heights"][0],
                    "prominence": props["prominences"][0],
                    "auc": np.abs(metrics.auc(t, sig[i])),
                }
            )

    peak_data = pd.DataFrame(peak_data)
    return peak_data
