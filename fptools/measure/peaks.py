import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
from ..io import SessionCollection
from ..io.session import FieldList


def measure_peaks(sessions: SessionCollection, signal: str, include_meta: FieldList = "all", **kwargs) -> pd.DataFrame:
    """Measure peaks within a signal.

    Default detection parameters are as follows:
    {"height": (None, None), "threshold": (None, None), "distance": None, "prominence": (None, None), "width": (None, None) }
    These parameters are designed to not filter any peaks, but also provoke `scipy.signal.find_peaks()` into returning all peak measurement fields.
    You may override any of these parameters to `scipy.signal.find_peaks()` via this functions `**kwargs`

    The returned dataframe will contain the following information (one row corresponds to one peak)
        - any metadata from the Session, as specified by the `include_meta` parameter
        - trial: index of the observation from the signal data that the peak was found
        - peak_num: which peak, within a given trial, for the case of multiple found peaks within a single trial
        - peak_index: the index along the signal where the peak was found
        - peak_time: this is `peak_index` converted to the time domain (i.e. relative seconds)
        The following measurements are reported by `scipy.signal.find_peaks()`:
            - peak_heights: this is the height of the peak, as returned by `scipy.signal.find_peaks()`
            - left_thresholds, right_thresholds: this is the peak vertical distance to its neighbouring samples, as returned by `scipy.signal.find_peaks()`
            - prominences: this is the prominence of the peak, as returned by `scipy.signal.find_peaks()`
            - left_bases, right_bases: The peak's bases as indices in x to the left and right of each peak, as returned by `scipy.signal.find_peaks()`. The higher base of each pair is a peak's lowest contour line.
            - widths: the width of the peak, as returned by `scipy.signal.find_peaks()`.
            - width_heights: The height of the contour lines at which the widths where evaluated, as returned by `scipy.signal.find_peaks()`..
            - left_ips, right_ips: Interpolated positions of left and right intersection points of a horizontal line at the respective evaluation height, as returned by `scipy.signal.find_peaks()`..
        - auc: area under the curve of the given trial, as returned by `sklearn.metrics.auc()`

    Args:
        sessions: collection of sessions to work on
        signal: name of the signal to measure
        include_meta: metadata fields to include in the final output. Special string "all" will include all metadata fields
        **kwargs: additional kwargs to pass to `scipy.signal.find_peaks()`

    Returns:
        pandas `DataFrame` with peak measurements.
    """
    detection_params = {
        "height": (None, None),
        "threshold": (None, None),
        "distance": None,
        "prominence": (None, None),
        "width": (None, None),
    }
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

            for peak_i in range(len(peaks)):
                peak_data.append(
                    {
                        **meta,
                        "trial": i,
                        "peak_num": peak_i,
                        "peak_index": peaks[peak_i],
                        "peak_time": t[peaks[peak_i]],
                        **{k: v[peak_i] for k, v in props.items()},
                        "auc": metrics.auc(t, sig[i, props["left_bases"][peak_i] : props["right_bases"][peak_i]]),
                    }
                )

    peak_data = pd.DataFrame(peak_data)
    return peak_data
