from typing import Protocol, Union, cast, runtime_checkable
import numpy as np
import pandas as pd
import scipy
from sklearn import metrics
from ..io import SessionCollection, Session, Signal
from ..io.session import FieldList

PeakFilter = Union[None, float, tuple[Union[None, float], Union[None, float]]]


@runtime_checkable
class PeakFilterProvider(Protocol):
    def __call__(self, session: Session, signal: Signal, trial: int, trail_data: np.ndarray) -> PeakFilter:
        """Peak Filter Provider protocol.

        A Peak Filter Provider allows one to dynamically provide filter values for `scipy.signal.find_peaks()`.

        Args:
            session: the current `Session` being operated upon
            signal: the current `Signal` being operated upon
            trial: the current index of signal
            trail_data: array of signal values for the current trial

        Returns:
            The return value should be something that `scipy.signal.find_peaks()` can use for the property you wish to filter.
            Typically, this means one of the following: `None`, a scalar float, or a tuple containing some combination of None or scalar floats
        """
        pass


def measure_peaks(
    sessions: SessionCollection, signal: str, include_meta: FieldList = "all", include_detection_params: bool = False, **kwargs
) -> pd.DataFrame:
    """Measure peaks within a signal.

    Default detection parameters are as follows:
    {"height": (None, None), "threshold": (None, None), "distance": None, "prominence": (None, None), "width": (None, None) }

    These parameters are designed to not filter any peaks, but also provoke `scipy.signal.find_peaks()` into returning all peak measurement fields.

    You may override any of these parameters to `scipy.signal.find_peaks()` via this functions `**kwargs`.

    Also allowed for any valid detection parameter is a callable (see `PeakFilterProvider` for signature) that is given the session, signal, trial index and trial
    data, and should return valid detection parameter values (i.e. None, float, or tuple of the preceeding). The callable is evaluated for each observation in the
    signal in a given session.

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
        include_detection_params: if True, include detection parameters as columns in the output dataframe
        **kwargs: additional kwargs to pass to `scipy.signal.find_peaks()`

    Returns:
        pandas `DataFrame` with peak measurements.
    """
    # these serve as the base detection params. Designed to evoke all the extra results from `find_peaks()`
    # without actually filtering anything. These can be overridden by the user via kwargs
    detection_params: dict[str, Union[PeakFilter, PeakFilterProvider]] = {
        "height": (None, None),
        "threshold": (None, None),
        "distance": None,
        "prominence": (None, None),
        "width": (None, None),
    }
    # update the detection params with any the user provided.
    detection_params.update(**kwargs)

    peak_data = []
    for session in sessions:
        # determine metadata fileds to include
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        # fetch time and signal data
        t = session.signals[signal].time
        sig = np.atleast_2d(session.signals[signal].signal)
        for i in range(sig.shape[0]):
            # setup detection params for the current trial.
            # the user could possibly provide us a callable to dynamically determine detection parameter values
            current_detection_params = detection_params.copy()
            for key in current_detection_params.keys():
                param = current_detection_params[key]
                if isinstance(param, PeakFilterProvider):
                    current_detection_params[key] = param(session, session.signals[signal], i, sig[i, :])

            # detect peaks
            peaks, props = scipy.signal.find_peaks(sig[i, :], **current_detection_params)

            # for each peak, prepare a "row" for the output dataframe
            for peak_i in range(len(peaks)):
                peak_slice = slice(props["left_bases"][peak_i], props["right_bases"][peak_i])
                result = {
                    **meta,
                    "trial": i,
                    "peak_num": peak_i,
                    "peak_index": peaks[peak_i],
                    "peak_time": t[peaks[peak_i]],
                    **{k: v[peak_i] for k, v in props.items()},
                    "auc": metrics.auc(t[peak_slice], sig[i, peak_slice]),
                }

                # include any detection params if the user requested that feature
                if include_detection_params:
                    result.update({f"param_{k}": v for k, v in current_detection_params.items()})

                # collect the result
                peak_data.append(result)

    # convert results to a dataframe and return
    peak_data = pd.DataFrame(peak_data)
    return peak_data
