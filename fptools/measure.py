from typing import Literal, Union, cast
from fptools.io import SessionCollection
import scipy
import numpy as np
from sklearn import metrics
import pandas as pd

from fptools.viz.signal_collector import collect_signals


FieldList = Union[Literal["all"], list[str]]


def measure_snr_overall(sessions: SessionCollection, signals: Union[str, list[str]], include_meta: FieldList = "all") -> pd.DataFrame:
    """Measure Signal to Noise ratio (SNR) of the overall stream.

    SNR defined as log10(mean(signal)^2 / std(signal)^2), and expressed in decibels (dB)

    Args:
        sessions: sessions to pull signals from
        signals: one or more signal names to operate on"
        include_meta: metadata fields to include in the resulting dataframe. If "all", include all metadata fields

    Returns:
        pandas.DataFrame with calculated SNR
    """
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
            data.append(
                {
                    **meta,
                    "signal": sig_name,
                    "snr": np.log10(np.power(np.mean(sig.signal), 2) / np.power(np.std(sig.signal), 2)),
                }
            )
    return pd.DataFrame(data)


def measure_snr_event(
    sessions: SessionCollection,
    signals: Union[str, list[str]],
    events: Union[str, list[str]],
    noise_range: Union[tuple[float, float], list[tuple[float, float]]],
    signal_range: Union[tuple[float, float], list[tuple[float, float]]],
    include_meta: FieldList = "all",
) -> pd.DataFrame:
    """Measure Signal to Noise ratio (SNR) in signals surrounding events.

    Args:
        signals: one or more signals to measure
        events: one or more events at which to measure signals
        noise_range: tuple(s) of start/stop times relative to the event to consider as noise
        signal_range: tuple(s) of start/stop times relative to the event to consider as signal
        include_meta: metadata fields to include in output. if "all" then all fields will be included

    Returns:
        pandas.DataFrame with collected SNR data
    """
    sigs_to_measure = []
    if isinstance(signals, str):
        sigs_to_measure.append(signals)
    else:
        sigs_to_measure.extend(signals)

    events_to_measure = []
    if isinstance(events, str):
        events_to_measure.append(events)
    else:
        events_to_measure.extend(events)

    nrs: list[tuple[float, float]] = []
    if len(np.array(noise_range).shape) == 1:
        nrs = [cast(tuple, noise_range)] * len(events_to_measure)
    else:
        nrs.extend(cast(list, noise_range))

    srs: list[tuple[float, float]] = []
    if len(np.array(signal_range).shape) == 1:
        srs = [cast(tuple, signal_range)] * len(events_to_measure)
    else:
        srs.extend(cast(list, signal_range))

    data = []
    for session in sessions:
        # determine metadata fileds to include
        if include_meta == "all":
            meta = session.metadata
        else:
            meta = {k: v for k, v in session.metadata.items() if k in include_meta}

        for sig_name in sigs_to_measure:
            for ei, event_name in enumerate(events_to_measure):
                n = collect_signals(session, event_name, sig_name, start=nrs[ei][0], stop=nrs[ei][1])
                s = collect_signals(session, event_name, sig_name, start=srs[ei][0], stop=srs[ei][1])
                data.append(
                    {
                        **meta,
                        "signal": sig_name,
                        "snr": np.median((s.signal.max(axis=1) - s.signal.min(axis=1)) ** 2 / n.signal.std(axis=1) ** 2),
                    }
                )
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
