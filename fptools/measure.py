from typing import Literal, Union
from fptools.io import Session
import scipy
import numpy as np
from sklearn import metrics
import pandas as pd


FieldList = Union[Literal["all"], list[str]]

def measure_peaks(sessions: list[Session], signal: str, include_meta: FieldList = "all"):
    detection_params = {
        'prominence': (None, None),
        'distance': 10000,
        'height': (None,None)
    }
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
                print('found more than one peak!!')

            peak_data.append({
                **meta,
                'trial': i,
                'peak_i': peaks[0],
                'peak_t': t[peaks[0]],
                'height': props['peak_heights'][0],
                'prominence': props['prominences'][0],
                'auc': np.abs(metrics.auc(t, sig[i])),
            })

    peak_data = pd.DataFrame(peak_data)
    return peak_data
