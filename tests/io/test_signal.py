import datetime
import numpy as np
from pytest import approx
from fptools.io import Signal


def test_signal():
    sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)

    sig.describe()
    sig.to_dataframe()

    assert sig.nobs == 1, "Signal should only have one observation"
    assert sig.nsamples == 1000, "Signal should have 1,000 samples"
    assert sig.duration.total_seconds() == approx(999), "duration should be 999 seconds"
    assert sig.tindex(100) == 99

def test_signal_time_reciprocal():
    sig1 = Signal("sig1", np.sin(np.arange(1000)), fs=1)
    sig2 = Signal("sig2", np.sin(np.arange(1000)), time=sig1.time)

    assert sig1._check_other_compatible(sig2)
