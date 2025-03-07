import datetime
import numpy as np
from pytest import approx
import pytest
from fptools.io import Signal


def test_signal():
    sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)
    sig.marks['mark1'] = 1.0
    sig.marks['mark2'] = 2.0

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

def test_signal_math():
    sin_sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)
    const_sig = Signal("sig2", np.ones(1000)*4, fs=1)

    # test Signal-scalar addition, subtraction, multiplication, division
    assert sin_sig == ((sin_sig + 1) - 1)
    assert sin_sig == ((sin_sig * 2) / 2)

    # test Signal-Signal addition, subtraction, multiplication, division
    assert sin_sig == ((sin_sig + sin_sig) - sin_sig)
    assert const_sig == ((const_sig * const_sig) / const_sig)


def test_signal_time_init():

    # test that time and fs mismatch raises ValueError
    with pytest.raises(ValueError):
        Signal("sig1", np.sin(np.arange(1000)), time=np.arange(1001), fs=0.5)

    # test that time and fs both being None raises ValueError
    with pytest.raises(ValueError):
        Signal("sig1", np.sin(np.arange(1000)), time=None, fs=None)

    # test that time signal length mismatch raises ValueError
    with pytest.raises(ValueError):
        Signal("sig1", np.sin(np.arange(1000)), time=np.arange(500), fs=None)