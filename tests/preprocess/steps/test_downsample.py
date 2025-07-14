import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Downsample


def test_downsample_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=100))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=100))

    step = Downsample(["signal1", "signal2"])
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 100, "Signal should have 100 samples"
    assert np.isclose(session.signals["signal1"].fs, 10), "Signal sampling rate should be 10"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 100, "Signal should have 980 samples"
    assert np.isclose(session.signals["signal2"].fs, 10), "Signal sampling rate should be 10"
