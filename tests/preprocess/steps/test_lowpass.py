import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Lowpass


def test_lowpass_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)+1), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)+1), np.sin(np.arange(1000)+1)]), fs=1))

    step = Lowpass(["signal1", "signal2"])
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1_lowpass"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1_lowpass"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal1_lowpass"].signal[0], 0.9941761993621171), "first sample should be about 1"
    assert np.isclose(session.signals["signal1_lowpass"].signal[500], -1.0912116567615586e-05), "middle samples should be about 0"

    assert session.signals["signal2_lowpass"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2_lowpass"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal2_lowpass"].signal[0, 0], 0.9941761993621171), "first sample should be about 1"
    assert np.isclose(session.signals["signal2_lowpass"].signal[0, 500], -1.0912116567615586e-05), "middle samples should be about 0"