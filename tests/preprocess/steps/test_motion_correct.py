import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import MotionCorrect


def test_motion_correct():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    session.add_signal(Signal("signal1_ctrl", np.cos(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))
    session.add_signal(Signal("signal2_ctrl", np.vstack([np.cos(np.arange(1000)), np.cos(np.arange(1000))]), fs=1))

    step = MotionCorrect([("signal1", "signal1_ctrl"), ("signal2", "signal2_ctrl")])
    step(session)

    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert "signal1" in session.signals, "original signal should be in session"
    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal1"].signal[0], 3.883586795555539e-05), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal1"].signal[500], -0.46778185775401077), "middle samples should be an empirical value"

    assert "signal1_motion_est" in session.signals, "Motion estimate signal should be in session"
    assert session.signals["signal1_motion_est"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1_motion_est"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal1_motion_est"].signal[0], -3.883586795555539e-05), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal1_motion_est"].signal[500], 1.0052431534627584e-05), "middle samples should be an empirical value"


    assert "signal2" in session.signals, "original signal should be in session"
    assert session.signals["signal2"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal2"].signal[0, 0], 3.883586795555539e-05), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal2"].signal[0, 500], -0.46778185775401077), "middle samples should be an empirical value"

    assert "signal2_motion_est" in session.signals, "Motion estimate signal should be in session"
    assert session.signals["signal2_motion_est"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2_motion_est"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal2_motion_est"].signal[0, 0], -3.883586795555539e-05), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal2_motion_est"].signal[0, 500], 1.0052431534627584e-05), "middle samples should be an empirical value"