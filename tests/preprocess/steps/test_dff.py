import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Dff


def test_dff_with_center():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    session.add_signal(Signal("signal1_ctrl", np.cos(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))
    session.add_signal(Signal("signal2_ctrl", np.vstack([np.cos(np.arange(1000)), np.cos(np.arange(1000))]), fs=1))

    step = Dff([("signal1", "signal1_ctrl"), ("signal2", "signal2_ctrl")], center=True)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal should have 1000 samples"
    assert session.signals["signal1"].units == "ΔF/F", "Signal should have units ΔF/F"
    assert np.isclose(session.signals["signal1"].signal[0], -100), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal1"].signal[500], -47.075613525551994), "middle samples should be an empirical value"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal should have 1000 samples"
    assert session.signals["signal2"].units == "ΔF/F", "Signal should have units ΔF/F"
    assert np.isclose(session.signals["signal2"].signal[0, 0], -100), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal2"].signal[0, 500], -47.075613525551994), "middle samples should be an empirical value"


def test_dff_no_center():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    session.add_signal(Signal("signal1_ctrl", np.cos(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))
    session.add_signal(Signal("signal2_ctrl", np.vstack([np.cos(np.arange(1000)), np.cos(np.arange(1000))]), fs=1))

    step = Dff([("signal1", "signal1_ctrl"), ("signal2", "signal2_ctrl")], center=False)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal should have 1000 samples"
    assert session.signals["signal1"].units == "ΔF/F", "Signal should have units ΔF/F"
    assert np.isclose(session.signals["signal1"].signal[0], 0), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal1"].signal[500], 52.924386474448006), "middle samples should be an empirical value"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal should have 1000 samples"
    assert session.signals["signal2"].units == "ΔF/F", "Signal should have units ΔF/F"
    assert np.isclose(session.signals["signal2"].signal[0, 0], 0), "first sample should be an empirical value"
    assert np.isclose(session.signals["signal2"].signal[0, 500], 52.924386474448006), "middle samples should be an empirical value"