import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Zscore


def test_zscore_no_baseline_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = Zscore(["signal1", "signal2"], mode="zscore", baseline=None)
    step(session)

    # test plotting
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    # assert some things about signal1, a 1D signal
    assert session.signals["signal1"].nobs == 1, "Signal1 should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal1 should have 100 samples"
    assert np.isclose(session.signals["signal1"].signal.sum(), 0), "Signal1 sum should be about zero"

    # assert some things about signal2, a 2D signal
    assert session.signals["signal2"].nobs == 2, "Signal2 should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal2 should have 980 samples"
    assert np.isclose(session.signals["signal2"].signal.sum(), 0), "Signal2 sum should be about zero"


def test_zscore_with_baseline_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = Zscore(["signal1", "signal2"], mode="zscore", baseline=(1, 3))
    step(session)

    # test plotting
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    # assert some things about signal1, a 1D signal
    assert session.signals["signal1"].nobs == 1, "Signal1 should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal1 should have 100 samples"
    assert np.isclose(session.signals["signal1"].signal.sum(), -1000.0306841393035), "Signal1 sum should be a specific value (empirically determined)"

    # assert some things about signal2, a 2D signal
    assert session.signals["signal2"].nobs == 2, "Signal2 should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal2 should have 980 samples"
    assert np.isclose(session.signals["signal2"].signal[0].sum(), -1000.0306841393035), "Signal2 sum should be a specific value (empirically determined)"


def test_modified_zscore_no_baseline_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = Zscore(["signal1", "signal2"], mode="modified_zscore", baseline=None)
    step(session)

    # test plotting
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    # assert some things about signal1, a 1D signal
    assert session.signals["signal1"].nobs == 1, "Signal1 should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal1 should have 100 samples"
    assert np.isclose(session.signals["signal1"].signal.sum(), 0.0020581360526907844), "Signal1 sum should be a specific value (empirically determined)"

    # assert some things about signal2, a 2D signal
    assert session.signals["signal2"].nobs == 2, "Signal2 should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal2 should have 980 samples"
    assert np.isclose(session.signals["signal2"].signal[0].sum(), 0.0020581360526907844), "Signal2 sum should be a specific value (empirically determined)"


def test_modified_zscore_with_baseline_signals():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = Zscore(["signal1", "signal2"], mode="modified_zscore", baseline=(1, 3))
    step(session)

    # test plotting
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    # assert some things about signal1, a 1D signal
    assert session.signals["signal1"].nobs == 1, "Signal1 should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal1 should have 100 samples"
    assert np.isclose(session.signals["signal1"].signal.sum(), -674.5206964519602), "Signal1 sum should be a specific value (empirically determined)"

    # assert some things about signal2, a 2D signal
    assert session.signals["signal2"].nobs == 2, "Signal2 should have 2 observations"
    assert session.signals["signal2"].nsamples == 1000, "Signal2 should have 980 samples"
    assert np.isclose(session.signals["signal2"].signal[0].sum(), -674.5206964519602), "Signal2 sum should be a specific value (empirically determined)"
