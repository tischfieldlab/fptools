import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import TrimSignals


def test_trim_signals_begin_end():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = TrimSignals(["signal1", "signal2"], begin=10, end=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 980, "Signal should have 980 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 990, "Signal should end at 990 seconds"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 980, "Signal should have 980 samples"
    assert session.signals["signal2"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal2"].time[-1] == 990, "Signal should end at 990 seconds"

def test_trim_signals_begin():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = TrimSignals(["signal1", "signal2"], begin=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 1000, "Signal should end at 1000 seconds"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal2"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal2"].time[-1] == 1000, "Signal should end at 1000 seconds"


def test_trim_signals_begin_auto():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))
    # set the Fi1i scalar
    session.scalars["Fi1i"] = np.array([10])

    step = TrimSignals(["signal1", "signal2"], begin="auto")
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 1000, "Signal should end at 1000 seconds"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal2"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal2"].time[-1] == 1000, "Signal should end at 1000 seconds"


def test_trim_signals_end():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=1))

    step = TrimSignals(["signal1", "signal2"], end=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 1, "Signal should start at 1 seconds"
    assert session.signals["signal1"].time[-1] == 990, "Signal should end at 990 seconds"

    assert session.signals["signal2"].nobs == 2, "Signal should have 2 observations"
    assert session.signals["signal2"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal2"].time[0] == 1, "Signal should start at 1 seconds"
    assert session.signals["signal2"].time[-1] == 990, "Signal should end at 990 seconds"
