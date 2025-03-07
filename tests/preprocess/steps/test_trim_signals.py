import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import TrimSignals


def test_trim_signals_begin_end():
    session = Session()
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))

    step = TrimSignals(["signal1"], begin=10, end=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nsamples == 980, "Signal should have 980 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 990, "Signal should end at 990 seconds"

def test_trim_signals_begin():
    session = Session()
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))

    step = TrimSignals(["signal1"], begin=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 1000, "Signal should end at 1000 seconds"


def test_trim_signals_begin_auto():
    session = Session()
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))
    session.scalars["Fi1i"] = np.array([10])

    step = TrimSignals(["signal1"], begin="auto")
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 11, "Signal should start at 11 seconds"
    assert session.signals["signal1"].time[-1] == 1000, "Signal should end at 1000 seconds"


def test_trim_signals_end():
    session = Session()
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=1))

    step = TrimSignals(["signal1"], end=10)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nsamples == 990, "Signal should have 990 samples"
    assert session.signals["signal1"].time[0] == 1, "Signal should start at 1 seconds"
    assert session.signals["signal1"].time[-1] == 990, "Signal should end at 990 seconds"