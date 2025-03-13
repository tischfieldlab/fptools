import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Remove


def test_remove():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=100))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=100))
    session.epocs["epoc1"] = np.array([1, 2])
    session.scalars["scalar1"] = np.array([1, 2])

    step = Remove(signals=["signal1", "signal2"], epocs=["epoc1"], scalars=["scalar1"])
    step(session)

    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert len(session.signals) == 0, "Session should have two signals"
    assert len(session.epocs) == 0, "Session should have one epoc"
    assert len(session.scalars) == 0, "Session should have one scalar"




