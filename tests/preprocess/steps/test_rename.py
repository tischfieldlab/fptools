import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import Rename


def test_rename():
    session = Session()
    # add a 1D signal
    session.add_signal(Signal("signal1", np.sin(np.arange(1000)), fs=100))
    # add a 2D signal
    session.add_signal(Signal("signal2", np.vstack([np.sin(np.arange(1000)), np.sin(np.arange(1000))]), fs=100))
    session.epocs["epoc1"] = np.array([1, 2])
    session.scalars["scalar1"] = np.array([1, 2])

    step = Rename(signals={"signal1": "signal1_renamed", "signal2": "signal2_renamed"}, epocs={"epoc1": "epoc1_renamed"}, scalars={"scalar1": "scalar1_renamed"})
    step(session)

    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert len(session.signals) == 2, "Session should have two signals"
    assert "signal1_renamed" in session.signals, "Session should have a signal named 'signal1_renamed'"
    assert "signal2_renamed" in session.signals, "Session should have a signal named 'signal2_renamed'"
    assert "signal1" not in session.epocs, "Session should not have a signal named 'signal1'"
    assert "signal2" not in session.epocs, "Session should not have a signal named 'signal2'"

    assert len(session.epocs) == 1, "Session should have one epoc"
    assert "epoc1_renamed" in session.epocs, "Session should have an epoc named 'epoc1_renamed'"
    assert "epoc1" not in session.epocs, "Session should not have an epoc named 'epoc1'"

    assert len(session.scalars) == 1, "Session should have one scalar"
    assert "scalar1_renamed" in session.scalars, "Session should have a scalar named 'scalar1_renamed'"
    assert "scalar1" not in session.scalars, "Session should not have a scalar named 'scalar1'"




