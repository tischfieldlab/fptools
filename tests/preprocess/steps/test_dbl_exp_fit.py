import numpy as np
import matplotlib.pyplot as plt

from fptools.io import Session, Signal
from fptools.preprocess.steps import DblExpFit
from fptools.preprocess.lib import double_exponential, fs2t

def _gen_data(sig_name: str, ndim: int = 1):
    t = fs2t(1, 1000)
    # these were somewhat empirically determined parameters
    signal = double_exponential(t, 1.66410966e-04, 7.75225962e+00, 1.59735799e+02, 1.23719300e+04, 1.33545428e-02)
    if ndim == 2:
        signal = np.vstack([signal, signal])
    return Signal(sig_name, signal, time=t)


def test_dbl_exp_fit_no_apply():
    session = Session()
    # add a 1D signal
    session.add_signal(_gen_data("signal1", ndim=1))
    # add a 2D signal
    session.add_signal(_gen_data("signal2", ndim=2))

    step = DblExpFit(["signal1", "signal2"], apply=False)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal should have 1000 samples"
    assert "signal1_dxpfit" in session.signals, "Signal should have a double exponential fit signal"
    assert session.signals["signal1_dxpfit"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1_dxpfit"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.array_equal(session.signals["signal1"].time, session.signals["signal1_dxpfit"].time), "Time should be the same"
    assert np.allclose(session.signals["signal1"].signal, session.signals["signal1_dxpfit"].signal), "Signal should be the same"


    assert session.signals["signal2"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2"].nsamples == 1000, "Signal should have 1000 samples"
    assert "signal2_dxpfit" in session.signals, "Signal should have a double exponential fit signal"
    assert session.signals["signal2_dxpfit"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2_dxpfit"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.array_equal(session.signals["signal2"].time, session.signals["signal2_dxpfit"].time), "Time should be the same"
    assert np.allclose(session.signals["signal2"].signal, session.signals["signal2_dxpfit"].signal), "Signal should be the same"



def test_dbl_exp_fit_with_apply():
    session = Session()
    # add a 1D signal
    session.add_signal(_gen_data("signal1", ndim=1))
    # add a 2D signal
    session.add_signal(_gen_data("signal2", ndim=2))

    step = DblExpFit(["signal1", "signal2"], apply=True)
    step(session)
    fig, ax = plt.subplots(1,1)
    step.plot(session, ax)
    plt.close(fig)

    assert session.signals["signal1"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1"].nsamples == 1000, "Signal should have 1000 samples"
    assert "signal1_dxpfit" in session.signals, "Signal should have a double exponential fit signal"
    assert session.signals["signal1_dxpfit"].nobs == 1, "Signal should have 1 observation"
    assert session.signals["signal1_dxpfit"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal1"].signal.sum(), 0, atol=1e-5), "detrended signal sum should be close to 0"

    assert session.signals["signal2"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2"].nsamples == 1000, "Signal should have 1000 samples"
    assert "signal2_dxpfit" in session.signals, "Signal should have a double exponential fit signal"
    assert session.signals["signal2_dxpfit"].nobs == 2, "Signal should have 1 observation"
    assert session.signals["signal2_dxpfit"].nsamples == 1000, "Signal should have 1000 samples"
    assert np.isclose(session.signals["signal2"].signal[0].sum(), 0, atol=1e-5), "detrended signal sum should be close to 0"

