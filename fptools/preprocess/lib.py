from typing import Optional, Union
import numpy as np
import scipy
import scipy.stats
import scipy.signal


def fs2t(fs: float, length: int) -> np.ndarray:
    """Generate a time array given a sample frequency and number of samples.

    Args:
        fs: sampling rate, in Hz
        length: number of samples to generate

    Returns
        array of time values, in seconds
    """
    return np.linspace(1, length, length) / fs


def t2fs(time: np.ndarray) -> float:
    """Estimate the sample frequency given a time array.

    Args:
        time: array of time values, in seconds, from which to estimate the sample frequency

    Returns
        The estimated sample frequency, in Hz
    """
    return 1 / np.median(np.diff(time))


def lowpass_filter(signal: np.ndarray, fs: float, Wn: float = 10) -> np.ndarray:
    """zero-phase lowpass filter a signal.

    Args:
        signal: array to be filtered
        fs: sampling frequency of the signal, in Hz
        Wn: critical frequency, see `scipy.signal.butter()`

    Returns:
        lowpass filtered signal
    """
    b, a = scipy.signal.butter(2, Wn, btype="lowpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal, axis=-1)


def double_exponential(t: np.ndarray, const: float, amp_fast: float, amp_slow: float, tau_slow: float, tau_multiplier: float) -> np.ndarray:
    """Compute a double exponential function with constant offset.

    Args:
        t: Time vector in seconds.
        const: Amplitude of the constant offset.
        amp_fast: Amplitude of the fast component.
        amp_slow: Amplitude of the slow component.
        tau_slow: Time constant of slow component in seconds.
        tau_multiplier: Time constant of fast component relative to slow.

    Returns:
        dependent values evaluated over `t` given the remaining parameters.
    """
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def fit_double_exponential(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Run the fitting procedure for a double exponential curve.

    Args:
        time: array of sample times
        signal: array of sample values

    Returns:
        array of values from the fitted double exponential curve, samples at the times in `time`.
    """
    if signal.ndim == 2:
        max_sig = np.max(signal)
        initial_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])

        parm_opt = np.zeros_like(signal, shape=(signal.shape[0], len(initial_params)))
        for i, sig_row in enumerate(signal):
            parm_opt[i], _ = scipy.optimize.curve_fit(double_exponential, time, sig_row, p0=initial_params, bounds=bounds, maxfev=1000)
        return np.array([double_exponential(time, *p) for p in parm_opt])

    elif signal.ndim == 1:
        max_sig = np.max(signal)
        initial_params = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
        parm_opt, _ = scipy.optimize.curve_fit(double_exponential, time, signal, p0=initial_params, bounds=bounds, maxfev=1000)
        return double_exponential(time, *parm_opt)

    else:
        raise ValueError("signal must be 1D or 2D")


def detrend_double_exponential(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detrend a signal by fitting and subtracting a double exponential curve to the data.

    See `double_exponential` for the underlying curve design, and `fit_double_exponential` for the curve fitting procedure.

    Args:
        time: array of sample/observation times
        signal: array of samples

    Returns:
        Tuple of (detrended_signal, signal_fit)
    """
    signal_fit = fit_double_exponential(time, signal)
    return signal - signal_fit, signal_fit


def estimate_motion(signal: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the contribution of motion artifacts in `signal`.

    Performs linear regression of control (x, independent) vs signal (y, dependent)

    Args:
        signal: the signal to correct for motion artifacts
        control: signal to use for background signal

    Returns:
        tuple of (corrected_signal, estimated_motion)
    """
    assert signal.shape == control.shape, "signal and control must have same shapes"
    if signal.ndim == 2:
        slopes = np.zeros_like(signal, shape=(signal.shape[0], 1))
        intercepts = np.zeros_like(signal, shape=(signal.shape[0], 1))
        for i, (signal_row, control_row) in enumerate(zip(signal, control)):
            slopes[i], intercepts[i], _, _, _ = scipy.stats.linregress(x=control_row, y=signal_row)

        est_motion = slopes * (control) + intercepts
        return signal - est_motion, est_motion

    elif signal.ndim == 1:
        slope, intercept, _, _, _ = scipy.stats.linregress(x=control, y=signal)
        est_motion = slope * control + intercept
        return signal - est_motion, est_motion

    else:
        raise ValueError("signal must be 1D or 2D")


def are_arrays_same_length(*arrays: np.ndarray) -> bool:
    """Check if all arrays are the same shape in the last axis."""
    lengths = [arr.shape[-1] for arr in arrays]
    return bool(np.all(np.array(lengths) == lengths[0]))


def downsample(*signals: np.ndarray, window: int = 10, factor: int = 10) -> tuple[np.ndarray, ...]:
    """Downsample one or more signals by factor across windows of size `window`.

    performs a moving window average using windows of size `window`, then takes every
    n-th observation as given by `factor`.

    Args:
        signals: one or more signals to downsample
        window: size of the window used for averaging
        factor: step size for taking the final downsampled signal

    Returns:
        downsampled signal(s)
    """
    return tuple(
        np.apply_along_axis(np.convolve, axis=-1, arr=sig, v=np.ones(window) / window, mode="valid")[..., ::factor] for sig in signals
    )


def trim(*signals: np.ndarray, begin: Optional[int] = None, end: Optional[int] = None) -> tuple[np.ndarray, ...]:
    """Trim samples from the beginning or end of a signal.

    Args:
        signals: one or more signals to be trimmed
        begin: number of samples to trim from the beginning
        end: number of samples to trim from the end

    Returns:
        tuple of trimmed signals
    """
    assert are_arrays_same_length(*signals)
    if begin is None:
        begin = 0

    if end is None:
        end = int(signals[0].shape[-1])
    else:
        end = int(signals[0].shape[-1] - end)

    assert begin < end

    return tuple(sig[..., begin:end] for sig in signals)


def zscore(data: np.ndarray, mu: Optional[Union[float, np.ndarray]] = None, sigma: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """Z-score some data.

    Z-scores are calculated as (x - mean) / std.

    Args:
        data: data to transform into z-scores
        mu: mean to use for z-scoring. If None, the mean of the data is used.
        sigma: standard deviation to use for z-scoring. If None, the standard deviation of the data is used.

    Returns:
        z-scored data
    """
    if mu is None:
        mu = data.mean(axis=-1, keepdims=True)

    if sigma is None:
        sigma = data.std(axis=-1, keepdims=True)

    return (data - mu) / sigma


def mad(data: np.ndarray) -> np.ndarray:
    """Calculate the Median Absolute Deviation (MAD) of a dataset.

    Args:
        data: A list or NumPy array of numerical data.

    Returns:
        The MAD of the data.
    """
    return np.median(np.absolute(data - np.median(data, axis=-1, keepdims=True)), axis=-1, keepdims=True)


def madscore(
    data: np.ndarray, mu: Optional[Union[float, np.ndarray]] = None, sigma: Optional[Union[float, np.ndarray]] = None
) -> np.ndarray:
    """Calculate the MAD score of a dataset.

    Args:
        data: data to transform into MAD-scores.
        mu: central value to use for MAD scoring. If None, the median of the data is used.
        sigma: scale value to use for MAD scoring. If None, the MAD of the data is used.

    Returns:
        The MAD score of the data.
    """
    if mu is None:
        mu = np.median(data, axis=-1, keepdims=True)

    if sigma is None:
        sigma = mad(data)

    return (data - mu) / sigma


def modified_zscore(
    data: np.ndarray, mu: Optional[Union[float, np.ndarray]] = None, sigma: Optional[Union[float, np.ndarray]] = None
) -> np.ndarray:
    """Calculate the modified z-score of a dataset.

    Args:
        data: data to transform into modified z-scores
        mu: central value to use for z-scoring. If None, the median of the data is used.
        sigma: scale value to use for z-scoring. If None, the MAD of the data is used.

    Returns:
        The modified z-score of the data.
    """
    if mu is None:
        mu = np.median(data, axis=-1, keepdims=True)

    if sigma is None:
        sigma = mad(data)

    return 0.6745 * (data - mu) / sigma
