import numpy as np
import scipy
import scipy.stats


def fs2t(fs: float, length: int) -> np.ndarray:
    '''Generate a time array given a sample frequency and number of samples

    Parameters:
    fs: sampling rate, in Hz
    length: number of samples to generate

    Returns
    array of time values, in seconds
    '''
    return np.linspace(1, length, length) / fs


def t2fs(time: np.ndarray) -> float:
    '''Estimate the sample frequency given a time array

    Parameters:
    time: array of time values, in seconds, from which to estimate the sample frequency

    Returns
    The estimated sample frequency, in Hz
    '''
    return 1 / np.median(np.diff(time))

def lowpass_filter(signal: np.ndarray, fs: float, Wn: float = 10) -> np.ndarray:
    '''zero-phase lowpass filter a signal.

    Parameters:
    signal: array to be filtered
    fs: sampling frequency of the signal, in Hz
    Wn: critical frequency, see `scipy.signal.butter()`

    Returns:
    lowpass filtered signal
    '''
    b, a = scipy.signal.butter(2, Wn, btype='lowpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def double_exponential(t: np.ndarray, const: float, amp_fast: float, amp_slow: float, tau_slow: float, tau_multiplier: float) -> np.ndarray:
    '''Compute a double exponential function with constant offset.

    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset.
    amp_fast: Amplitude of the fast component.
    amp_slow: Amplitude of the slow component.
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow.
    '''
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def fit_double_exponential(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    max_sig = np.max(signal)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    parm_opt, parm_cov = scipy.optimize.curve_fit(double_exponential,
                                                  time,
                                                  signal,
                                                  p0=inital_params,
                                                  bounds=bounds,
                                                  maxfev=1000)
    return double_exponential(time, *parm_opt)


def detrend_double_exponential(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Detrend a signal by fitting and subtracting a double exponential curve to the data.

    See `double_exponential` for the underlying curve design, and `fit_double_exponential` for the curve fitting procedure.

    Parameters:
    time: array of sample/observation times
    signal: array of samples

    Returns:
    Tuple of (detrended_signal, signal_fit)
    '''
    signal_fit = fit_double_exponential(time, signal)
    return signal - signal_fit, signal_fit


def estimate_motion(signal: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Estimate the contribution of motion artifacts in `signal`.

    Performs linear regression of control (x, independent) vs signal (y, dependent)

    Parameters:
    signal: the signal to correct for motion artifacts
    control: signal to use for background signal

    Returns:
    tuple of (corrected_signal, estimated_motion)
    '''
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=control, y=signal)
    est_motion = intercept + slope * (control)
    return signal - est_motion, est_motion


def are_arrays_same_length(*arrays: np.ndarray) -> bool:
    '''Check if all arrays are the same shape in the first axis
    '''
    lengths = [arr.shape[0] for arr in arrays]
    return bool(np.all(np.array(lengths) == lengths[0]))


# def downsample(*signals, factor: int = 10):
#     assert are_arrays_same_length(*signals)

#     old_shape = signals[0].shape[0]
#     new_shape = old_shape // factor
#     downsampled = [np.empty_like(sig, shape=new_shape) for sig in signals]
#     for i in range(0, old_shape, factor):
#         for si, sig in enumerate(signals):
#             downsampled[si][i] = np.mean(sig[i:i+factor])  # This is the moving window mean

#     return downsampled


def downsample(*signals, window: int = 10, factor: int = 10) -> tuple[np.ndarray, ...]:
    '''Downsample one or more signals by factor across windows of size `window`

    performs a moving window average using windows of size `window`, then takes every
    n-th observation as given by `factor`.

    Parameters:
    signals: one or more signals to downsample
    window: size of the window used for averaging
    factor: step size for taking the final downsampled signal

    Returns:
    downsampled signal
    '''
    #assert are_arrays_same_length(*signals)
    return tuple(np.convolve(sig, np.ones(window) / window, mode='valid')[::factor] for sig in signals)



def trim(*signals, begin=None, end=None) -> tuple[np.ndarray, ...]:
    '''Trim samples from the beginning or end of a signal

    Parameters:
    signals: one or more signals to be trimmed
    begin: number of samples to trim from the beginning
    end: number of samples to trim from the end

    Returns:
    tuple of trimmed signals
    '''
    assert are_arrays_same_length(*signals)
    if begin is None:
        begin = 0
    if end is None:
        end = signals[0].shape[0]
    return tuple(sig[begin:end] for sig in signals)


def df_f(signal: np.ndarray, isosbestic: np.ndarray):
    pass


def zscore_signals(*signals):
    return (scipy.stats.zscore(sig) for sig in signals)
