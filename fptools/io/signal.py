import datetime
from functools import partial
from typing import Any, Callable, Optional, Union, cast

import numpy as np
from typing import Union


class Signal(object):
    """Represents a real valued signal with fixed interval sampling."""

    def __init__(
        self, name: str, signal: np.ndarray, time: Optional[np.ndarray] = None, fs: Optional[float] = None, units: str = "AU"
    ) -> None:
        """Initialize a this Signal Object.

        At least one of `time` or `fs` must be provided. If `time` is provided, the sampling frequency (`fs`) will be estimated
        from `time`. If `fs` is provided, the sampled timepoints (`time`) will be estimated. If both are provided, the values
        will be checked against one another, and if they do not match, a ValueError will be raised.

        Args:
            name: Name of this signal
            signal: Array of signal values
            time: Array of sampled timepoints
            fs: Sampling frequency, in Hz
            units: units of this signal
        """
        self.name: str = name
        self.signal: np.ndarray = signal
        self.units: str = units
        self.marks: dict[str, float] = {}
        self.time: np.ndarray
        self.fs: float

        if time is not None and fs is not None:
            # both time and sampling frequency provided
            self.time = time
            self.fs = fs
            # just do a sanity check that the two pieces of information make sense
            if not np.isclose(self.fs, 1 / np.median(np.diff(time))):
                raise ValueError(
                    f"Both `time` and `fs` were provided, but they do not match!\n  fs={fs}\ntime={1 / np.median(np.diff(time))}"
                )

        elif time is None and fs is not None:
            # sampleing frequency is provided, infer time from fs
            self.fs = fs
            self.time = np.linspace(1, signal.shape[0], signal.shape[0]) / self.fs

        elif fs is None and time is not None:
            # time is provided, so lets estimate the sampling frequency
            self.time = time
            self.fs = 1 / np.median(np.diff(time))

        else:
            # neither time or sampling frequency provided, we need at least one!
            raise ValueError("Both `time` and `fs` cannot be `None`, one must be supplied!")

        if not self.signal.shape[-1] == self.time.shape[0]:
            raise ValueError(f"Signal and time must have the same length! signal.shape={self.signal.shape}; time.shape={self.time.shape}")

    @property
    def nobs(self) -> int:
        """Get the number of observations for this Signal (i.e. number of trials)."""
        return self.signal.shape[0] if len(self.signal.shape) > 1 else 1

    @property
    def nsamples(self) -> int:
        """Get the number of samples for this Signal (i.e. length of signal)."""
        return self.signal.shape[-1]

    @property
    def duration(self) -> datetime.timedelta:
        """Get the duration of this Signal."""
        return datetime.timedelta(seconds=self.time[-1] - self.time[0])

    def tindex(self, t: float) -> int:
        """Get the time index closest to time `t`."""
        return (np.abs(self.time - t)).argmin()

    def copy(self, new_name: Optional[str] = None) -> "Signal":
        """Return a deep copy of this signal.

        Args:
            new_name: if not None, assign the copy this new name

        Returns:
            A copy of this Signal
        """
        if new_name is None:
            new_name = self.name
        s = type(self)(new_name, self.signal.copy(), time=self.time.copy(), fs=self.fs, units=self.units)
        s.marks.update(**self.marks)
        return s

    def __check_other_compatible(self, other: "Signal") -> bool:
        return self.nsamples == other.nsamples and self.nobs == self.nobs and self.fs == other.fs

    def __add__(self, other: "Signal") -> "Signal":
        """Add another signal to this signal, returning a new signal.

        Args:
            other: the other signal to be added to this signal

        Return:
            A new Signal with the addition result.
        """
        assert self.__check_other_compatible(other)
        s = self.copy()
        s.signal += other.signal
        return s

    def __sub__(self, other: "Signal") -> "Signal":
        """Subtract another signal from this signal, returning a new signal.

        Args:
            other: the other signal to be subtracted from this signal

        Return:
            A new Signal with the subtraction result.
        """
        assert self.__check_other_compatible(other)
        s = self.copy()
        s.signal -= other.signal
        return s

    def __mul__(self, other: "Signal") -> "Signal":
        """Multiply another signal against this signal, returning a new signal.

        Args:
            other: the other signal to be multiplied by this signal

        Return:
            A new Signal with the multiplication result.
        """
        assert self.__check_other_compatible(other)
        s = self.copy()
        s.signal *= other.signal
        return s

    def __div__(self, other: "Signal") -> "Signal":
        """Divide this signal by another signal, returning a new signal.

        Args:
            other: the other signal to divide this signal by

        Return:
            A new Signal with the division result.
        """
        assert self.__check_other_compatible(other)
        s = self.copy()
        s.signal /= other.signal
        return s

    def aggregate(self, func: Union[str, np.ufunc, Callable[[np.ndarray], np.ndarray]]) -> "Signal":
        """Aggregate this signal.

        If there is only a single observation, that observation is returned unchanged, otherwise  observations will
        be aggregated by `func` along axis=0.

        Marks, units, and time will be propegated. The new signal will be named according to this signal, with `#{func_name}` appended.

        Args:
            func: string or callable that take a (nobs x nsample) array and returns a (nsample,) shaped array. If a string
                will be interpreted as the name of a numpy function (e.x. mean, median, etc)

        Returns:
            aggregated `Signal`
        """
        if self.nobs == 1:
            return self  # maybe we should raise??

        else:
            f: Callable[[np.ndarray], np.ndarray]
            if isinstance(func, str):
                f = partial(cast(np.ufunc, getattr(np, func)), axis=0)
                f_name = func

            elif isinstance(func, np.ufunc):
                f = partial(func, axis=0)
                f_name = func.__name__

            else:
                f = func
                f_name = func.__name__

            s = Signal(f"{self.name}#{f_name}", f(self.signal), time=self.time, units=self.units)
            s.marks.update(self.marks)
            return s

    def describe(self, as_str: bool = False, prefix: str = "") -> Union[str, None]:
        """Describe this Signal.

        Args:
            as_str: if True, return description as a string, otherwise print the description and return None
            prefix: a string to prepend to each line of output

        Returns:
            `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = f"{prefix}{self.name}:\n"
        buffer += f"{prefix}    units = {self.units}\n"
        buffer += f"{prefix}    n_observations = {self.nobs}\n"
        buffer += f"{prefix}    n_samples = {self.nsamples}\n"
        buffer += f"{prefix}    duration = {self.duration}\n"
        buffer += f"{prefix}    sample_rate = {self.fs}\n"
        buffer += f"{prefix}    min|mean|max = {self.signal.min():.2f}|{self.signal.mean():.2f}|{self.signal.max():.2f}\n"
        if len(self.marks) > 0:
            buffer += f"{prefix}    marks ({len(self.marks)}):\n"
            for k, v in self.marks.items():
                buffer += f"{prefix}        {datetime.timedelta(seconds=v)} {k}\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None
