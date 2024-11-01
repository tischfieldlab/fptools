from collections import Counter
import datetime
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import pandas as pd


FieldList = Union[Literal["all"], list[str]]

class Signal(object):
    def __init__(self, name: str, signal: np.ndarray, time: Optional[np.ndarray] = None, fs: Optional[float] = None, units: str = 'AU') -> None:
        '''Initialize a this Signal Object.

        At least one of `time` or `fs` must be provided. If `time` is provided, the sampling frequency (`fs`) will be estimated
        from `time`. If `fs` is provided, the sampled timepoints (`time`) will be estimated. If both are provided, the values
        will be checked against one another, and if they do not match, a ValueError will be raised.

        Parameters:
        name: Name of this signal
        signal: Array of signal values
        time: Array of sampled timepoints
        fs: Sampling frequency, in Hz
        units: units of this signal
        '''
        self.name = name
        self.signal = signal
        self.units = units
        self.marks = {}

        if time is None and fs is None:
            # neither time or sampling frequency provided, we need at least one!
            raise ValueError('Both `time` and `fs` cannot be `None`, one must be supplied!')

        elif time is None and fs is not None:
            # sampleing frequency is provided, infer time from fs
            self.fs = fs
            self.time = np.linspace(1, signal.shape[0], signal.shape[0]) / self.fs

        elif fs is None and time is not None:
            # time is provided, so lets estimate the sampling frequency
            self.time = time
            self.fs = 1 / np.median(np.diff(time))

        else:
            # both time and sampling frequency provided
            self.time = time
            self.fs = fs
            # just do a sanity check that the two pieces of information make sense
            if not np.isclose(self.fs, 1 / np.median(np.diff(time))):
                raise ValueError(f'Both `time` and `fs` were provided, but they do not match!\n  fs={fs}\ntime={1 / np.median(np.diff(time))}')

        if not self.signal.shape[-1] == self.time.shape[0]:
            raise ValueError(f'Signal and time must have the same length! signal.shape={self.signal.shape}; time.shape={self.time.shape}')

    @property
    def nobs(self) -> int:
        """Get the number of observations for this Signal (i.e. number of trials)
        """
        return self.signal.shape[0] if len(self.signal.shape) > 1 else 1

    @property
    def nsamples(self) -> int:
        """Get the number of samples for this Signal (i.e. length of signal)
        """
        return self.signal.shape[-1]

    @property
    def duration(self) -> datetime.timedelta:
        '''Get the duration of this Signal
        '''
        return datetime.timedelta(seconds=self.time[-1] - self.time[0])

    def tindex(self, t: float) -> int:
        '''Get the time index closest to time `t`
        '''
        return (np.abs(self.time - t)).argmin()

    def copy(self) -> 'Signal':
        '''Return a deep copy of this signal
        '''
        s = type(self)(self.name, self.signal.copy(), time=self.time.copy(), fs=self.fs, units=self.units)
        s.marks.update(**self.marks)
        return s

    def aggregate(self, func: Union[str, Callable[[np.ndarray], np.ndarray]]) -> "Signal":
        '''Aggregate this signal.

        If there is only a single observation, that observation is returned unchanged, otherwise  observations will
        be aggregated by `func` along axis=0.

        Marks, units, and time will be propegated. The new signal will be named according to this signal, with `#{func_name}` appended.

        Parameters:
        func: string or callable that take a (nobs x nsample) array and returns a (nsample,) shaped array. If a string
              will be interpreted as the name of a numpy function (e.x. mean, median, etc)

        Returns:
        aggregated signal
        '''
        if self.nobs == 1:
            return self #maybe we should raise??
        else:
            f: Callable[[np.ndarray], np.ndarray]
            if isinstance(func, str):
                f = getattr(np, func)
            else:
                f = func

            s = Signal(f'{self.name}#{f.__name__}', f(self.signal, axis=0), time=self.time, units=self.units)
            s.marks.update(self.marks)
            return s

    def describe(self, as_str: bool = False, prefix: str = "") -> Union[str, None]:
        """Describe this Signal

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None
        prefix: a string to prepend to each line of output

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer  = f'{prefix}{self.name}:\n'
        buffer += f'{prefix}    units = {self.units}\n'
        buffer += f'{prefix}    n_observations = {self.nobs}\n'
        buffer += f'{prefix}    n_samples = {self.nsamples}\n'
        buffer += f'{prefix}    duration = {self.duration}\n'
        buffer += f'{prefix}    sample_rate = {self.fs}\n'
        if len(self.marks) > 0:
            buffer += f'{prefix}    marks ({len(self.marks)}):\n'
            for k, v in self.marks.items():
                buffer += f'{prefix}        {datetime.timedelta(seconds=v)} {k}\n'

        if as_str:
            return buffer
        else:
            print(buffer)
            return None




class Session(object):
    """Holds data and metadata for a single session."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.signals: dict[str, Signal] = {}
        self.epocs: dict[str, np.ndarray] = {}

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this session

        describes the metadata, scalars, and arrays contained in this session.

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = ""

        buffer += "Metadata:\n"
        if len(self.metadata) > 0:
            for k, v in self.metadata.items():
                buffer += f"    {k}: {v}\n"
        else:
            buffer += '    < No Metadata Available >\n'
        buffer += "\n"

        buffer += "Epocs:\n"
        if len(self.epocs) > 0:
            for k, v in self.epocs.items():
                #buffer += f'    "{k}" with shape {v.shape}:\n    {np.array2string(v, prefix="    ")}\n\n'
                buffer += f'    {k}:\n'
                buffer += f'        num_events = {v.shape}\n'
                buffer += f'        avg_rate = {datetime.timedelta(seconds=np.diff(v)[0])}\n'
                buffer += f'        earliest = {datetime.timedelta(seconds=v[0])}\n'
                buffer += f'        latest = {datetime.timedelta(seconds=v[-1])}\n'
        else:
            buffer += '    < No Epocs Available >\n'
        buffer += "\n"

        buffer += "Signals:\n"
        if len(self.signals) > 0:
            for k, v in self.signals.items():
                buffer += v.describe(as_str=True, prefix="    ")
        else:
            buffer += '    < No Signals Available >\n'
        buffer += "\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def add_signal(self, signal: Signal, overwrite: bool = False):
        """Add a signal to this Session

        Raises an error if the new signal name already exists and `overwrite` is not True.

        Parameters:
        signal: the signal to add to this Session
        """
        if signal.name in self.signals and not overwrite:
            raise KeyError(f"Key `{signal.name}` already exists in data!")

        self.signals[signal.name] = signal

    def rename_signal(self, old_name: str, new_name: str):
        """Rename a signal, from `old_name` to `new_name`.

        Raises an error if the new signal name already exists.

        Parameters:
        old_name: the current name for the signal
        new_name: the new name for the signal
        """
        if new_name in self.signals:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.signals[new_name] = self.signals[old_name]
        self.signals.pop(old_name)

    def rename_epoc(self, old_name: str, new_name: str):
        """Rename a epoc, from `old_name` to `new_name`.

        Raises an error if the new epoc name already exists.

        Parameters:
        old_name: the current name for the epoc
        new_name: the new name for the epoc
        """
        if new_name in self.epocs:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.epocs[new_name] = self.epocs[old_name]
        self.epocs.pop(old_name)




class SessionCollection(list[Session]):
    """Collection of session data"""

    @property
    def metadata(self) -> pd.DataFrame:
        """Get a dataframe containing metadata across all sessions in this collection."""
        return pd.DataFrame([item.metadata for item in self])

    @property
    def metadata_keys(self) -> List[str]:
        """Get a list of the keys present in metadata across all sessions in this collection"""
        return list(set([key for item in self for key in item.metadata.keys()]))

    def add_metadata(self, key: str, value: Any) -> None:
        """Set a metadata field on each session in this collection

        Parameters:
        key: name of the metadata field
        value: value for the metadata field
        """
        for item in self:
            item.metadata[key] = value

    def update_metadata(self, meta: dict[str, Any]) -> None:
        """Set multiple metadata fields on each session in this collection

        Parameters:
        meta: metadata information to set on each session
        """
        for item in self:
            item.metadata.update(meta)

    def rename_signal(self, old_name: str, new_name: str) -> None:
        """Rename a signal on each session in this collection

        Parameters:
        old_name: current name of the signal
        new_name: the new name for the signal
        """
        for item in self:
            item.rename_signal(old_name, new_name)

    def rename_epoc(self, old_name: str, new_name: str) -> None:
        """Rename an epoc on each session in this collection

        Parameters:
        old_name: current name of the epoc
        new_name: the new name for the epoc
        """
        for item in self:
            item.rename_epoc(old_name, new_name)

    def filter(self, predicate: Callable[[Session], bool]) -> "SessionCollection":
        """Filter the items in this collection, returning a new `SessionCollection` containing sessions which pass `predicate`.

        Parameters:
        predicate: a callable accepting a single session and returning bool.

        Returns:
        a new `SessionCollection` containing only itemss which pass `predicate`.
        """
        return type(self)(item for item in self if predicate(item))

    def map(self, action: Callable[[Session], Session]) -> "SessionCollection":
        """Apply a function to each session in this collection, returning a new collection with the results

        Parameters:
        action: callable accepting a single session and returning a new session

        Returns:
        a new `SessionCollection` containing the results of `action`
        """
        return type(self)(action(item) for item in self)

    def apply(self, func: Callable[[Session], None]) -> None:
        """Apply a function to each session in this collection

        Parameters:
        func: callable accepting a single session and returning None
        """
        for item in self:
            func(item)

    def get_signal(self, name: str) -> list[Signal]:
        """Get data across sessions in this collection for the signal named `name`

        Parameters:
        name: Name of the signals to collect

        Returns
        List of Signals, each corresponding to a single session
        """
        return [item.signals[name] for item in self]

    def aggregate_signals(self, name: str, method='mean') -> Signal:
        """Aggregate signals across sessions in this collection for the signal name `name`
        """
        signals = self.get_signal(name)
        if len(signals) <= 0:
            raise ValueError('No signals were passed!')

        # check all signals have the same number of samples
        assert np.all(np.equal([s.nsamples for s in signals], signals[0].nsamples))

        if method is not None:
            signals = [s.aggregate(method) for s in signals]

        s = Signal(signals[0].name, np.vstack([s.signal for s in signals]), time=signals[0].time, units=signals[0].units)
        s.marks.update(signals[0].marks)
        return s

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this collection of sessions

        Parameters:
        as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
        `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = ""

        buffer += f"Number of sessions: {len(self)}\n\n"

        signals = Counter([item for session in self for item in session.signals.keys()])
        buffer += "Signals present in data with counts:\n"
        for k, v in signals.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        epocs = Counter([item for session in self for item in session.epocs.keys()])
        buffer += "Epocs present in data with counts:\n"
        for k, v in epocs.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None
