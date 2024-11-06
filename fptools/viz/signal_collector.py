import numpy as np
import scipy

from fptools.io import Session, Signal
from fptools.preprocess.lib import fs2t


def collect_signals(session: Session, event: str, signal: str, pre: float = 1.0, post: float = 2.0) -> Signal:
    """Collect a signal from a session around an event.

    Parameters:
    session: the Session to operate on
    event: the name of the event to use
    signal: the name of the signal to collect
    pre: amount of time in seconds to collect prior to each event
    post: amount of time in seconds to collect after each event

    Returns:
    the collected Signal
    """
    sig = session.signals[signal]
    events = session.epocs[event]
    pre_idxs = int(np.rint(pre * sig.fs))
    post_idxs = int(np.rint(post * sig.fs))
    n_samples = pre_idxs + post_idxs
    new_time = fs2t(sig.fs, n_samples) - pre

    accum = np.zeros_like(sig.signal, shape=(events.shape[0], n_samples))
    padded_signal = np.pad(sig.signal, (pre_idxs, post_idxs), mode="constant", constant_values=0)
    for ei, evt in enumerate(events):
        event_idx = sig.tindex(evt)
        start = event_idx - pre_idxs
        stop = event_idx + post_idxs
        accum[ei, :] = padded_signal[(start + pre_idxs) : (stop + pre_idxs)]

    s = Signal(f"{signal}@{event}", accum, time=new_time, units=sig.units)
    s.marks[event] = 0
    return s


def collect_signals_2event(
    session: Session, event1: str, event2: str, signal: str, pre: float = 2.0, inter: float = 2.0, post: float = 2.0
):
    """Collect a signal from a session around two events.

    Collects a fixed amount of time before event1 and after event2. The "real" time between event1 and event2
    is scaled to a "meta" time specified by `inter`. This is done by resampling the inter-event time using
    linear interpolation.

    Parameters:
    session: the Session to operate on
    event1: the name of the first event to use
    event2': the name of the second event to use
    signal: the name of the signal to collect
    pre: amount of time, in seconds, to collect prior to each event
    inter: amouont of "meta" time, in seconds, to collect between events
    post: amount of time, in seconds, to collect after each event

    Returns:
    the collected Signal
    """
    # unpack arguments
    sig = session.signals[signal]
    events_1 = session.epocs[event1]
    events_2 = session.epocs[event2]
    # in a rare case, a stray event2 before event1, this will prune those
    # not sure if it's the best idea to do this............
    events_2 = events_2[events_2 > events_1.min()]
    # assert len(events_1) == len(events_2)

    # calculate index offsets etc
    pre_idxs = int(np.rint(pre * sig.fs))
    inter_idxs = int(np.rint(inter * sig.fs))
    post_idxs = int(np.rint(post * sig.fs))
    n_samples = pre_idxs + inter_idxs + post_idxs
    new_time = fs2t(sig.fs, n_samples) - pre

    # destination slices in the final signal
    slice1 = slice(0, pre_idxs)
    slice2 = slice(pre_idxs, pre_idxs + inter_idxs)
    slice3 = slice(pre_idxs + inter_idxs, pre_idxs + inter_idxs + post_idxs)

    accum = np.zeros_like(sig.signal, shape=(events_1.shape[0], n_samples))
    padded_signal = np.pad(sig.signal, (pre_idxs, post_idxs), mode="constant", constant_values=0)
    for ei, (evt1, evt2) in enumerate(zip(events_1, events_2)):
        event1_idx = sig.tindex(evt1)
        event2_idx = sig.tindex(evt2)

        # collect the first third (pre to event1)
        accum[ei, slice1] = padded_signal[event1_idx : (event1_idx + pre_idxs)]

        # collect the second third (event1 to event2)
        # idea here is train a scipy.interpolate.interp1d() object with our real data
        # and then resample `inter_idxs` number of points from `inter_time`
        inter_time = sig.time[event1_idx:event2_idx]
        inter_sig = padded_signal[(event1_idx + pre_idxs) : (event2_idx + pre_idxs)]
        inter_source_intp = scipy.interpolate.interp1d(inter_time, inter_sig)
        inter_time_query = np.linspace(inter_time[0], inter_time[-1], inter_idxs, endpoint=True)
        accum[ei, slice2] = inter_source_intp(inter_time_query)

        # collect the final third (event 2 to post)
        accum[ei, slice3] = padded_signal[(event2_idx + pre_idxs) : (event2_idx + pre_idxs + post_idxs)]

    # construct the new signal object, and copy over proper metadata and add marks
    s = Signal(f"{signal}@{event1}>{event2}", accum, time=new_time, units=sig.units)
    s.marks[event1] = 0
    s.marks[event2] = inter

    # return the collected signal
    return s
