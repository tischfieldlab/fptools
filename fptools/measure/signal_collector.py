from typing import Optional
import numpy as np
import scipy

from fptools.io import Session, Signal
from fptools.preprocess.lib import fs2t


def collect_signals(
    session: Session, event: str, signal: str, start: float = -1.0, stop: float = 3.0, out_name: Optional[str] = None
) -> Signal:
    """Collect a signal from a session around an event.

    Args:
        session: the Session to operate on
        event: the name of the event to use
        signal: the name of the signal to collect
        start: start of the collection interval, in seconds, relative to each event. Negative values imply prior to event, positive values imply after the event
        stop: end of the collection interval, in seconds, relative to each event. Negative values imply prior to event, positive values imply after the event
        out_name: if not None, the name of the returned `Signal` object. Otherwise, the new `Signal` object's name will be generated as `{signal}@{event}`

    Returns:
        the collected Signal
    """
    assert start < stop

    sig = session.signals[signal]
    events = session.epocs[event]

    n_samples = int(np.rint((stop - start) * sig.fs))
    offset = int(np.rint(start * sig.fs))
    new_time = fs2t(sig.fs, n_samples) + start

    accum = np.zeros_like(sig.signal, shape=(events.shape[0], n_samples))
    padding = abs(offset) + n_samples
    padded_signal = np.pad(sig.signal, (padding, padding), mode="constant", constant_values=0)
    for ei, evt in enumerate(events):
        event_idx = sig.tindex(evt) + padding  # add padding
        accum[ei, :] = padded_signal[(event_idx + offset) : (event_idx + offset + n_samples)]

    if out_name is None:
        out_name = f"{signal}@{event}"

    s = Signal(out_name, accum, time=new_time, units=sig.units)
    s.marks[event] = 0
    return s


def collect_signals_2event(
    session: Session,
    event1: str,
    event2: str,
    signal: str,
    pre: float = 2.0,
    inter: float = 2.0,
    post: float = 2.0,
    out_name: Optional[str] = None,
) -> Signal:
    """Collect a signal from a session around two events.

    Collects a fixed amount of time before event1 and after event2. The "real" time between event1 and event2
    is scaled to a "meta" time specified by `inter`. This is done by resampling the inter-event time using
    linear interpolation.

    Args:
        session: the Session to operate on
        event1: the name of the first event to use
        event2: the name of the second event to use
        signal: the name of the signal to collect
        pre: amount of time, in seconds, to collect prior to each event
        inter: amount of "meta" time, in seconds, to collect between events
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
    if out_name is None:
        out_name = f"{signal}@{event1}>{event2}"

    s = Signal(out_name, accum, time=new_time, units=sig.units)
    s.marks[event1] = 0
    s.marks[event2] = inter

    # return the collected signal
    return s


# def collect_signals_2event2(
#     session: Session,
#     event1: str,
#     event2: str,
#     signal: str,
#     e1_range: tuple[float, float] = (-2.0, 0.0),
#     e2_range: tuple[float, float] = (-2.0, 0.0),
#     inter: float = 2.0,
#     out_name: Optional[str] = None
# ):
#     """Collect a signal from a session around two events.

#     Collects a fixed amount of time before event1 and after event2. The "real" time between event1 and event2
#     is scaled to a "meta" time specified by `inter`. This is done by resampling the inter-event time using
#     linear interpolation.

#     Args:
#         session: the Session to operate on
#         event1: the name of the first event to use
#         event2': the name of the second event to use
#         signal: the name of the signal to collect
#         pre: amount of time, in seconds, to collect prior to each event
#         inter: amount of "meta" time, in seconds, to collect between events
#         post: amount of time, in seconds, to collect after each event

#     Returns:
#         the collected Signal
#     """
#     # unpack arguments
#     sig = session.signals[signal]
#     events_1 = session.epocs[event1]
#     events_2 = session.epocs[event2]
#     # in a rare case, a stray event2 before event1, this will prune those
#     # not sure if it's the best idea to do this............
#     events_2 = events_2[events_2 > events_1.min()]
#     # assert len(events_1) == len(events_2)

#     # calculate index offsets etc
#     pre_idxs = int(np.rint(pre * sig.fs))
#     inter_idxs = int(np.rint(inter * sig.fs))
#     post_idxs = int(np.rint(post * sig.fs))
#     n_samples = pre_idxs + inter_idxs + post_idxs
#     new_time = fs2t(sig.fs, n_samples) - pre

#     e1_nsamples = int(np.rint((e1_range[1] - e1_range[0]) * sig.fs))
#     e1_offset = int(np.rint(e1_range[0] * sig.fs))

#     e2_nsamples = int(np.rint((e2_range[1] - e2_range[0]) * sig.fs))
#     e2_offset = int(np.rint(e2_range[0] * sig.fs))

#     # destination slices in the final signal
#     slice1 = slice(0, pre_idxs)
#     slice2 = slice(pre_idxs, pre_idxs + inter_idxs)
#     slice3 = slice(pre_idxs + inter_idxs, pre_idxs + inter_idxs + post_idxs)

#     accum = np.zeros_like(sig.signal, shape=(events_1.shape[0], n_samples))
#     padded_signal = np.pad(sig.signal, (pre_idxs, post_idxs), mode="constant", constant_values=0)
#     for ei, (evt1, evt2) in enumerate(zip(events_1, events_2)):
#         event1_idx = sig.tindex(evt1)
#         event2_idx = sig.tindex(evt2)

#         # collect the first third (pre to event1)
#         accum[ei, slice1] = padded_signal[event1_idx : (event1_idx + pre_idxs)]

#         # collect the second third (event1 to event2)
#         # idea here is train a scipy.interpolate.interp1d() object with our real data
#         # and then resample `inter_idxs` number of points from `inter_time`
#         inter_time = sig.time[event1_idx:event2_idx]
#         inter_sig = padded_signal[(event1_idx + pre_idxs) : (event2_idx + pre_idxs)]
#         inter_source_intp = scipy.interpolate.interp1d(inter_time, inter_sig)
#         inter_time_query = np.linspace(inter_time[0], inter_time[-1], inter_idxs, endpoint=True)
#         accum[ei, slice2] = inter_source_intp(inter_time_query)

#         # collect the final third (event 2 to post)
#         accum[ei, slice3] = padded_signal[(event2_idx + pre_idxs) : (event2_idx + pre_idxs + post_idxs)]

#     # construct the new signal object, and copy over proper metadata and add marks
#     if out_name is None:
#         out_name = f"{signal}@{event1}>{event2}"

#     s = Signal(out_name, accum, time=new_time, units=sig.units)
#     s.marks[event1] = 0
#     s.marks[event2] = inter

#     # return the collected signal
#     return s
