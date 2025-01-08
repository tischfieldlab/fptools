# Sessions and SessionCollections
A `Session` serves as the basic container for data. It typically would correspond to data from a single animal and a single bout of recording.
A group of `Session`s can be added to a `SessionCollection`. A `SessionCollection` offers several convience methods for working with many sessions.
Many functions in this package accept a `Session` or `SessionCollection`.

`Session`s typically will contain the following types of data on the attributes:

  - [`metadata`](#metadata): arbitrary metadata for a given session
  - [`signals`](#signals): continuous-valued time-series data sampled at fixed intervals
  - [`epocs`](#epocs): discrete event timestamps

## Metadata
Sessions can keep track of metadata about a recording as a simple dictionary, accessable via the `metadata` property.
Metadata is typically populated when a session is loaded, but metadata can be added at anytime.

When working with a `SessionCollection`, metadata for all sessions can be returned as a `pandas.DataFrame` from the `.metadata` property. There are several conveince methods that are useful for decorating sessions with addition metadata.

## Signals
A `Signal` encapsulates continuous-valued time-series data sampled at fixed intervals, along with some metadata, such as `name`, `unit`, `marks`. A `Signal` can describe itself.

```
> signal.describe()
Dopamine:
    units = ΔF/F
    n_observations = 1
    n_samples = 366051
    duration = 0:59:58.417848
    sample_rate = 101.72526245132634
```

### Shape of `Signal.signal`
Signal data can first be described by the number of samples it contains (i.e. the length of the array), and is retrievable by `Signal.nsamples`. The duration, as a wall clock amount of time, can be retrieved as a `datetime` by `Signal.duration`, and is equivelent to `Signal.nsamples * Signal.fs`. 

A `Signal` can contain data from a single observation (in this case `Signal.nobs == 1`), or from muliple observation (typical in the case of the return from `collect_signals()`, in this case `Signal.nobs > 1`). In the case of a single observation, the shape of the signal will be 1D, while in the case of multiple observations, the signal will be 2D `(nobs, nsamples)`.

### Signal aggregation
In the case there are multiple observations, one would commonly want to aggregate to produce a mean/median signal representative of all observations. To do so, use the `aggregate()` method of the signal object. This method return a new `Signal` with propegated marks, units, sampling frequency and time. The new signal will be named according to this signal, with `#{func}` appended. The parameter `func` determines how the signal is aggregated, and can accept several types of argument.

 - simple strings, like `mean`, `median`, `min`, `max` will invoke the `numpy` implementation (really any numpy function that is a `ufunc` and exists in the global `numpy` namespace)
 - reference to a `numpy` `ufunc`, such as `numpy.mean`
 - any other arbitrary function that accepts and array and returns an array


## Epocs
Epocs contain behavioral data. Currently, these are stored as a simple dictionary on `Session` objects, with string keys as the event name, and the values as numpy arrays of relative event times, in seconds.
