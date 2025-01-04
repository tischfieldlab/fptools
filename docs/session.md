# Sessions and SessionCollections
A `Session` serves as the basic container for data. It typically would correspond to data from a single animal and a single bout of recording.
A group of `Session`s can be added to a `SessionCollection`. A `SessionCollection` offers several convience methods for working with many sessions.
Many functions in this package accept a `Session` or `SessionCollection`.

## Metadata
Sessions can keep track of metadata about a recording as a simple dictionary, accessable via the `metadata` property.
Metadata is typically populated when a session is loaded, but metadata can be added at anytime.

When working with a `SessionCollection`, metadata for all sessions can be returned as a `pandas.DataFrame` from the `.metadata` property. There are several conveince methods that are useful for decorating sessions with addition metadata.

## Signals
A `Signal` encapsulates continuous-valued time-series data sampled at fixed intervals, along with some metadata, such as `name`, `unit`, `marks`.

::: fptools.io.Signal
    options:
        summary: false