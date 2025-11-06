from collections import Counter, defaultdict
import copy
import datetime
from functools import partial
import math
import os
import sys
from typing import Any, Callable, Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd

from .signal import Signal


FieldList = Union[Literal["all"], list[str]]


def empty_array() -> np.ndarray:
    """Create an empty numpy array.

    Returns:
        empty numpy array
    """
    return np.ndarray([], dtype=np.float64)


def empty_df() -> pd.DataFrame:
    """Create an empty Pandas dataframe.

    Returns:
        empty pd.DataFrame

    """
    return pd.DataFrame()


class Session(object):
    """Holds data and metadata for a single session."""

    def __init__(self) -> None:
        """Initialize this Session object."""
        self._signatures: dict[str, str] = {}
        self.name: str = ""
        self.metadata: dict[str, Any] = {}
        self.signals: dict[str, Signal] = {}
        self.epocs: dict[str, np.ndarray] = defaultdict(empty_array)  # epocs are numpy arrays, default to empty array
        self.scalars: dict[str, np.ndarray] = defaultdict(empty_array)  # scalars are numpy arrays, default to empty array
        self.dlc: dict[str, np.ndarray] = defaultdict(empty_array)  # dlc data are structured numpy arrays, default to empty array
        self.analysis: dict[str, np.ndarray] = defaultdict(empty_array) #analysis data are strictly 1d numpy arrays, default to empty array
        self.misc: dict[str, Any] = {}  # WARNING: careful what datatypes stored in misc, some datatypes will not play well with saving into an hdf5

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this session.

        describes the metadata, scalars, and arrays contained in this session.

        Args:
            as_str: if True, return description as a string, otherwise print the description and return None

        Returns:
            `None` if `as_str` is `False`; if `as_str` is `True`, returns the description as a `str`
        """
        buffer = f'Session with name "{self.name}"\n\n'

        buffer += "Metadata:\n"
        if len(self.metadata) > 0:
            for k, v in self.metadata.items():
                buffer += f"    {k}: {v}\n"
        else:
            buffer += "    < No Metadata Available >\n"
        buffer += "\n"

        buffer += "Epocs:\n"
        if len(self.epocs) > 0:
            for k, v in self.epocs.items():
                # buffer += f'    "{k}" with shape {v.shape}:\n    {np.array2string(v, prefix="    ")}\n\n'
                buffer += f"    {k}:\n"
                buffer += f"        num_events = {v.shape}\n"
                if len(v) >= 2:
                    buffer += f"        avg_rate = {datetime.timedelta(seconds=np.diff(v)[0])}\n"
                buffer += f"        earliest = {datetime.timedelta(seconds=v[0])}\n"
                buffer += f"        latest = {datetime.timedelta(seconds=v[-1])}\n"
        else:
            buffer += "    < No Epocs Available >\n"
        buffer += "\n"

        buffer += "Scalars:\n"
        if len(self.scalars) > 0:
            for k, v in self.scalars.items():
                buffer += f"    {k}:\n"
                buffer += f"        num_events = {v.shape}\n"
                buffer += f"        earliest = {datetime.timedelta(seconds=v[0])}\n"
                buffer += f"        latest = {datetime.timedelta(seconds=v[-1])}\n"
        else:
            buffer += "    < No Epocs Available >\n"
        buffer += "\n"

        buffer += "Signals:\n"
        if len(self.signals) > 0:
            for k, v in self.signals.items():
                buffer += v.describe(as_str=True, prefix="    ")
        else:
            buffer += "    < No Signals Available >\n"
        buffer += "\n"

        buffer += "DLC Data:\n"
        if len(self.dlc) > 0:
            buffer += f"{len(self.dlc)} DLC arrays found: \n"
            for k, v in self.dlc.items():
                buffer += f"    {k}: \n"
                buffer += f"        array_shape = {v.shape} \n"
        else:
            buffer += "    < No DLC arrays Available >\n"
        buffer += "\n"

        buffer += "Analysis Data:\n"
        if len(self.analysis) > 0:
            buffer += f"{len(self.analysis)} Analysis data arrays found: \n"
            for k, v in self.analysis.items():
                buffer += f"    {k}: \n"
                buffer += f"        array_shape = {v.shape} \n"
        else:
            buffer += "    < No analysis data arrays available >\n"
        buffer += "\n"

        buffer += "Misc:\n"
        if len(self.misc) > 0:
            buffer += f"{len(self.misc)} Misc items found: \n"
            for k, v in self.misc.items():
                buffer += f"    {k}: \n"
                buffer += f"        data type = {type(v)} \n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None
        
    def add_epoc(self, arr: np.ndarray, name: str, overwrite: bool = False) -> None:
        """Add epoc data to this Session.
        
        Raises an error if the new epoc name already exists and `overwrite` is not True.
        
        Args:
            epoc: 1D numpy array of epoc timestamps
            overwrite: if True, allow overwriting a pre-existing epoc with the same name, if False, will raise error instead
        """
        if name in self.epocs and not overwrite:
            raise KeyError(f"Key `{name}` already exists in analysis data!")
        
        if isinstance(arr, np.ndarray):
            if arr.ndim != 1:
                raise ValueError(f"Epoc data must be 1-dimensional, but has {arr.ndim} dimensions.")
            
            self.epocs[name] = arr
        
        else:
            raise TypeError("Invalid `arr` argument data type. Supported data types are numpy arrays.")

    def add_signal(self, signal: Signal, overwrite: bool = False) -> None:
        """Add a signal to this Session.

        Raises an error if the new signal name already exists and `overwrite` is not True.

        Args:
            signal: the signal to add to this Session
            overwrite: if True, allow overwriting a pre-existing signal with the same name, if False, will raise instead.
        """
        if signal.name in self.signals and not overwrite:
            raise KeyError(f"Key `{signal.name}` already exists in data!")

        self.signals[signal.name] = signal

    def remove_signal(self, name: str) -> Signal:
        """Remove a signal from this Session.

        Args:
            name: the name of the signal to remove

        Returns:
            the signal that was removed
        """
        return self.signals.pop(name)

    def rename_signal(self, old_name: str, new_name: str) -> None:
        """Rename a signal, from `old_name` to `new_name`.

        Raises an error if the new signal name already exists.

        Args:
            old_name: the current name for the signal
            new_name: the new name for the signal
        """
        if new_name in self.signals:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.signals[new_name] = self.signals[old_name]
        self.signals[new_name].name = new_name
        self.signals.pop(old_name)

    def rename_scalar(self, old_name: str, new_name: str):
        """Rename a scalar, from `old_name` to `new_name`.

        Raises an error if the new scalar name already exists.

        Args:
        old_name: the current name for the scalar
        new_name: the new name for the scalar
        """
        if new_name in self.scalars:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.scalars[new_name] = self.scalars[old_name]
        self.scalars.pop(old_name)

    def rename_epoc(self, old_name: str, new_name: str) -> None:
        """Rename a epoc, from `old_name` to `new_name`.

        Raises an error if the new epoc name already exists.

        Args:
            old_name: the current name for the epoc
            new_name: the new name for the epoc
        """
        if new_name in self.epocs:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.epocs[new_name] = self.epocs[old_name]
        self.epocs.pop(old_name)

    def add_dlc(self, dlc: Union[pd.DataFrame, np.ndarray], name: str, overwrite: bool = False) -> None:
        """Add a DLC data to this Session.

        Raises an error if the new DLC data name already exists and `overwrite` is not True.

        Args:
            dlc: pd.DataFrame or structured numpy arrays, the DLC data to add to this Session
            overwrite: if True, allow overwriting a pre-existing signal with the same name, if False, will raise instead.
        """
        if name in self.dlc and not overwrite:
            raise KeyError(f"Key `{name}` already exists in data!")
        
        # if isinstance(dlc, pd.DataFrame):
        #     dlc.columns = dlc.columns.to_flat_index()

        #     string_cols = dlc.select_dtypes(include=['str']).columns
        #     for col in string_cols:
        #         dlc[col] = dlc[col].astype(np.bytes_).astype('S50')

        #     nparray = dlc.to_records(index=False)
        #     self.dlc[name] = nparray
        
        # elif isinstance(dlc, np.ndarray):
        self.dlc[name] = dlc
        
        # else:
        #     raise TypeError("Invalid `dlc` argument data type. Supported data types are pd.DataFrame and numpy arrays.") 

    def rename_dlc(self, old_name: str, new_name: str) -> None:
        """Rename a dlc array, from `old_name` to `new_name`.

        Raises an error if the new dlc array name already exists.

        Args:
            old_name: the current name for the dlc array
            new_name: the new name for the dlc array
        """
        if new_name in self.dlc:
            raise KeyError(f"Key `{new_name}` already exists in data!")

        self.dlc[new_name] = self.dlc[old_name]
        self.dlc.pop(old_name)

    def add_analysis(self, arr: np.ndarray, name: str, overwrite: bool = False) -> None:
        """Add analysis data to this Session.
        
        Raises an error if the new analysis data name already exists and `overwrite` is not True.
        
        Args:
            analysis: 1D numpy array of analysis data
            overwrite: if True, allow overwriting a pre-existing analysis with the same name, if False, will raise error instead
        """
        if name in self.analysis and not overwrite:
            raise KeyError(f"Key `{name}` already exists in analysis data!")
        
        if isinstance(arr, np.ndarray):
            if arr.ndim != 1:
                raise ValueError(f"Analysis array must be 1-dimensional, but has {arr.ndim} dimensions.")
            
            self.analysis[name] = arr
        
        else:
            raise TypeError("Invalid `arr` argument data type. Supported data types are numpy arrays.")
    
    def rename_analysis(self, old_name: str, new_name: str) -> None:
        """Rename an analysis array, from `old_name` to `new_name`.

        Raises an error if the new analysis array name already exists.

        Args:
            old_name: the current name for the analysis array
            new_name: the new name for the analysis array
        """
        if new_name in self.analysis:
            raise KeyError(f"Key `{new_name}` already exists in analysis data!")

        self.analysis[new_name] = self.analysis[old_name]
        self.analysis.pop(old_name)

    def add_misc(self, data: Any, name: str, overwrite: bool = False) -> None:
        """Add Misc item to this Session
        
        Raises an error if the new misc ite, already exists and `overwrite` is not True.
        
        Args:
            data: misc item
            overwrite: if True, allow overwriting a pre-existing misc item with the same name, if False, will raise error instead
        """
        if name in self.misc and not overwrite:
            raise KeyError(f"Key `{name}` already exists in misc items!")
        
        else:
            self.misc[name] = data
    
    def rename_misc(self, old_name: str, new_name: str) -> None:
        """Rename a misc item, from `old_name` to `new_name`.
        
        Raises an error if the new analysis array name already exists.
        
        Args:
            old_name: the current name for the misc item
            new_name: the new name for the misc item
        """
        if new_name in self.misc:
            raise KeyError(f"Key `{new_name} already exists in misc items!")
        
        self.misc[new_name] = self.misc[old_name]
        self.misc.pop(old_name)

    def epoc_dataframe(self, include_epocs: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with epoc data and metadata.

        Args:
            include_epocs: list of array names to include in the dataframe. Special str "all" is also accepted.
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
            DataFrame with data from this session
        """
        # determine metadata fields to include
        if include_meta == "all":
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine arrays to include
        if include_epocs == "all":
            epoc_names = list(self.epocs.keys())
        else:
            epoc_names = [k for k in self.epocs.keys() if k in include_epocs]

        # iterate arrays and include any the user requested
        # also add in any requested metadata
        events = []
        for k, v in self.epocs.items():
            if k in epoc_names:
                for value in v:
                    events.append({**meta, "event": k, "time": value})

        df = pd.DataFrame(events)

        # sort the dataframe by time, but check that we have values, otherwise will raise keyerror
        if len(df.index) > 0:
            df = df.sort_values("time")

        return df

    def scalar_dataframe(self, include_scalars: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with scalar data and metadata.

        Args:
            include_scalars: list of scalar names to include in the dataframe. Special str "all" is also accepted.
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
            DataFrame with data from this session
        """
        # determine metadata fields to include
        if include_meta == "all":
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine scalars to include
        if include_scalars == "all":
            scalar_names = list(self.scalars.keys())
        else:
            scalar_names = [k for k in self.scalars.keys() if k in include_scalars]

        scalars = []
        for sn in scalar_names:
            scalars.append({**meta, "scalar_name": sn, "scalar_value": self.scalars[sn]})

        return pd.DataFrame(scalars)

    def dlc_dataframe(self, id: Union[str, int] = 0) -> pd.DataFrame:
        """Fetch DLC data as a pandas dataframe.
        Args:
            id: identifier to select which dlc data to use in the dataframe. If str is provided, will access that named dlc data. If int is provided, will use the data from that index position among the dlc data.

            By default index 0 will be accessed.

        Returns:
            DataFrame with data from this session
        """
        if isinstance(id, str):
            return pd.DataFrame(self.dlc[id])

        elif isinstance(id, int):
            data_list = list(self.dlc.values())
            return pd.DataFrame(data_list[id])

        else:
            raise TypeError("Invalid `id` argument data type. Supported data identifier types are str and int.")
        
    def analysis_dataframe(self, include_analysis: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with analysis data and metadata.

        Args:
            include_analysis: list of analysis array names to include in the dataframe. Special str "all" is also accepted.
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
            DataFrame with data from this session
        """
        # determine metadata fields to include
        if include_meta == "all":
            meta = self.metadata
        else:
            meta = {k: v for k, v in self.metadata.items() if k in include_meta}

        # determine arrays to include
        if include_analysis == "all":
            analysis_names = list(self.analysis.keys())
        else:
            analysis_names = [k for k in self.analysis.keys() if k in include_analysis]

        # TODO: iterate arrays and include any the user requested
        # also add in any requested metadata
        data = []
        for k, v in self.analysis.items():
            if k in analysis_names:
                if len(v) == 1:
                    for value in v:
                        data.append({**meta, "metric": k, "obs": np.nan, "value": value})
                else:
                    obsn = 1
                    for value in v:
                        data.append({**meta, "metric": k, "obs": obsn,"value": value})
                        obsn += 1

        df = pd.DataFrame(data)

        return df

    def __eq__(self, value: object) -> bool:
        """Test this Session for equality to another Session.

        Args:
            value: value to test against for equality

        Returns:
            True if this Session is equal to value, False otherwise. Name, metadata, signals, epocs, scalars are considered for equality.
        """
        # check we have a session instance to compare to
        if not isinstance(value, Session):
            return False

        # check name for equality
        if self.name != value.name:
            return False

        # check metadata for equality
        if self.metadata.keys() != value.metadata.keys():
            return False
        for meta_key in self.metadata.keys():
            val1 = self.metadata[meta_key]
            val2 = value.metadata[meta_key]
            if isinstance(val1, float) and isinstance(val2, float) and math.isnan(val1) and math.isnan(val2):
                continue  # allow NaN values for the same key to be considered equal
            elif val1 != val2:
                return False

        # check signals for equality
        if self.signals.keys() != value.signals.keys():
            return False
        for k in self.signals.keys():
            if self.signals[k] != value.signals[k]:
                return False

        # check epocs for equality
        if self.epocs.keys() != value.epocs.keys():
            return False
        for k in self.epocs.keys():
            if not np.array_equal(self.epocs[k], value.epocs[k]):
                return False

        # check scalars for equality
        if self.scalars.keys() != value.scalars.keys():
            return False
        for k in self.scalars.keys():
            if not np.array_equal(self.scalars[k], value.scalars[k]):
                return False

        return True

    def _estimate_memory_use_itemized(self) -> dict[str, int]:
        """Estimate the memory use of this Session in bytes, itemized by component.

        Returns:
            Dictionary with keys as component names and values as their estimated memory use in bytes.
        """
        return {
            "self": sys.getsizeof(self),
            "name": sys.getsizeof(self.name),
            "metadata": sum(map(sys.getsizeof, self.metadata.values())) + sum(map(sys.getsizeof, self.metadata.keys())),
            **{f"signal.{sig.name}": sig._estimate_memory_use() for sig in self.signals.values()},
            **{f"epocs.{k}": sys.getsizeof(k) + v.nbytes for k, v in self.epocs.items()},
            **{f"scalars.{k}": sys.getsizeof(k) + v.nbytes for k, v in self.scalars.items()},
            **{f"dlc.{k}": sys.getsizeof(k) + v.nbytes for k, v in self.dlc.items()},
            **{f"analysis.{k}": sys.getsizeof(k) + v.nbytes for k, v in self.analysis.items()},
            **{f"misc.{k}": sys.getsizeof(k) + v.nbytes for k, v in self.misc.items()},
        }

    def _estimate_memory_use(self) -> int:
        """Estimate the total memory use of this Session in bytes."""
        return sum(self._estimate_memory_use_itemized().values())

    def save(self, path: str):
        """Save this Session to a HDF5 file.

        Args:
            path: path where the data should be saved
        """

        with h5py.File(path, mode="w") as h5:
            # save name
            h5.create_dataset("/name", data=self.name)

            # save signatures
            sig_group = h5.create_group("/signatures")
            for k, v in self._signatures.items():
                sig_group.create_dataset(k, data=v)

            # save signals
            h5.create_group("/signals")
            for k, sig in self.signals.items():
                group = h5.create_group(f"/signals/{k}")
                group.create_dataset("signal", data=sig.signal, compression="gzip")
                group.create_dataset("time", data=sig.time, compression="gzip")
                group.attrs["units"] = sig.units
                group.attrs["fs"] = sig.fs
                mark_group = group.create_group("marks")
                for mk, mv in sig.marks.items():
                    mark_group.create_dataset(mk, data=mv)

            # save epocs
            h5.create_group("/epocs")
            for k, epoc in self.epocs.items():
                h5.create_dataset(f"/epocs/{k}", data=np.atleast_1d(epoc), compression="gzip")

            # save scalars
            h5.create_group("/scalars")
            for k, scalar in self.scalars.items():
                h5.create_dataset(f"/scalars/{k}", data=scalar, compression="gzip")

            # save dlc data
            h5.create_group("/dlc")
            for k, dlc in self.dlc.items():
                if isinstance(dlc, pd.DataFrame):
                    dlc.columns = dlc.columns.to_flat_index()

                    for col in dlc.select_dtypes(include='object').columns:
                        dlc[col] = dlc[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, (list, tuple)) else x)

                    for col in dlc.select_dtypes(include='object').columns:
                        dlc[col] = dlc[col].astype(np.bytes_).astype('S50')
                    # string_cols = [col for col in dlc.columns if isinstance(dlc[col].iloc[2], str)]
                    # for col in string_cols:
                    #     dlc[col] = dlc[col].astype(np.bytes_).astype('S50')
                    nparray = dlc.to_records(index=False)
                    h5.create_dataset(f"/dlc/{k}", data=nparray)    
                elif isinstance(dlc, np.ndarray):
                    h5.create_dataset(f"/dlc/{k}", data=dlc)

            # save analysis data
            h5.create_group("/analysis")
            for k, analysis in self.analysis.items():
                h5.create_dataset(f"/analysis/{k}", data=analysis)
           
            # save misc data
            h5.create_group("/misc")
            for k, misc in self.misc.items():
                if isinstance(misc, pd.DataFrame):
                    misc.columns = misc.columns.to_flat_index()

                    for col in dlc.select_dtypes(include='object').columns:
                        dlc[col] = dlc[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, (list, tuple)) else x)

                    for col in dlc.select_dtypes(include='object').columns:
                        dlc[col] = dlc[col].astype(np.bytes_).astype('S50')                  
                    # string_cols = [col for col in dlc.columns if isinstance(dlc[col].iloc[2], str)]
                    # for col in string_cols:
                    #     misc[col] = misc[col].astype(np.bytes_).astype('S50')
                    nparray = misc.to_records(index=False)
                    h5.create_dataset(f"/misc/{k}", data=nparray)   
                else:
                    h5.create_dataset(f"/misc/{k}", data=misc)

            # save metadata
            meta_group = h5.create_group("/metadata")
            for k, v in self.metadata.items():
                if isinstance(v, str):
                    meta_group[k] = v
                    meta_group[k].attrs["type"] = "str"
                elif isinstance(v, bool):
                    meta_group[k] = v
                    meta_group[k].attrs["type"] = "bool"
                elif isinstance(v, int):
                    meta_group[k] = v
                    meta_group[k].attrs["type"] = "int"
                elif isinstance(v, float):
                    meta_group[k] = v
                    meta_group[k].attrs["type"] = "float"
                elif isinstance(v, datetime.datetime):
                    meta_group[k] = v.isoformat()
                    meta_group[k].attrs["type"] = "datetime"
                elif isinstance(v, datetime.timedelta):
                    meta_group[k] = v.total_seconds()
                    meta_group[k].attrs["type"] = "timedelta"
                else:
                    meta_group[k] = v

    @classmethod
    def read_signature(cls, path: str) -> dict[str, str]:
        """Read the signature of a session from an HDF5 file.

        Args:
            path: path to the hdf5 file to read.

        Returns:
            dictionary with the signature of the session.
        """
        signatures = {}
        with h5py.File(path, mode="r") as h5:
            if "/signatures" in h5:
                for sig_name in h5["/signatures"].keys():
                    signatures[sig_name] = h5[f"/signatures/{sig_name}"][()].decode("utf-8")
        return signatures

    @classmethod
    def load(cls, path: str) -> "Session":
        """Load a Session from an HDF5 file.

        Args:
            path: path to the hdf5 file to load

        Returns:
            Session with data loaded
        """
        session = cls()
        with h5py.File(path, mode="r") as h5:
            # read name
            session.name = h5["/name"][()].decode("utf-8")

            # read signatures
            if "/signatures" in h5:
                for sig_name in h5["/signatures"].keys():
                    session._signatures[sig_name] = h5[f"/signatures/{sig_name}"][()].decode("utf-8")

            # read signals
            if "/signals" in h5:
                for signame in h5["/signals"].keys():
                    sig_group = h5[f"/signals/{signame}"]
                    sig = Signal(
                        signame,
                        sig_group["signal"][()],
                        time=sig_group["time"][()],
                        fs=sig_group.attrs["fs"],
                        units=sig_group.attrs["units"],
                    )
                    for mark_name in sig_group["marks"].keys():
                        sig.marks[mark_name] = sig_group[f"marks/{mark_name}"][()]
                    session.add_signal(sig)

            # read epocs
            if "/epocs" in h5:
                for epoc_name in h5["/epocs"].keys():
                    session.epocs[epoc_name] = h5[f"/epocs/{epoc_name}"][()]

            # read scalars
            if "/scalars" in h5:
                for scalar_name in h5["/scalars"].keys():
                    session.scalars[scalar_name] = h5[f"/scalars/{scalar_name}"][()]

            # read dlc
            if "/dlc" in h5:
                for dlc_name in h5["/dlc"].keys():
                    session.dlc[dlc_name] = h5[f"/dlc/{dlc_name}"][()]

            # read analysis
            if "/analysis" in h5:
                for analysis_name in h5["/analysis"].keys():
                    session.analysis[analysis_name] = h5[f"/analysis/{analysis_name}"][()]

            # read misc
            if "/misc" in h5:
                for misc_name in h5["/misc"].keys():
                    session.misc[misc_name] = h5[f"/misc/{misc_name}"][()]

            # read metadata
            if "/metadata" in h5:
                for meta_name in h5[f"/metadata"].keys():
                    meta = h5[f"/metadata/{meta_name}"]
                    if "type" in meta.attrs:
                        mtype = meta.attrs["type"]
                        if mtype == "str":
                            session.metadata[meta_name] = meta[()].decode("utf-8")
                        elif mtype == "bool":
                            session.metadata[meta_name] = bool(meta[()])
                        elif mtype == "int":
                            session.metadata[meta_name] = int(meta[()])
                        elif mtype == "float":
                            session.metadata[meta_name] = float(meta[()])
                        elif mtype == "datetime":
                            session.metadata[meta_name] = datetime.datetime.fromisoformat(meta[()].decode("utf-8"))
                        elif mtype == "timedelta":
                            session.metadata[meta_name] = datetime.timedelta(seconds=meta[()])
                        else:
                            raise ValueError(f'Did not understand type {mtype} for metadata key "{meta_name}"')
                    else:
                        session.metadata[meta_name] = meta[()]

        return session


class SessionCollection(list[Session]):
    """Collection of session data."""

    def __init__(self, *args) -> None:
        """Initialize this `SessionCollection`."""
        super().__init__(*args)
        self.__meta_meta: dict[str, dict[Literal["order"], Any]] = {}

    @property
    def metadata(self) -> pd.DataFrame:
        """Get a dataframe containing metadata across all sessions in this collection."""
        df = pd.DataFrame([item.metadata for item in self])

        for k, v in self.__meta_meta.items():
            if "order" in v:
                df[k] = pd.Categorical(df[k], categories=v["order"], ordered=True)

        return df

    @property
    def metadata_keys(self) -> list[str]:
        """Get a list of the keys present in metadata across all sessions in this collection."""
        return list(set([key for item in self for key in item.metadata.keys()]))

    def add_metadata(self, key: str, value: Any) -> None:
        """Set a metadata field on each session in this collection.

        Args:
            key: name of the metadata field
            value: value for the metadata field
        """
        for item in self:
            item.metadata[key] = value

    def update_metadata(self, meta: dict[str, Any]) -> None:
        """Set multiple metadata fields on each session in this collection.

        Args:
            meta: metadata information to set on each session
        """
        for item in self:
            item.metadata.update(meta)

    def set_metadata_props(self, key: str, order: Optional[list[Any]] = None):
        """Set properties of a metadata column.

        Args:
            key: name of the metadata item, always required
            order: optional, if specified will set the metadata column to be ordered categorical, according to `order`
        """
        assert key in self.metadata_keys

        if key not in self.__meta_meta:
            self.__meta_meta[key] = {}

        if order is not None:
            self.__meta_meta[key]["order"] = order

    def rename_signal(self, old_name: str, new_name: str) -> None:
        """Rename a signal on each session in this collection.

        Args:
            old_name: current name of the signal
            new_name: the new name for the signal
        """
        for item in self:
            item.rename_signal(old_name, new_name)

    def rename_epoc(self, old_name: str, new_name: str) -> None:
        """Rename an epoc on each session in this collection.

        Args:
            old_name: current name of the epoc
            new_name: the new name for the epoc
        """
        for item in self:
            item.rename_epoc(old_name, new_name)

    def rename_scalar(self, old_name: str, new_name: str) -> None:
        """Rename an scalar on each session in this collection.

        Args:
            old_name: current name of the scalar
            new_name: the new name for the scalar
        """
        for item in self:
            item.rename_scalar(old_name, new_name)

    def rename_analysis(self, old_name: str, new_name: str) -> None:
        """Rename an analysis on each session in this collection.

        Args:
            old_name: current name of the analysis
            new_name: the new name for the analysis
        """
        for item in self:
            item.rename_analysis(old_name, new_name)

    def rename_misc(self, old_name: str, new_name: str) -> None:
        """Rename a misc item on each session in this collection.

        Args:
            old_name: current name of the misc item
            new_name: the new name for the misc item
        """
        for item in self:
            item.rename_misc(old_name, new_name)

    def filter(self, predicate: Callable[[Session], bool]) -> "SessionCollection":
        """Filter the items in this collection, returning a new `SessionCollection` containing sessions which pass `predicate`.

        Args:
            predicate: a callable accepting a single session and returning bool.

        Returns:
            a new `SessionCollection` containing only items which pass `predicate`.
        """
        sc = type(self)(item for item in self if predicate(item))
        sc.__meta_meta.update(**copy.deepcopy(self.__meta_meta))
        return sc

    def select(self, *bool_masks: np.ndarray) -> "SessionCollection":
        """Select sessions in this collection, returning a new `SessionCollection` containing sessions which all bool masks are true.

        Args:
            bool_masks: one or more boolean arrays, the reduced logical_and indicating which sessions to select

        Returns:
            a new `SessionCollection` containing only items which pass bool_masks.
        """
        sc = type(self)(item for item, include in zip(self, np.logical_and.reduce(bool_masks)) if include)
        sc.__meta_meta.update(**copy.deepcopy(self.__meta_meta))
        return sc

    def map(self, action: Callable[[Session], Session]) -> "SessionCollection":
        """Apply a function to each session in this collection, returning a new collection with the results.

        Args:
            action: callable accepting a single session and returning a new session

        Returns:
            a new `SessionCollection` containing the results of `action`
        """
        sc = type(self)(action(item) for item in self)
        sc.__meta_meta.update(**copy.deepcopy(self.__meta_meta))
        return sc

    def apply(self, func: Callable[[Session], None]) -> None:
        """Apply a function to each session in this collection.

        Args:
            func: callable accepting a single session and returning None
        """
        for item in self:
            func(item)

    WHAT_LIST = Literal["all", "signal", "epocs", "metadata"]

    @staticmethod
    def merge(
        *session_collections: "SessionCollection", primary_key: str, what: Union[WHAT_LIST, list[WHAT_LIST]], prefixes: list[str]
    ) -> "SessionCollection":
        """Merge session collections while preserving data.

        Args:
            session_collections: SessionCollections to merge
            primary_key: metadata key used to join sessions
            what: the data within each session to merge
            prefixes: list of prefixes, of the same length as the number of passed SessionCollections. each prefix will be prepended to signals to avoid overwriting
        """
        available_whats = ["signal", "epocs", "metadata"]
        use_what: list[str] = []
        if isinstance(what, str):
            if what == "all":
                use_what.extend(available_whats)
            else:
                use_what.append(what)
        else:
            use_what.extend(what)

        sorter: dict[str, list[Session]] = defaultdict(list[Session])
        for collection in session_collections:
            for session in collection:
                sorter[session.metadata[primary_key]].append(session)

        final = SessionCollection()
        for k, v in sorter.items():
            new_session = Session()
            for i, old_session in enumerate(v):
                if "signal" in use_what:
                    for _, sig in old_session.signals.items():
                        new_session.add_signal(sig.copy(f"{prefixes[i]}{sig.name}"))

                if "epocs" in use_what:
                    for name, epocs in old_session.epocs.items():
                        new_session.epocs[name] = epocs

                if "metadata" in use_what:
                    new_session.metadata.update(old_session.metadata)
            final.append(new_session)
        return final

    @property
    def signal_keys(self) -> list[str]:
        """Get a list of Signal keys in this SessionCollection."""
        return list(set([key for item in self for key in item.signals.keys()]))

    def get_signal(self, name: str) -> list[Signal]:
        """Get data across sessions in this collection for the signal named `name`.

        Args:
            name: Name of the signals to collect

        Returns:
            List of Signals, each corresponding to a single session
        """
        return [item.signals[name] for item in self if name in item.signals]
    
    def add_analysis(self, name: str, epoc_func: Callable[[Session], np.ndarray]) -> None:
        """Apply an epoc function to each session in this collection, adding the results to each session's epocs attribute.
        
        Args:
            name: Name of the new analysis data key
            epoc_func: callable accepting a single session and returning a 1d numpy array
        """
        for session in self:
            session.add_epoc(epoc_func(session), name)

    def epoc_dataframe(self, include_epocs: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with epoc data and metadata across all the sessions in this collection.

        Args:
            include_epocs: list of epoc names to include in the dataframe. Special str "all" is also accepted.
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
            DataFrame with data from across this collection
        """
        dfs = [session.epoc_dataframe(include_epocs=include_epocs, include_meta=include_meta) for session in self]
        return pd.concat(dfs).sort_values("time").reset_index(drop=True)

    def scalar_dataframe(self, include_scalars: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with scalar data and metadata across all the sessions in this collection.

        Args:
            include_scalars: list of scalar names to include in the dataframe. Special str "all" is also accepted.
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted.

        Returns:
            DataFrame with data from across this collection
        """
        dfs = [session.scalar_dataframe(include_scalars=include_scalars, include_meta=include_meta) for session in self]
        return pd.concat(dfs).reset_index(drop=True)

    def signal_dataframe(self, signal: str, include_meta: FieldList = "all") -> pd.DataFrame:
        """Get data for a given signal across sessions, also injecting metadata.

        See also: `Signal.to_dataframe()`

        Args:
            signal: Name of the signal to collect
            include_meta: include_meta: metadata fields to include in the final output. Special string "all" will include all metadata fields

        Returns:
            signal across sessions as a `pandas.DataFrame`
        """
        dfs = []
        for session in self:
            if include_meta == "all":
                meta = session.metadata
            else:
                meta = {k: v for k, v in session.metadata.items() if k in include_meta}

            sig = session.signals[signal]
            df = sig.to_dataframe()

            df["obs"] = list(range(sig.nobs))

            for k, v in meta.items():
                df[k] = v

            # reorder columns
            df = df[
                [c for c in df.columns.values if not str(c).startswith("Y.")] + [c for c in df.columns.values if str(c).startswith("Y.")]
            ]

            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
    
    def analysis_dataframe(self, include_analysis: FieldList = "all", include_meta: FieldList = "all") -> pd.DataFrame:
        """Produce a dataframe with analysis data and metadata across all sessions in this collection.
        
        Args:
            include_analysis: list of analysis names to include in the dataframe. Special str "all" is also accepted
            include_meta: list of metadata fields to include in the dataframe. Special str "all" is also accepted
            
        Returns:
            DataFrame with analysis data from across this collection
        """
        dfs = [session.analysis_dataframe(include_analysis=include_analysis, include_meta=include_meta) for session in self]
        return pd.concat(dfs).reset_index(drop=True)
    
    def add_analysis(self, name: str, analysis: Callable[[Session], np.ndarray]) -> None:
        """Apply an analysis function to each session in this collection, adding the results to each session's analysis attribute.
        
        Args:
            name: Name of the new analysis data key
            analysis: callable accepting a single session and returning a 1d numpy array
        """
        for session in self:
            session.add_analysis(analysis(session), name)

    def map(self, action: Callable[[Session], Session]) -> "SessionCollection":
        """Apply a function to each session in this collection, returning a new collection with the results.

        Args:
            action: callable accepting a single session and returning a new session

        Returns:
            a new `SessionCollection` containing the results of `action`
        """
        sc = type(self)(action(item) for item in self)
        sc.__meta_meta.update(**copy.deepcopy(self.__meta_meta))
        return sc

    def aggregate_signals(self, name: str, method: Union[None, str, np.ufunc, Callable[[np.ndarray], np.ndarray]] = "median") -> Signal:
        """Aggregate signals across sessions in this collection for the signal name `name`.

        Args:
            name: name of the signal to aggregate
            method: the method used for aggregation

        Returns:
            Aggregated `Signal`
        """    
        signals = [s for s in self.get_signal(name) if s.nobs > 0]
        if len(signals) <= 0:
            raise ValueError("No signals were passed!")

        # check all signals have the same number of samples
        assert np.all(np.equal([s.nsamples for s in signals], signals[0].nsamples))

        if method is not None:
            signals = [s.aggregate(method) for s in signals]

        s = Signal(signals[0].name, np.vstack([s.signal for s in signals]), time=signals[0].time, units=signals[0].units)
        s.marks.update(signals[0].marks)
        return s

    def describe(self, as_str: bool = False) -> Union[str, None]:
        """Describe this collection of sessions.

        Args:
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

        scalars = Counter([item for session in self for item in session.scalars.keys()])
        buffer += "Scalars present in data with counts:\n"
        for k, v in scalars.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        dlcs = Counter([item for session in self for item in session.dlc.keys()])
        buffer += "DLC Structured Numpy Arrays present in data:\n"
        for k, v in dlcs.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        analysis = Counter([item for session in self for item in session.analysis.keys()])
        buffer += "Analysis arrays present in data:\n"
        for k, v in analysis.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        misc = Counter([item for session in self for item in session.misc.keys()])
        buffer += "Misc items present in data:\n"
        for k, v in misc.items():
            buffer += f'({v}) "{k}"\n'
        buffer += "\n"

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def _estimate_memory_use_itemized(self) -> dict[str, int]:
        """Estimate the memory use of this SessionCollection in bytes, itemized by component.

        Returns:
            Dictionary with keys as component names and values as their estimated memory use in bytes.
        """
        return {s.name: s._estimate_memory_use() for s in self}

    def _estimate_memory_use(self) -> int:
        """Estimate the memory use of this SessionCollection in bytes."""
        return sum(self._estimate_memory_use_itemized().values())

    def save(self, path: str):
        """Save each Session in this SessionCollection to an hdf5 file.

        Each Session will be named as the `Session.name` with an ".h5" file extension

        Args:
            path: path to a directory
        """
        # ensure the directory exists
        os.makedirs(path, exist_ok=True)

        for session in self:
            session.save(os.path.join(path, f"{session.name}.h5"))
