import glob
import os
from typing import Optional
from pathlib import Path

import pandas as pd

import tdt

from .common import DataTypeAdaptor
from .session import Session, Signal
from .tdt import TDT_EXCLUDE_STREAMS, TDTLoader


def has_neighboring_dlc_h5(tbk) -> bool:
    """Checks if the TBK file has a neighboring H5 file that looks like a DLC data output.

    Args:
        tbk (str): TBK file path to check

    Returns:
        True if the TBK file has DLC neighbors, otherwise false
    """
    dlc_id_substrs = ["DLC", "shuffle", "snapshot"]
    neighbor_files = glob.glob(os.path.join(os.path.dirname(tbk), "*"))
    neighbor_files = [file for file in neighbor_files if file.endswith(".h5")]

    dlc_files = [file for file in neighbor_files if all(sub in file for sub in dlc_id_substrs)]

    if len(dlc_files) > 0:
        return True

    return False


class FindTDTDLCBlocks:

    def __init__(self, model_name: Optional[list[str]] = None, filtered_only: bool = True):
        """Initialize this TDT-DLC Data Locator.

        Args:
            model_name: If provided, only look for DLC files with that model name(s), If None, load all files that look like DLC data
            filtered_only: If true, only load filtered DLC data, otherwise, load any DLC data
        """
        self.model_name = model_name
        self.filtered_only = filtered_only

    def __call__(self, path: str):
        """Data Locator for TDT blocks with DLC data.

        Given a path to a directory, will search that path recursively for TDT blocks that contain DLC output files in .h5 format.

        Args:
            path: path to search for TDT blocks with DLC data

        Returns:
            list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
        """
        tbk_files = glob.glob(os.path.join(path, "**/*.[tT][bB][kK]"), recursive=True)
        # tbk_files = [file for file in tbk_files if has_neighboring_dlc_h5(file)]

        items_out = []
        for tbk in tbk_files:
            adapt = DataTypeAdaptor()
            adapt.path = os.path.dirname(tbk)  # the directory for the block
            adapt.name = os.path.basename(adapt.path)  # the name of the directory
            adapt.loaders.append(TDTLoader(exclude_streams=TDT_EXCLUDE_STREAMS))
            adapt.loaders.append(DLCLoader(model_name=self.model_name, filtered_only=self.filtered_only))
            items_out.append(adapt)

        return items_out


# def find_tdt_w_dlc_blocks(path: str) -> list[DataTypeAdaptor]:
#     """Data Locator for TDT blocks with DLC data.

#     Given a path to a directory, will search that path recursively for TDT blocks that contain DLC output files in .h5 format.

#     Args:
#         path: path to search for TDT blocks with DLC data

#     Returns:
#         list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
#     """

#     tbk_files = glob.glob(os.path.join(path, "**/*.[tT][bB][kK]"), recursive=True)
#     tbk_files = [file for file in tbk_files if has_neighboring_dlc_h5(file)]

#     items_out = []
#     for tbk in tbk_files:
#         adapt = DataTypeAdaptor()
#         adapt.path = os.path.dirname(tbk)  # the directory for the block
#         adapt.name = os.path.basename(adapt.path)  # the name of the directory
#         adapt.loaders.append(TDTLoader(exclude_streams=TDT_EXCLUDE_STREAMS))
#         adapt.loaders.append(DLCLoader())
#         items_out.append(adapt)

#     return items_out


class DLCLoader:
    def __init__(self, model_name: Optional[list[str]] = None, filtered_only: bool = True) -> None:
        """Initialize this DLCLoader."""
        self.model_name = model_name
        self.filtered_only = filtered_only

    def __call__(self, session: Session, path: str) -> Session:
        """Data Loader for DLC .h5 files in TDT blocks.

        Args:
            session: the session for data to be loaded into
            path: path to a TDT block folder containing DLC data

        Returns:
            Session object with data added
        """
        # find the dlc .h5 file
        if self.model_name is None:
            pattern = os.path.join(path, f"*DLC*.h5")
            files = glob.glob(pattern)

            if len(files) <= 0:
                raise FileNotFoundError(f"Could not find any DLC files in block {session.name}!")

            for file in files:
                df = pd.read_hdf(file)
                key = f"{df.columns[0][0]}"
                df.columns = df.columns.droplevel(level=0).to_flat_index()
                nparray = df.to_records(index=False)

                if "_filtered" in file:
                    key += "_filtered"
                    session.dlc[key] = nparray
                else:
                    if not self.filtered_only:
                        key += "_unfiltered"
                        session.dlc[key] = nparray
        else:
            for mn in self.model_name:
                if self.filtered_only:
                    pattern = os.path.join(path, f"*{mn}*_filtered.h5")
                else:
                    pattern = os.path.join(path, f"*{mn}*.h5")
                files = glob.glob(pattern)
                if len(files) <= 0:
                    raise FileNotFoundError(f'Could not find any DLC files for model "{mn}" in block {session.name}!')

                for file in files:
                    df = pd.read_hdf(file)
                    df.columns = df.columns.droplevel(level=0).to_flat_index()
                    nparray = df.to_records(index=False)
                    key = f"{mn}"
                    if "_filtered" in file:
                        key += "_filtered"
                    else:
                        key += "_unfiltered"
                    session.dlc[key] = nparray

        return session
