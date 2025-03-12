import glob
import os
from typing import Optional

import tdt

from .common import DataTypeAdaptor
from .session import Session, Signal


TDT_EXCLUDE_STREAMS = ["Fi1d", "Fi1r"]
"""List of stream names to exclude from loading.
Used by load_tdt_block().
"""


def find_tdt_blocks(path: str) -> list[DataTypeAdaptor]:
    """Data Locator for TDT blocks.

    Given a path to a directory, will search that path recursively for TDT blocks.

    Args:
        path: path to search for TDT blocks

    Returns:
        list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
    """
    tbk_files = glob.glob(os.path.join(path, "**/*.[tT][bB][kK]"), recursive=True)
    items_out = []
    for tbk in tbk_files:
        adapt = DataTypeAdaptor()
        adapt.path = os.path.dirname(tbk)  # the directory for the block
        adapt.name = os.path.basename(adapt.path)  # the name of the directory
        adapt.loaders.append(TDTLoader(exclude_streams=TDT_EXCLUDE_STREAMS))
        items_out.append(adapt)
    return items_out


class TDTLoader:
    def __init__(
        self,
        exclude_streams: Optional[list[str]] = None,
        exclude_epocs: Optional[list[str]] = None,
        exclude_scalars: Optional[list[str]] = None,
    ) -> None:
        """Initialize this TDTLoader."""
        self.exclude_streams = exclude_streams or []
        self.exclude_epocs = exclude_epocs or []
        self.exclude_scalars = exclude_scalars or []

    def __call__(self, session: Session, path: str) -> Session:
        """Data Loader for TDT blocks.

        Args:
            session: the session for data to be loaded into
            path: path to a TDT block folder

        Returns:
            Session object with data added
        """
        # read the block
        block = tdt.read_block(path)

        # add metadata from the block
        session.metadata.update(dict(block.info.items()))

        # add streams (as signals)
        for k in block.streams.keys():
            if k in self.exclude_streams:
                continue
            stream = block.streams[k]
            session.add_signal(Signal(k, stream.data, fs=stream.fs, units="mV"))

        # add epocs
        for k in block.epocs.keys():
            if k in self.exclude_epocs:
                continue
            session.epocs[k] = block.epocs[k].onset

        # add scalars
        for k in block.scalars.keys():
            if k in self.exclude_scalars:
                continue
            session.scalars[k] = block.scalars[k].ts

        return session
