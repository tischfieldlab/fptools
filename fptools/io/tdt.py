import glob
import os

import tdt

from .common import DataTypeAdaptor
from .session import Session, Signal


def find_tdt_blocks(path: str) -> list[DataTypeAdaptor]:
    """Data Loactor for TDT blocks.

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
        adapt.loaders.append(load_tdt_block)
        items_out.append(adapt)
    return items_out


def load_tdt_block(session: Session, path: str) -> Session:
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
        stream = block.streams[k]
        session.add_signal(Signal(k, stream.data, fs=stream.fs, units="mV"))

    # add epocs
    for k in block.epocs.keys():
        session.epocs[k] = block.epocs[k].onset

    # add scalars
    for k in block.scalars.keys():
        session.scalars[k] = block.scalars[k].ts

    return session
