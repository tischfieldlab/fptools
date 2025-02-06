import concurrent.futures
import glob
import os
import traceback
from typing import Any, Literal, Optional, Protocol, TypedDict, cast

import joblib
import pandas as pd
import tdt
from tqdm.auto import tqdm


from .common import SignalMapping, DataTypeAdaptor

from .session import Session, SessionCollection, Signal


def find_tdt_blocks(path: str) -> list[DataTypeAdaptor]:
    tbk_files = glob.glob(os.path.join(path, "**/*.Tbk"), recursive=True)
    items_out = []
    for tbk in tbk_files:
        adapt = DataTypeAdaptor()
        adapt.path = os.path.dirname(tbk)  # the directory for the block
        adapt.name = os.path.basename(adapt.path)  # the name of the directory
        adapt.loader = load_tdt_block
        items_out.append(adapt)
    return items_out


def load_tdt_block(path: str) -> Session:
    block = tdt.read_block(path)

    session = Session()
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
        session.scalars[k] = block.scalars[k]

    return session
