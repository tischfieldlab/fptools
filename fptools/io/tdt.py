import concurrent.futures
import glob
import os
import traceback
from typing import Any, Callable, Literal, Optional, Protocol, TypedDict, cast

import joblib
import pandas as pd
import tdt
from tqdm.auto import tqdm

from .session import Session, SessionCollection, Signal


class SignalMapping(TypedDict):
    tdt_name: str
    dest_name: str
    role: Literal["experimental", "control"]


class Preprocessor(Protocol):
    def __call__(self, session: Session, block: Any, signal_map: list[SignalMapping], **kwargs) -> Session: ...


def load_manifest(path: str) -> pd.DataFrame:
    """Load a manifest file, accepting most common tabular formats.

    Expects to find a column named `blockname`.
    """
    ext = os.path.splitext(path)[1]

    df: pd.DataFrame
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError("did not understand format")

    if "blockname" in df.columns:
        df.set_index("blockname", inplace=True)

    return df


def load_data(
    tank_path: str,
    signal_map: list[SignalMapping],
    manifest_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    preprocess: Optional[Preprocessor] = None,
    cache: bool = True,
    cache_dir: str = "cache",
    **kwargs,
) -> SessionCollection:
    """Load blocks from `tank_path` and return a `SessionCollection`

    Loading will happen in parallel, split across `max_workers` worker processes.

    For quicker future loading, results may be cached. Cacheing is controlled by the `cache` parameter, and the location of cached
    files is controlled by the `cache_dir` parameter.

    You can specify a manifest (in TSV, CSV or XLSX formats) containing additional metadata to be injected into the loaded data.
    This manifest should have at minimum one column with header `blockname` containing each block's name. You may include any other arbitrary
    data columns you may wish. One special column name is `exclude` which should contain boolean `True` or `False`. If a block
    is marked with `True` in this column, then the block will not be loaded or returned in the resulting `SessionCollection`.

    You can also specify a preprocess routine to be applied to each block prior to being returned via the `preprocess` parameter. This
    should be a callable taking a `Session` as the first parameter, a TDT struct object (the return value from calling `tdt.read_block()`)
    as the second parameter, and any additional kwargs necessary. Any **kwargs passed to `load_data()` will be passed to the preprocess callable.
    Your callable preprocess routine should attach any data to the passed `Session` object and return this `Session` object as it's sole return value.
    For example preprocessing routines, please see the implementations in the `tdt.preprocess.pipelines` module.

    Parameters:
    tank_path: path that will be recursively searched for blocks
    manifest_path: if provided, path to metadata in a tabular format, indexed with `blockname`. See above for more details
    max_workers: number of workers in the process pool for loading blocks. If None, defaults to the number of CPUs on the machine.
    preprocess: preprocess routine to run on the data. See above for more details.
    cache: If `True`, results will be cached for future use, or results will be loaded from the cache.
    cache_dir: path to the cache
    **kwargs: additional keyword arguments to pass to the `preprocess` callable.

    Returns:
    `SessionCollection` containing loaded data
    """
    has_manifest = False
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        has_manifest = True

    if cache:
        os.makedirs(cache_dir, exist_ok=True)

    futures: dict[concurrent.futures.Future[Session], str] = {}
    sessions = SessionCollection()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # collect common worker args in one place
        worker_args = {"preprocess": preprocess, "cache": cache, "cache_dir": cache_dir, **kwargs}
        for p in glob.glob(os.path.join(tank_path, "**/*.tbk"), recursive=True):
            block_dir = os.path.dirname(p)
            block_name = os.path.basename(block_dir)

            if has_manifest:
                block_meta = manifest.loc[block_name]
                # possibly exclude the block, if flagged in the manifest
                if "exclude" in block_meta and block_meta["exclude"]:
                    tqdm.write(f"Excluding block {block_name} due to manifest exclude flag")
                    continue

            # submit the task to the pool
            f = executor.submit(__load_block, block_dir, signal_map, **worker_args)
            futures[f] = block_name

        # collect completed tasks
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                s = f.result()
                if has_manifest:
                    s.metadata.update(manifest.loc[futures[f]].to_dict())
                sessions.append(s)
            except Exception as e:
                tqdm.write(
                    f'Encountered problem loading data at "{futures[f]}":\n{traceback.format_exc()}\nData will be omitted from the final collection!\n'
                )
                pass

    return sessions


def __load_block(
    block_path: str,
    signal_map: list[SignalMapping],
    preprocess: Optional[Preprocessor] = None,
    cache: bool = True,
    cache_dir: str = "cache",
    **kwargs,
) -> Session:
    blockname = os.path.basename(block_path)
    cache_path = os.path.join(cache_dir, f"{blockname}.pkl")

    if cache and os.path.exists(cache_path):
        print(f'loading cache: "{cache_path}"')
        return joblib.load(cache_path)

    else:
        block = tdt.read_block(block_path)

        session = Session()
        session.metadata.update(dict(block.info.items()))

        for k in block.epocs.keys():
            session.epocs[k] = block.epocs[k].onset

        preprocess_impl: Preprocessor
        if preprocess is None:
            preprocess_impl = cast(Preprocessor, __default_preprocess)
        else:
            preprocess_impl = preprocess

        session = preprocess_impl(session, block, signal_map, **kwargs)

        if cache:
            joblib.dump(session, cache_path, compress=True)

        return session


def __default_preprocess(session: Session, block: Any, signal_map: list[SignalMapping]) -> Session:
    """A default preprocessing pipeline that doesnt do much of anything.

    Only adds each stream to the session as  a `Signal`
    """
    for sm in signal_map:
        stream = block.streams[sm["tdt_name"]]
        session.add_signal(Signal(sm["dest_name"], stream.data, fs=stream.fs, units="mV"))

    return session
