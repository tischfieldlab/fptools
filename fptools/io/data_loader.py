import glob
import os
import traceback
from typing import Any, Optional, Protocol, cast
import concurrent
import joblib
import pandas as pd

from .common import Loader, SignalMapping, DataTypeAdaptor
from .med_associates import find_ma_blocks, parse_ma_session
from .tdt import find_tdt_blocks, load_tdt_block
from .session import Session, SessionCollection, Signal
from tqdm.auto import tqdm


class Preprocessor(Protocol):
    def __call__(self, session: "Session", block: Any, signal_map: list[SignalMapping], **kwargs) -> "Session":
        """Preprocessor protocol.

        Args:
            session: the session to operate upon
            block: data, result of `tdt.load_block()`
            signal_map: mapping of signal information
            **kwargs: additional kwargs a preprocessor might need

        Returns:
            Session object with preprocessed data added.
        """
        pass


def load_manifest(path: str, index: Optional[str] = None) -> pd.DataFrame:
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

    if index is not None:
        if index in df.columns:
            df.set_index(index, inplace=True)
        else:
            ValueError(f"Cannot set manifest index to column {index}; available columns: {df.columns.values}")

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
    """Load blocks from `tank_path` and return a `SessionCollection`.

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

    Args:
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
        for dset in _find_data(tank_path):

            if has_manifest:
                try:
                    block_meta = manifest.loc[dset.name]
                except KeyError:
                    # this block is not listed in the manifest! Err on the side of caution and exclude the block
                    tqdm.write(f"WARNING: Excluding block {dset.name} because it is not listed in the manifest!!")
                    continue

                # possibly exclude the block, if flagged in the manifest
                if "exclude" in block_meta and block_meta["exclude"]:
                    tqdm.write(f"Excluding block {dset.name} due to manifest exclude flag")
                    continue

            # submit the task to the pool
            f = executor.submit(_load, dset, signal_map, **worker_args)
            futures[f] = dset.name

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


def _load(
    dset: DataTypeAdaptor,
    signal_map: list[SignalMapping],
    preprocess: Optional[Preprocessor] = None,
    cache: bool = True,
    cache_dir: str = "cache",
    **kwargs,
) -> Session:
    cache_path = os.path.join(cache_dir, f"{dset.name}.pkl")

    if cache and os.path.exists(cache_path):
        # check if the cached version exists, if so, load and return that
        print(f'loading cache: "{cache_path}"')
        return joblib.load(cache_path)

    else:
        # otherwise, we need to load the data from scratch

        ## load the data
        session = dset.loader(dset.path)

        if preprocess is not None:
            session = preprocess(session, block, signal_map, **kwargs)

        if cache:
            joblib.dump(session, cache_path, compress=True)

        return session


def _find_data(path: str) -> list[DataTypeAdaptor]:
    """Given a path to some data, attempt to figure out what type of data it is"""
    return [*find_tdt_blocks(path), *find_ma_blocks(path)]
