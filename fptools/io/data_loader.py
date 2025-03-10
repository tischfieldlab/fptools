import multiprocessing
import os
import traceback
from typing import Literal, Optional, Union
import concurrent
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from fptools.preprocess.common import Processor

from .common import DataLocator, DataTypeAdaptor
from .med_associates import find_ma_blocks
from .tdt import find_tdt_blocks
from .session import Session, SessionCollection
from tqdm.auto import tqdm


def load_manifest(path: str, index: Optional[str] = None) -> pd.DataFrame:
    """Load a manifest file, accepting most common tabular formats.

    *.tsv (tab-separated values), *.csv (comma-separated values), or *.xlsx (Excel) file extensions are supported.

    Optionally index the dataframe with one of the loaded columns.

    Args:
        path: path to the file to load
        index: if not None, set this column to be the index of the DataFrame

    Returns:
        pandas.DataFrame containing the manifest data.
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
        raise ValueError(
            f'Did not understand manifest format. Supported file extensions are *.tsv (tab-separated), *.csv (comma-separated), or *.xlsx (Excel), but got "{ext}"'
        )

    if index is not None:
        if index in df.columns:
            df.set_index(index, inplace=True)
        else:
            raise ValueError(f"Cannot set manifest index to column {index}; available columns: {df.columns.values}")

    return df


def load_data(
    tank_path: str,
    manifest_path: Optional[str] = None,
    manifest_index: str = "blockname",
    max_workers: Optional[int] = None,
    locator: Union[Literal["auto", "tdt", "ma"], DataLocator] = "auto",
    preprocess: Optional[Processor] = None,
    cache: bool = True,
    cache_dir: str = "cache",
) -> SessionCollection:
    """Load blocks from `tank_path` and return a `SessionCollection`.

    Loading will happen in parallel, split across `max_workers` worker processes.

    For quicker future loading, results may be cached. Caching is controlled by the `cache` parameter, and the location of cached
    files is controlled by the `cache_dir` parameter.

    You can specify a manifest (in TSV, CSV or XLSX formats) containing additional metadata to be injected into the loaded data.
    This manifest should have at minimum one column with header `blockname` containing each block's name. You may include any other arbitrary
    data columns you may wish. One special column name is `exclude` which should contain boolean `True` or `False`. If a block
    is marked with `True` in this column, then the block will not be loaded or returned in the resulting `SessionCollection`.

    You can also specify a preprocess routine to be applied to each block prior to being returned via the `preprocess` parameter. This
    should be a callable taking a `Session` as the first and only parameter. Your callable preprocess routine should attach any data to the passed
    `Session` object and return this `Session` object as it's sole return value.
    For example preprocessing routines, please see the implementations in the `tdt.preprocess.pipelines` module.

    Args:
        tank_path: path that will be recursively searched for blocks
        manifest_path: if provided, path to metadata in a tabular format, indexed with `blockname`. See above for more details
        manifest_index: the name of the column to be set as the manifest DataFrame index.
        max_workers: number of workers in the process pool for loading blocks. If None, defaults to the number of CPUs on the machine.
        locator: locator to use for finding data on `tank_path`
        preprocess: preprocess routine to run on the data. See above for more details.
        cache: If `True`, results will be cached for future use, or results will be loaded from the cache.
        cache_dir: path to the cache

    Returns:
        `SessionCollection` containing loaded data
    """
    has_manifest = False
    if manifest_path is not None:
        manifest = load_manifest(manifest_path, index=manifest_index)
        has_manifest = True

    # if caching is enabled, make sure the cache directory exists
    if cache:
        os.makedirs(cache_dir, exist_ok=True)

    # create a collection to hold the loaded sessions
    sessions = SessionCollection()

    futures: dict[concurrent.futures.Future[Session], str] = {}
    context = multiprocessing.get_context("spawn")
    max_tasks_per_child = 1
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context, max_tasks_per_child=max_tasks_per_child) as executor:
        # collect common worker args in one place
        worker_args = {"preprocess": preprocess, "cache": cache, "cache_dir": cache_dir}

        # iterate over all datasets found by the locator
        for dset in _get_locator(locator)(tank_path):

            # check if we were given a manifest. If so, try to load metadata from the manifest
            # also perform some sanity checks along the way, and check some special flags (ex `exclude`)
            if has_manifest:
                try:
                    block_meta = manifest.loc[dset.name]
                except KeyError:
                    # this block is not listed in the manifest! Err on the side of caution and exclude the block
                    tqdm.write(f'WARNING: Excluding block "{dset.name}" because it is not listed in the manifest!!')
                    continue

                # possibly exclude the block, if flagged in the manifest
                if "exclude" in block_meta and block_meta["exclude"]:
                    tqdm.write(f'Excluding block "{dset.name}" due to manifest exclude flag')
                    continue

            # submit the task to the pool
            f = executor.submit(_load, dset, preprocess=preprocess, cache=cache, cache_dir=cache_dir)
            futures[f] = dset.name

        # collect completed tasks
        for f in tqdm(as_completed(futures), total=len(futures)):
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
    preprocess: Optional[Processor] = None,
    cache: bool = True,
    cache_dir: str = "cache",
) -> Session:
    """Load data for the given DataTypeAdaptor.

    Handles session instance creation, loading across possibly multiple loaders, caching, preprocessing

    Args:
        dset: a DataTypeAdaptor instance describing what data and how to load it
        signal_map: used to inform the preprocessor about data relationships
        preprocess: preprocess routine to run on the data.
        cache: If `True`, results will be cached for future use, or results will be loaded from the cache.
        cache_dir: path to the cache
        **kwargs: additional keyword arguments to pass to the `preprocess` callable.
    """
    cache_path = os.path.join(cache_dir, f"{dset.name}.h5")

    if cache and os.path.exists(cache_path):
        # check if the cached version exists, if so, load and return that
        print(f'loading cache: "{cache_path}"')
        return Session.load(cache_path)

    else:
        # otherwise, we need to load the data from scratch

        # load the data
        session = Session()
        session.name = dset.name
        for loader in dset.loaders:
            session = loader(session, dset.path)

        # preprocess data if preprocessor is specified
        if preprocess is not None:
            session = preprocess(session)

        # cache the session, if requested
        if cache:
            session.save(cache_path)

        return session


def _get_locator(locator: Union[Literal["auto", "tdt", "ma"], DataLocator] = "auto") -> DataLocator:
    """Translate a flexible locator argument to a concrete DataLoader implementation.

    Args:
        locator: type of locator to return. Several special strings supported or a concrete DataLoader implementation.

    Returns:
        A concrete DataLoader
    """
    if locator == "auto":
        return _find_data
    elif locator == "tdt":
        return find_tdt_blocks
    elif locator == "ma":
        return find_ma_blocks
    else:
        return locator


def _find_data(path: str) -> list[DataTypeAdaptor]:
    """Data locator for type "auto".

    Tries to find any type of data (TDT and med-associates currently supported)

    Args:
        path: path to search for data

    Returns:
        list of DataTypeAdaptor, each adaptor corresponding to one session, of data to be loaded
    """
    return [*find_tdt_blocks(path), *find_ma_blocks(path)]
