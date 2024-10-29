import concurrent.futures
import glob
import os
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import tdt
from tqdm.auto import tqdm

from .session import Session, SessionCollection, Signal


def load_manifest(path: str):
    ext = os.path.splitext(path)[1]

    df: pd.DataFrame
    if ext == '.tsv':
        df = pd.read_csv(path, sep='\t')
    elif ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.xlsx':
        df = pd.read_excel(path)
    else:
        raise ValueError('did not understand format')

    if 'blockname' in df.columns:
        df.set_index('blockname', inplace=True)

    return df


def load_data(tank_path: str, manifest_path: str, max_workers: Optional[int] = None, preprocess: Optional[Callable[[Session, Any], Session]] = None, cache: bool = True, cache_dir: str = 'cache', **kwargs) -> SessionCollection:

    manifest = load_manifest(manifest_path)

    if cache:
        os.makedirs(cache_dir, exist_ok=True)

    futures: dict[concurrent.futures.Future[Session], str] = {}
    sessions = SessionCollection()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # collect common worker args in one place
        worker_args = {
            'preprocess': preprocess,
            'cache': cache,
            'cache_dir': cache_dir,
            **kwargs
        }
        for p in glob.glob(os.path.join(tank_path, '**/*.tbk'), recursive=True):
            block_dir = os.path.dirname(p)
            block_name = os.path.basename(block_dir)
            block_meta = manifest.loc[block_name]

            # possibly exclude the block, if flagged in the manifest
            if 'exclude' in block_meta and block_meta['exclude']:
                tqdm.write(f'Excluding block {block_name} due to manifest exclude flag')
                continue

            # submit the task to the pool
            f = executor.submit(__load_block, block_dir, **worker_args)
            futures[f] = block_name

        # collect completed tasks
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                s = f.result()
                s.metadata.update(manifest.loc[futures[f]].to_dict())
                sessions.append(s)
            except Exception as e:
                print(f'Encountered problem loading data at "{futures[f]}":\n{e}\nData will be omitted from the final collection!\n')
                pass

    return sessions



def __load_block(block_path: str, preprocess: Optional[Callable[[Session, Any], Session]] = None, cache: bool = True, cache_dir: str = 'cache', **kwargs) -> Session:
    blockname = os.path.basename(block_path)
    cache_path = os.path.join(cache_dir, f'{blockname}.pkl')

    if cache and os.path.exists(cache_path):
        print(f'loading cache: "{cache_path}"')
        return joblib.load(cache_path)

    else:
        block = tdt.read_block(block_path)

        session = Session()
        session.metadata.update(dict(block.info.items()))

        for k in block.epocs.keys():
            session.epocs[k] = block.epocs[k].onset

        if preprocess is None:
            preprocess = __default_preprocess

        session = preprocess(session, block, **kwargs)

        if cache:
            joblib.dump(session, cache_path, compress=True)

        return session


def __default_preprocess(session: Session, block: Any) -> Session:
    '''A default preprocessing pipeline that doesnt do much of anything.

    Only adds each stream to the session as  a `Signal`
    '''

    for k, v in block.streams:
        session.signals[k] = Signal(k, v.data, fs=v.fs, units='mV')

    return session
