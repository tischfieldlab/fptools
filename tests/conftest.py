import multiprocessing
import matplotlib
import os
from typing import Literal, cast
import pytest

from fptools.io.data_loader import load_data, SessionCollection
from fptools.io.test import download_test_data
from fptools.preprocess.pipelines import LowpassDFFPipeline


def pytest_configure(config):
    # Force matplotlib to use the Agg backend so that it doesn't try to open windows during testing
    matplotlib.use("Agg")


@pytest.fixture(scope="session", autouse=True)
def always_spawn() -> None:
    """Force multiprocessing to always use the `spawn` start method.

    scope == 'session' so that it only runs once during testing.
    autouse == True so that this fixture automatically runs at the start of testing.

    This fixes some issues when running pytest and testing functions that use multiprocessing.
    """
    multiprocessing.set_start_method("spawn", force=True)

def _get_test_data_location() -> str:
    """Get the path to the location for testing data.

    Returns:
        path to the test data location
    """
    return os.path.join(os.getcwd(), 'test_data')

def list_files(start_path) -> None:
    """Helper method to print a directory tree with formatting.

    Args:
        start_path: root path to traverse
    """
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(sub_indent, f))

def pytest_sessionstart(session) -> None:
    """Ensure test data is downloaded and extracted at the start of testing.

    Data will be downloaded as a zip file hosted on Box, and then unpacked to the local `test_data` folder
    in the root of the package.
    """
    dest = download_test_data(_get_test_data_location())

    print()
    print(f"Here are the contents of the test data folder \"{dest}\":")
    list_files(dest)


@pytest.fixture
def tdt_test_data_path() -> str:
    """Get a path to a directory with TDT test data."""
    return os.path.join(_get_test_data_location(), 'TDT-DLS-GRABDA2m-Male-PR4-2Day')


@pytest.fixture
def ma_test_data_path() -> str:
    """Get a path to a directory with Med-Associates test data."""
    return os.path.join(_get_test_data_location(), 'MA-PR4-4Day')

@pytest.fixture
def tdt_preprocessed_sessions(tdt_test_data_path) -> SessionCollection:
    """Fixture to provide some preprocessed TDT test data.

    lowpass_dff preprocessor will be run on the data.

    show_steps should be false, can increase memory consumption during tests, causing actions to fail

    Args:
        tdt_test_data_path: fixture that provides a file system path to the TDT test data
    """
    rename_map: dict[Literal["signals", "epocs", "scalars"], dict[str, str]] = {
        "signals": {
            '_465A': 'Dopamine',
            '_415A': 'Isosbestic',
        },
        "epocs": {
            "RNP_": "RNP",
            "RMG_": "RMG",
            "URM_": "URM",
        }
    }

    sessions = load_data(tdt_test_data_path,
                     manifest_path=os.path.join(tdt_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="tdt",
                     preprocess=LowpassDFFPipeline(
                         signals=['_465A', '_415A'],
                         rename_map=rename_map,
                         trim_begin="auto",
                         downsample=10,
                         plot=False,
                     ),
                     cache=True,
                     cache_dir=os.path.join(tdt_test_data_path, 'cache_lowpass_dff'))

    return sessions


@pytest.fixture
def ma_preprocessed_sessions(ma_test_data_path) -> SessionCollection:
    """Fixture to provide some Med-Associates test data.

    Args:
        ma_test_data_path: fixture that provides a file system path to the med associates test data
    """
    sessions = load_data(ma_test_data_path,
                     manifest_path=os.path.join(ma_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="ma",
                     preprocess=None,
                     cache=False)

    sessions.rename_epoc('B', 'nosepoke')
    sessions.rename_epoc('C', 'magazine_entry')
    sessions.rename_epoc('D', 'reward_retrieval_latency')
    sessions.rename_epoc('F', 'rewarded_nosepoke')

    return sessions
