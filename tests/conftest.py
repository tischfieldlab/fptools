import multiprocessing
import requests
import zipfile
import io
import os
import glob
import pytest

from fptools.io.data_loader import load_data
from fptools.preprocess.pipelines import lowpass_dff


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    """Force multiprocessing to always use the `spawn` start method.

    Scope == 'session' so that it only runs once during testing.
    autouse == True so that this fixture automatically runs at the start of testing.
    """
    multiprocessing.set_start_method("spawn", force=True)


def list_files(startpath):
    """Helper method to print a directory tree with formatting.

    Args:
        startpath: root path to traverse
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def pytest_sessionstart(session):
    """Ensure test data is downloaded and extracted at the start of testing.

    Data will be downloaded as a zip file hosted on Box, and then unpacked to the local `test_data` folder
    in the root of the package.
    """
    test_data_link = r"https://rutgers.box.com/shared/static/pd6pl4ieo9je5ahh2f22z3t0yssxjern.zip"
    dest = os.path.join(os.getcwd(), 'test_data')

    # check if the data appears to be already in place
    if len(glob.glob(os.path.join(dest, "*"))) <= 0:
        print(f"Downloading test data and unpacking to \"{dest}\"")
        r = requests.get(test_data_link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dest)
    else:
        print("Test data appears to already be in place")

    print()
    print(f"Here are the contents of the test data folder \"{dest}\":")
    list_files(dest)


@pytest.fixture
def tdt_test_data_path() -> str:
    """Get a path to a directory with TDT test data."""
    return os.path.join(os.getcwd(), 'test_data', 'TDT-DLS-GRABDA2m-Male-PR4-2Day')


@pytest.fixture
def ma_test_data_path() -> str:
    """Get a path to a directory with Med-Associates test data."""
    return os.path.join(os.getcwd(), 'test_data', 'MA-PR4-4Day')

@pytest.fixture
def tdt_preprocessed_sessions(tdt_test_data_path):
    """Fixture to provide some preprocessed TDT test data.

    lowpass_dff preprocessor will be run on the data.

    show_steps should be false, can increase memory consumption during tests, causing actions to fail

    Args:
        tdt_test_data_path: fixture that provides a file system path to the TDT test data
    """
    signal_map = [{
        'tdt_name': '_465A',
        'dest_name': 'Dopamine',
        'role': 'experimental'
    }, {
        'tdt_name': '_415A',
        'dest_name': 'Isosbestic',
        'role': 'control'
    }]

    sessions = load_data(tdt_test_data_path,
                     signal_map,
                     os.path.join(tdt_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="tdt",
                     preprocess=lowpass_dff,
                     cache=True,
                     cache_dir=os.path.join(tdt_test_data_path, 'cache_lowpass_dff'),
                     show_steps=False)

    return sessions


@pytest.fixture
def ma_preprocessed_sessions(ma_test_data_path):
    """Fixture to provide some Med-Associates test data.

    Args:
        ma_test_data_path: fixture that provides a file system path to the med associates test data
    """
    signal_map = []

    sessions = load_data(ma_test_data_path,
                     signal_map,
                     os.path.join(ma_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="ma",
                     preprocess=None,
                     cache=False)

    sessions.rename_epoc('B', 'nosepoke')
    sessions.rename_epoc('C', 'magazine_entry')
    sessions.rename_epoc('D', 'reward_retrieval_latency')
    sessions.rename_epoc('F', 'rewarded_nosepoke')

    return sessions
