import multiprocessing
import requests
import zipfile
import io
import os
import glob
import pytest


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
