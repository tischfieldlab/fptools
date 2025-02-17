import glob
import io
import os
import zipfile

import requests


def download_test_data(dest: str) -> str:
    """Download test data to a directory

    Args:
        dest: path to a directory where test data should be saved to

    Returns:
        path to a directory containing the test data
    """
    # this is the URL to a zip file containing the test data
    test_data_link = r"https://rutgers.box.com/shared/static/pd6pl4ieo9je5ahh2f22z3t0yssxjern.zip"

    # ensure the directory exists
    os.makedirs(dest, exist_ok=True)

    # check if the data appears to be already in place
    if len(glob.glob(os.path.join(dest, "*"))) <= 0:
        print(f'Downloading test data and unpacking to "{dest}"')
        r = requests.get(test_data_link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dest)
    else:
        print("Test data appears to already be in place")

    return dest
