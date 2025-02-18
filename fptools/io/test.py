import glob
import os
from typing import Optional
import zipfile

import requests
from tqdm.auto import tqdm

def _get_default_test_data_location() -> str:
    """Get the default user test data location, in the users home directory.

    Returns:
        string path to the default user test data location.
    """
    return os.path.join(os.path.expanduser("~"), "fptools_test_data")

def download_test_data(dest: Optional[str] = None) -> str:
    """Download test data to a directory.

    Args:
        dest: path to a directory where test data should be saved to. If None, defaults to a folder in the current users home directory named "fptools_test_data".

    Returns:
        path to a directory containing the test data
    """
    # this is the URL to a zip file containing the test data
    test_data_link = r"https://rutgers.box.com/shared/static/pd6pl4ieo9je5ahh2f22z3t0yssxjern.zip"

    if dest is None:
        dest = _get_default_test_data_location()

    # ensure the destination directory exists
    os.makedirs(dest, exist_ok=True)

    # check if the data appears to be already in place
    if len(glob.glob(os.path.join(dest, "*"))) <= 0:
        print(f'Downloading test data and unpacking to "{dest}"')

        try:
            # location where the zip file will be saved
            zip_path = os.path.join(dest, 'test_data.zip')

            # initiate the request for the data file, check status and calculate the size
            response = requests.get(test_data_link, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            # commence with downloading the file, providing progress feedback along the way
            with open(zip_path, 'wb') as zip_file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    zip_file.write(data)

            # now unpack the zip file, providing progress feedback along the way
            with zipfile.ZipFile(zip_path, 'r') as zip_ref, tqdm(
                desc="Extracting",
                total=len(zip_ref.namelist()),
                unit="file",
            ) as bar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, path=dest)
                    bar.update(1)

            # finally, remove the zip file
            os.remove(zip_path)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except zipfile.BadZipFile as e:
            print(f"Zip file error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print(f'Test data appears to already be in place at "{dest}".')

    return dest

def list_datasets(path: Optional[str] = None) -> list[str]:
    """List the datasets on path.

    Args:
        path: the path to look on for datasets. If None, uses the default test data location.

    Returns:
        list of dataset names.
    """
    if path is None:
        path = _get_default_test_data_location()

    return [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
