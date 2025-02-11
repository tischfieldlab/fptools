



import requests
import zipfile
import io
import os
import glob


def pytest_sessionstart(session):
    ''' Ensure test data is downloaded and extracted.
    '''

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
