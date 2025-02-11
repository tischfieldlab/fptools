import os
import pytest
import numpy as np
from fptools.io import Signal, load_data
from fptools.preprocess.pipelines import lowpass_dff





def test_load_tdt_vanilla(tdt_test_data_path):
    """Test that we can load some TDT data via the data loader interface.

    Do not perform any preprocessing on the data.

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
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 16, "Expecting 16 sessions to be loaded"

def test_load_tdt_preprocess(tdt_test_data_path):
    """Test that we can load some TDT data via the data loader interface and run preprocessing.

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
                     cache=False,
                     show_steps=False)

    assert len(sessions) == 16


def test_load_med_associates_vanilla(ma_test_data_path):
    """Test that we can load some Med-Associates data via the data loader interface.

    No preprocessing is performed.

    Args:
        ma_test_data_path: fixture that provides a file system path to the med-associates test data
    """
    signal_map = []

    sessions = load_data(ma_test_data_path,
                     signal_map,
                     os.path.join(ma_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="ma",
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 96


def test_load_med_associates_auto_locator(ma_test_data_path):
    """Test that we can load some Med-Associates data via the data loader interface.

    No preprocessing is performed.
    We will use the "auto" locator here, rather than the med associates specific locator

    Args:
        ma_test_data_path: fixture that provides a file system path to the med-associates test data
    """
    signal_map = []

    sessions = load_data(ma_test_data_path,
                     signal_map,
                     os.path.join(ma_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="auto",
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 96
