import os
import pytest
import numpy as np
from fptools.io import Signal, load_data
from fptools.preprocess.pipelines import lowpass_dff


def test_signal():
    sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)


def test_load_tdt_vanilla(tdt_test_data_path):

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

    assert len(sessions) == 16

def test_load_tdt_preprocess(tdt_test_data_path):

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
                     cache=False)

    assert len(sessions) == 16


def test_load_med_associates_vanilla(ma_test_data_path):
    signal_map = []

    sessions = load_data(ma_test_data_path,
                     signal_map,
                     os.path.join(ma_test_data_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="ma",
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 96
