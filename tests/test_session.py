import os
import pytest
import numpy as np
from fptools.io import Signal, load_data

def test_signal():
    sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)


def test_load_tdt():
    tank_path = os.path.join(os.getcwd(), 'test_data', 'TDT-DLS-GRABDA2m-Male-PR4-2Day')

    signal_map = [{
        'tdt_name': '_465A',
        'dest_name': 'Dopamine',
        'role': 'experimental'
    }, {
        'tdt_name': '_415A',
        'dest_name': 'Isosbestic',
        'role': 'control'
    }]

    sessions = load_data(tank_path,
                     signal_map,
                     os.path.join(tank_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="tdt",
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 16


def test_load_med_associates():
    tank_path = os.path.join(os.getcwd(), 'test_data', 'MA-PR4-4Day')

    signal_map = []

    sessions = load_data(tank_path,
                     signal_map,
                     os.path.join(tank_path, 'manifest.xlsx'),
                     max_workers=2,
                     locator="ma",
                     preprocess=None,
                     cache=False)

    assert len(sessions) == 96
