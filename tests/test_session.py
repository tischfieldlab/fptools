import pytest
import numpy as np
from fptools.io import Signal

def test_signal():
    sig = Signal("sig1", np.sin(np.arange(1000)), fs=1)
