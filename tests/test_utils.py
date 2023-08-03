import pytest
from cardiomyocyte_emulator import APEmulator, load_default_emulator_model
import torch
import numpy as np

@pytest.fixture(scope="module")
def emulator_init():
    return APEmulator()

def test_init(emulator_init):
    pass

#TODO: Test normalizations... if needed?!