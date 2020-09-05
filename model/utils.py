import numpy as np


def create_1d_array(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start, stop + step, step)
