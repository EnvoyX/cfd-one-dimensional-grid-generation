import numpy as np
import matplotlib.pyplot as plt

# set up the domain and core 1-D Transfinite Interpolation technique (TFI) function
def apply_1d_tfi(x0: float, x1: float, s: np.ndarray) -> np.ndarray:
    """
    maps normalized distribution array s in [0, 1] to physical space [x0, x1].
    :param x0:
    :param x1:
    :param s:
    :return:
    """
    return (1.0 - s) * x0 + s * x1

# point distribution functions
