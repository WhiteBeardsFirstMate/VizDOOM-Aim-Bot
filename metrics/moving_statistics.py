import numpy as np

def moving_mean_and_var(x, window=20):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        idx = np.arange(len(x))
        mean = x.copy()
        var = np.zeros_like(x)
        return idx, mean, var
    w = np.ones(window, dtype=np.float32) / window
    mean = np.convolve(x, w, mode="valid")
    mean_sq = np.convolve(x**2, w, mode="valid")
    var = mean_sq - mean**2
    idx = np.arange(window - 1, len(x))
    return idx, mean, var
