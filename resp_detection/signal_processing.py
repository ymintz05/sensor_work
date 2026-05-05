import math
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

# ================================
# lowpass filter
# ================================

def smoothing_factor(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
        return a * x + (1 - a) * x_prev

class lowpass:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
    
# ================================
# bandpass filter
# ================================

class bandpass:
    def __call__(self, x_new):
        x = np.array([x_new], dtype=float)
        y, self.state = sosfilt(self.sos, x, zi=self.state)
        return y[0]

    def filter_chunk(self, x_chunk):
        x = np.asarray(x_chunk, dtype=float)
        y, self.state = sosfilt(self.sos, x, zi=self.state)
        return y

    def reset(self):
        self.state = self.zi * 0.0
    
    def __init__(self, fs, low=0.05, high=0.7, order=2):
        
        self.fs = fs
        self.low = low
        self.high = high
        self.order = order

        
        self.sos = butter(
            N=self.order,
            Wn=[self.low, self.high],
            btype="bandpass",
            fs=self.fs,
            output="sos"
        )

        # Initialize state 
        self.zi = sosfilt_zi(self.sos)
        self.state = self.zi * 0.0
