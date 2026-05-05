from dataclasses import dataclass

@dataclass
class Config:

    # ── Hardware ──────────────────────────────────────────────────────────────
    srate_hz:     float = 10.0
    process:      bool  = True

    # ── Filter ────────────────────────────────────────────────────────────────
    filter_type:  str   = "b"
    # bandpass
    low:          float = 0.05
    high:         float = 0.80
    order:        int   = 1
    # lowpass (one-euro)
    min_cutoff:   float = 1.0
    beta:         float = 0.0
    d_cutoff:     float = 1.0

    # ── Derivative ────────────────────────────────────────────────────────────
    deriv_tightness: int = 7   # ODD

    # ── Calibration ───────────────────────────────────────────────────────────
    initial_calibration_delay_s: float = 60.0
    recalibration_interval_s:    float = 30.0

    # ── Inhalation scoring ────────────────────────────────────────────────────
    """
    Determined by scoring NK2 annotated samples as a baseline, weight tuning
    to match as close as possible + a strong delay penalty
    """
    w1i:             float = 4.9   # departure slope weight
    w2i:             float = 1.4   # timing channel weight
    w3i:             float = 4.9   # prominence weight
    cutoff_inh:      float = 4.2   # minimum score to register detection
    floor_slope_inh: float = 0.05  # minimum slope z-score before weighted sum
    floor_timing_inh: float = 0.05 # minimum timing geometric mean before weighted sum

    # ── Exhalation scoring ────────────────────────────────────────────────────
    w1e:             float = 4.5
    w2e:             float = 0.5
    w3e:             float = 1.0
    cutoff_exh:      float = 3.7
    floor_slope_exh: float = 0.05
    floor_timing_exh: float = 0.05
 
