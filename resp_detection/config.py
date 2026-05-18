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
    deriv_tightness: int = 9   # ODD

    # ── Calibration ───────────────────────────────────────────────────────────
    initial_calibration_delay_s: float = 180.0
    recalibration_interval_s:    float = 30.0

    # ── Inhalation scoring ────────────────────────────────────────────────────
    """
    Determined by scoring NK2 annotated samples as a baseline, weight tuning
    to match as close as possible + a strong delay penalty
    """
    w1i:             float = 3.87926   # departure slope weight
    w2i:             float = 0.655381   # timing channel weight
    w3i:             float = 1.75485   # prominence weight
    cutoff_inh:      float = 0.997461   # minimum score to register detection
    floor_slope_inh: float = 0.006423  # minimum slope z-score before weighted sum
    floor_timing_inh: float = 0.00796803 # minimum timing geometric mean before weighted sum

    # ── Exhalation scoring ────────────────────────────────────────────────────
    w1e:             float = 3.21623
    w2e:             float = 2.8686
    w3e:             float = 0.951821
    cutoff_exh:      float = 1.03061
    floor_slope_exh: float = 0.0211272
    floor_timing_exh: float = 0.00295262
 
