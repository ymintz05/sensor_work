#!/usr/bin/env python3
"""
resp_tuning.py  —  edit the USER CONFIG block below, then:
    python resp_tuning.py

Improvements over baseline pipeline:
  1. Shape score normalisation  — curvature scaled by running EMA of confirmed events
  2. Adaptive refractory        — floor set to fraction of calibrated breath interval
  3. Rolling recalibration      — looks back 3× the recalib interval for robustness
  4. Artifact gating            — EMA z-score gate suppresses analysis during spikes
  5. Multi-scale derivative     — votes across windows [3, 5, cfg.deriv_tightness];
                                   requires majority → noise robust at low delay
  6. Phase lock override        — releases alternating constraint if expected interval
                                   has elapsed with no detection (handles missed events)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

NPZ_FILES = [
    "vernier_resp_20260416_214835.npz",
    "vernier_resp_20260424_115236.npz",
]

OUT_DIR  = "resp_plots"
SRATE_HZ = None   # set to e.g. 10.0 to override rate stored in files

# ── Stage 1: filter search ───────────────────────────────────────────────────
FILTER_TRIALS     = 150

ONSET_MAE_PENALTY = 0.8
SMOOTHNESS_BONUS  = 0.15

# ── Stage 2: weight/scoring search ──────────────────────────────────────────
WEIGHT_TRIALS       = 350
DERIV_DELAY_PENALTY = 0.5
MAE_PENALTY         = 0.3
# deriv_tightness = 2*MAX_DERIV_HALF+1  →  1→3, 2→5, 3→7
# "lock between 3-6" → use deriv_half 1-3 (tightness 3, 5, 7)
MAX_DERIV_HALF      = 3

# ═══════════════════════════════════════════════════════════════════════════════

import dataclasses
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
})

import neurokit2 as nk
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    srate_hz: float = 10.0

    # Butterworth bandpass — set by Stage 1
    low:   float = 0.10
    high:  float = 0.50
    order: int   = 2

    # Multi-scale derivative: always votes across [3, 5, deriv_tightness]
    # tightness must be ODD. Candidate time comes from this window's midpoint.
    deriv_tightness: int = 3

    # Refractory periods (hard minimum; adaptive floor may raise this further)
    inh_refrac_s: float = 1.5
    exh_refrac_s: float = 1.5

    # Adaptive refractory: effective floor = max(inh_refrac_s, refrac_factor * exp_interval)
    # 0.4 means the refractory is at least 40% of the expected breath cycle time
    refrac_factor: float = 0.4

    # Phase lock override: release alternating constraint if no detection in
    # phase_override_factor * expected_interval seconds (handles missed events)
    phase_override_factor: float = 1.5

    # Calibration — LOCKED
    initial_calibration_delay_s: float = 60.0
    recalibration_interval_s:    float = 30.0

    # Inhalation scoring (trough)
    # score = w1i*shape_norm + w2i*z_inh_interval + w3i*z_exh_to_inh + w4i*z_trough_amp
    # shape_norm = (d_f - d_i) / inh_shape_ema  — dimensionless, ~comparable to z-scores
    w1i:        float = 0.5
    w2i:        float = 1.0
    w3i:        float = 0.8
    w4i:        float = 0.5
    cutoff_inh: float = 0.5

    # Exhalation scoring (peak)
    # score = w1e*shape_norm + w2e*z_exh_interval + w3e*z_inh_to_exh
    w1e:        float = 0.5
    w2e:        float = 1.0
    w3e:        float = 0.8
    cutoff_exh: float = 0.5

    # Shape EMA alpha: how fast shape scale adapts to confirmed events
    # Higher = faster adaptation, lower = more stable estimate
    shape_ema_alpha: float = 0.15

    # Artifact gating: samples whose z-score (vs running signal EMA) exceed this
    # threshold are flagged as artifacts and skip analysis
    artifact_z_thresh: float = 4.0

    match_tolerance_s: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class bandpass:
    def __init__(self, fs, low=0.10, high=0.50, order=2):
        self.sos   = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        self.state = sosfilt_zi(self.sos) * 0.0

    def __call__(self, x_new):
        y, self.state = sosfilt(self.sos, np.array([x_new]), zi=self.state)
        return float(y[0])

    def filter_chunk(self, x):
        y, self.state = sosfilt(self.sos, np.asarray(x, dtype=float), zi=self.state)
        return y


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

_MIN_CALIB_SAMPLES = 300  # ~30 s @ 10 Hz


def _spread(vals):
    q1, q3 = np.percentile(vals, [25, 75])
    return float((q3 - q1) / 1.349)


def _cross_intervals(start_t, end_t):
    ivs = []
    for s in start_t:
        later = end_t[end_t > s]
        if len(later):
            ivs.append(later[0] - s)
    return np.array(ivs)


def _compute_calibration(processed_stream, srate_hz):
    sig = np.asarray(processed_stream, dtype=float)
    if len(sig) < _MIN_CALIB_SAMPLES:
        return None
    try:
        rsp_signals, _ = nk.rsp_process(sig, sampling_rate=srate_hz)
    except Exception:
        return None

    trough_idx = np.flatnonzero(rsp_signals["RSP_Troughs"])
    peak_idx   = np.flatnonzero(rsp_signals["RSP_Peaks"])
    if len(trough_idx) < 2 or len(peak_idx) < 2:
        return None

    times_inh = trough_idx / srate_hz
    times_exh = peak_idx   / srate_hz

    ivs_inh = np.diff(times_inh); exp_inh = np.median(ivs_inh); spr_inh = _spread(ivs_inh)
    ivs_exh = np.diff(times_exh); exp_exh = np.median(ivs_exh); spr_exh = _spread(ivs_exh)

    iv_i2e = _cross_intervals(times_inh, times_exh)
    iv_e2i = _cross_intervals(times_exh, times_inh)
    if len(iv_i2e) < 2 or len(iv_e2i) < 2:
        return None

    exp_i2e = np.median(iv_i2e); spr_i2e = _spread(iv_i2e)
    exp_e2i = np.median(iv_e2i); spr_e2i = _spread(iv_e2i)

    trough_vals    = sig[trough_idx]
    exp_trough_val = np.median(trough_vals)
    spr_trough_val = _spread(trough_vals)

    last_inh = (float(times_inh[-1]), float(sig[trough_idx[-1]]))
    last_exh = (float(times_exh[-1]), float(sig[peak_idx[-1]]))

    return (exp_inh, spr_inh, exp_exh, spr_exh,
            exp_i2e, spr_i2e, exp_e2i, spr_e2i,
            exp_trough_val, spr_trough_val,
            last_inh, last_exh)


def initial_calibration(stream, srate_hz):
    return _compute_calibration(stream, srate_hz)


def rolling_calibration(stream, srate_hz):
    calib = _compute_calibration(stream, srate_hz)
    return calib[:10] if calib is not None else None


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED RECORDER  (all improvements implemented here)
# ═══════════════════════════════════════════════════════════════════════════════

_ARTIFACT_EMA_ALPHA  = 0.02   # ~50-sample window for running mean/var
_ARTIFACT_CLEAR_N    = 3      # consecutive clean samples needed to exit artifact state
_RECALIB_LOOKBACK_X  = 3      # look back this many recalib intervals for rolling calib


class SimulatedRecorder:
    """
    Replays a pre-recorded signal sample-by-sample.

    Improvements vs baseline:
      1. shape_norm    — curvature divided by EMA of confirmed-event magnitudes
      2. adaptive_refrac — refractory floor adapts to live breath rate estimate
      3. wide recalib window — rolls 3× recalib_interval of history
      4. artifact gate — EMA z-score; skips analysis during spikes
      5. multi-scale vote — [3, 5, tightness] windows must agree
      6. phase override — stale phase lock released after expected interval elapses
    """

    def __init__(self, signal, cfg: Config = Config()):
        self.cfg    = cfg
        self.signal = np.asarray(signal, dtype=float)
        self.dt     = 1.0 / cfg.srate_hz

    # ── Public ────────────────────────────────────────────────────────────────
    def run(self, verbose=False):
        self._reset()
        for i, val in enumerate(self.signal):
            self._step(val, i * self.dt, verbose=verbose)
        return dict(
            inh_onset         = self.inh_onset,
            exh_onset         = self.exh_onset,
            samples_processed = list(self.samples_processed),
            timestamps        = list(self.timestamps),
            artifact_flags    = list(self.artifact_flags),
        )

    # ── Init ──────────────────────────────────────────────────────────────────
    def _reset(self):
        self._filt = None
        self.samples = []; self.timestamps = []
        self.samples_processed = []
        self.artifact_flags = []

        self.inh_onset = []; self.exh_onset = []
        self.last_phase = None
        self.calibration_vals = None
        self.initial_calib_done = False
        self.last_recalib_t = None

        # 1. Shape score normalization — EMA of confirmed-event curvature magnitudes
        #    Initialised to 1.0 so the first few events are essentially unscaled
        self.inh_shape_ema: float = 1.0
        self.exh_shape_ema: float = 1.0

        # 4. Artifact gate — EMA mean and variance of raw signal
        self.sig_ema: Optional[float] = None
        self.sig_var_ema: float       = 1.0
        self.in_artifact: bool        = False
        self.artifact_clear_count: int = 0

    # ── Filter ────────────────────────────────────────────────────────────────
    def _apply_filter(self, val):
        if self._filt is None:
            self._filt = bandpass(fs=self.cfg.srate_hz,
                                  low=self.cfg.low, high=self.cfg.high,
                                  order=self.cfg.order)
        return self._filt(val)

    # ── 4. Artifact gate ──────────────────────────────────────────────────────
    def _update_artifact_gate(self, val):
        """Update running signal statistics and set self.in_artifact."""
        a = _ARTIFACT_EMA_ALPHA

        if self.sig_ema is None:
            self.sig_ema = val
            self.sig_var_ema = 1.0
            self.in_artifact = False
            return

        diff = val - self.sig_ema
        self.sig_ema    += a * diff
        self.sig_var_ema = (1 - a) * (self.sig_var_ema + a * diff ** 2)
        sig_std = math.sqrt(max(self.sig_var_ema, 1e-9))
        z       = abs(val - self.sig_ema) / sig_std

        if z > self.cfg.artifact_z_thresh:
            self.in_artifact = True
            self.artifact_clear_count = 0
        elif self.in_artifact:
            self.artifact_clear_count += 1
            if self.artifact_clear_count >= _ARTIFACT_CLEAR_N:
                self.in_artifact = False

    # ── Per-sample step ───────────────────────────────────────────────────────
    def _step(self, val, ts, verbose=False):
        proc = self._apply_filter(val)
        self.samples.append(val)
        self.timestamps.append(ts)
        self.samples_processed.append(proc)
        self._update_artifact_gate(val)
        self.artifact_flags.append(self.in_artifact)

        # Run analysis only when calibrated and not in artifact
        if self.initial_calib_done and not self.in_artifact:
            self._analysis()

        # ── Initial calibration ───────────────────────────────────────────────
        if not self.initial_calib_done and ts >= self.cfg.initial_calibration_delay_s:
            calib = initial_calibration(self.samples_processed, self.cfg.srate_hz)
            if calib is not None:
                *vals, last_inh, last_exh = calib
                self.calibration_vals   = tuple(vals)
                self.inh_onset          = [last_inh]
                self.exh_onset          = [last_exh]
                self.last_phase         = "inh" if last_inh[0] > last_exh[0] else "exh"
                self.initial_calib_done = True
                self.last_recalib_t     = ts
                if verbose:
                    print(f"    [calib] initial  t={ts:.1f}s")

        # ── 3. Rolling recalibration — look back 3× recalib interval ─────────
        elif (self.initial_calib_done
              and (ts - self.last_recalib_t) >= self.cfg.recalibration_interval_s):
            # Use a wider lookback window (3× the interval) for more stable stats
            window_n = int(_RECALIB_LOOKBACK_X
                           * self.cfg.recalibration_interval_s
                           * self.cfg.srate_hz)
            recent = self.samples_processed[-window_n:]
            if recent:
                calib = rolling_calibration(recent, self.cfg.srate_hz)
                if calib is not None:
                    self.calibration_vals = calib
                    self.last_recalib_t   = ts
                    if verbose:
                        print(f"    [calib] rolling  t={ts:.1f}s  "
                              f"(window={window_n/self.cfg.srate_hz:.0f}s)")

    # ── 5. Multi-scale derivative & voting ────────────────────────────────────
    def _derivative_at_scale(self, n):
        """Finite-difference derivative over a window of size n (must be odd)."""
        if len(self.samples_processed) < n:
            return None
        mp   = n // 2
        last = self.samples_processed[-n:]
        d_i  = (last[mp]  - last[0])   / (self.dt * mp)
        d_f  = (last[-1]  - last[mp])  / (self.dt * mp)
        return d_i, d_f

    def _multi_scale_vote(self):
        """
        Compute inflection-point votes across window sizes [3, 5, deriv_tightness].
        Candidate time/value always come from the primary (deriv_tightness) window.

        Returns:
            (d_i, d_f, cand_val, cand_ts, votes_inh, votes_exh, n_scales)
            or None if insufficient data.
        """
        n_primary = self.cfg.deriv_tightness
        primary   = self._derivative_at_scale(n_primary)
        if primary is None:
            return None

        # Candidate is the midpoint of the primary window
        mp       = n_primary // 2
        cand_val = self.samples_processed[-(mp + 1)]
        cand_ts  = self.timestamps[-(mp + 1)]

        # Collect votes from all available window sizes
        scale_set = sorted(set([3, 5, n_primary]))
        votes_inh = 0
        votes_exh = 0
        n_scales  = 0
        for n in scale_set:
            d = self._derivative_at_scale(n)
            if d is not None:
                n_scales += 1
                if d[0] < 0 and d[1] > 0:
                    votes_inh += 1
                if d[0] > 0 and d[1] < 0:
                    votes_exh += 1

        d_i, d_f = primary
        return d_i, d_f, cand_val, cand_ts, votes_inh, votes_exh, n_scales

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _analysis(self):
        if (not self.samples_processed
                or self.calibration_vals is None
                or not self.inh_onset
                or not self.exh_onset):
            return

        vote_result = self._multi_scale_vote()
        if vote_result is None:
            return

        d_i, d_f, cand_val, cand_ts, votes_inh, votes_exh, n_scales = vote_result

        # Majority vote required (at least 2 if ≥2 scales, else 1)
        min_votes = max(1, min(2, n_scales))
        is_inh    = votes_inh >= min_votes
        is_exh    = votes_exh >= min_votes

        if not is_inh and not is_exh:
            return

        (exp_inh, spr_inh, exp_exh, spr_exh,
         exp_i2e, spr_i2e, exp_e2i, spr_e2i,
         exp_trough, spr_trough) = self.calibration_vals

        cfg = self.cfg

        last_inh_ts, _ = self.inh_onset[-1]
        last_exh_ts, _ = self.exh_onset[-1]
        dt_inh = cand_ts - last_inh_ts
        dt_exh = cand_ts - last_exh_ts

        # 2. Adaptive refractory: floor = max(static, refrac_factor * exp_interval)
        eff_inh_refrac = max(cfg.inh_refrac_s, cfg.refrac_factor * exp_inh)
        eff_exh_refrac = max(cfg.exh_refrac_s, cfg.refrac_factor * exp_exh)

        # 6. Phase lock override: if expected interval has elapsed, unlock phase
        #    This handles the case where the opposite phase was missed
        can_inh = self.last_phase in (None, "exh")
        can_exh = self.last_phase in (None, "inh")
        if not can_inh and dt_inh > cfg.phase_override_factor * exp_inh:
            can_inh = True  # too long since last inh — probably missed an exh
        if not can_exh and dt_exh > cfg.phase_override_factor * exp_exh:
            can_exh = True  # too long since last exh — probably missed an inh

        # ── Inhalation ────────────────────────────────────────────────────────
        if is_inh and can_inh and dt_inh >= eff_inh_refrac:
            z_inh    = np.exp(-0.5 * ((dt_inh   - exp_inh)    / (spr_inh    + 1e-6))**2)
            z_cross  = np.exp(-0.5 * ((dt_exh   - exp_e2i)    / (spr_e2i    + 1e-6))**2)
            z_trough = np.exp(-0.5 * ((cand_val - exp_trough) / (spr_trough + 1e-6))**2)

            # 1. Normalised shape score (dimensionless, ~comparable to z-scores)
            raw_shape  = d_f - d_i  # positive at trough
            shape_norm = raw_shape / (self.inh_shape_ema + 1e-6)

            score = (cfg.w1i * shape_norm
                     + cfg.w2i * z_inh
                     + cfg.w3i * z_cross
                     + cfg.w4i * z_trough)

            if score > cfg.cutoff_inh:
                self.inh_onset.append((cand_ts, cand_val))
                self.last_phase = "inh"
                # Update shape EMA with this confirmed event's curvature magnitude
                self.inh_shape_ema = ((1 - cfg.shape_ema_alpha) * self.inh_shape_ema
                                      + cfg.shape_ema_alpha * abs(raw_shape))

        # ── Exhalation ────────────────────────────────────────────────────────
        if is_exh and can_exh and dt_exh >= eff_exh_refrac:
            z_exh   = np.exp(-0.5 * ((dt_exh - exp_exh) / (spr_exh + 1e-6))**2)
            z_cross = np.exp(-0.5 * ((dt_inh - exp_i2e) / (spr_i2e + 1e-6))**2)

            raw_shape  = d_i - d_f  # positive at peak
            shape_norm = raw_shape / (self.exh_shape_ema + 1e-6)

            score = (cfg.w1e * shape_norm
                     + cfg.w2e * z_exh
                     + cfg.w3e * z_cross)

            if score > cfg.cutoff_exh:
                self.exh_onset.append((cand_ts, cand_val))
                self.last_phase = "exh"
                self.exh_shape_ema = ((1 - cfg.shape_ema_alpha) * self.exh_shape_ema
                                      + cfg.shape_ema_alpha * abs(raw_shape))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

PREFERRED_KEYS = ("data", "samples", "signal", "respiration", "rsp", "arr_0")
SRATE_KEYS     = ("srate", "srate_hz", "fs", "sampling_rate")


def _try_array(val):
    try:
        arr = np.asarray(val)
        if arr.dtype == object:
            inner = arr.item()
            if isinstance(inner, dict):
                return None
            arr = np.asarray(inner)
        if arr.dtype == object:
            return None
        flat = arr.flatten()
        return flat.astype(float) if flat.size > 0 else None
    except Exception:
        return None


def load_npz(path, srate_override=None):
    d = np.load(path, allow_pickle=True)
    print(f"  keys: {list(d.keys())}")
    sig = None
    for k in PREFERRED_KEYS:
        if k in d:
            c = _try_array(d[k])
            if c is not None and c.size > 10:
                sig = c
                print(f"  signal  <- '{k}'  ({len(sig)} samples)")
                break
    if sig is None:
        best_k, best = None, None
        for k in d.keys():
            c = _try_array(d[k])
            if c is not None and c.size > 10:
                if best is None or c.size > best.size:
                    best_k, best = k, c
        if best is not None:
            sig = best
            print(f"  signal  <- '{best_k}' (auto, {len(sig)} samples)")
    if sig is None:
        raise ValueError(f"No numeric signal in {path}. Keys: {list(d.keys())}")
    sr = srate_override
    if sr is None:
        for k in SRATE_KEYS:
            if k in d:
                c = _try_array(d[k])
                if c is not None and c.size == 1:
                    sr = float(c[0])
                    print(f"  srate_hz <- '{k}'  ({sr} Hz)")
                    break
    if sr is None:
        sr = 10.0
        print(f"  srate_hz <- not found, defaulting to {sr} Hz")
    return sig, sr


# ═══════════════════════════════════════════════════════════════════════════════
# NK2 BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_nk_baseline(signal, cfg):
    sr  = cfg.srate_hz
    sig = np.asarray(signal, dtype=float)
    f        = bandpass(fs=sr, low=cfg.low, high=cfg.high, order=cfg.order)
    filtered = f.filter_chunk(sig)
    rsp_sig, _ = nk.rsp_process(filtered, sampling_rate=sr)
    peak_idx   = np.flatnonzero(rsp_sig["RSP_Peaks"])
    trough_idx = np.flatnonzero(rsp_sig["RSP_Troughs"])
    return dict(raw=sig, filtered=filtered, rsp_sig=rsp_sig,
                peak_idx=peak_idx, trough_idx=trough_idx,
                peak_times=peak_idx/sr, trough_times=trough_idx/sr)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(detected_t, gt_t, tol_s=1.0):
    det = np.sort(np.asarray(detected_t, dtype=float))
    gt  = np.sort(np.asarray(gt_t,       dtype=float))
    if len(det) == 0 or len(gt) == 0:
        return dict(precision=0., recall=0., f1=0., mae_s=np.nan,
                    tp=0, fp=len(det), fn=len(gt))
    matched_gt, matched_det, errors = set(), set(), []
    for i, d in enumerate(det):
        diffs = np.abs(gt - d)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_s and j not in matched_gt:
            matched_gt.add(j); matched_det.add(i); errors.append(float(diffs[j]))
    tp = len(matched_det); fp = len(det) - tp; fn = len(gt) - tp
    P  = tp / (tp + fp) if tp + fp else 0.
    R  = tp / (tp + fn) if tp + fn else 0.
    F1 = 2 * P * R / (P + R) if P + R else 0.
    return dict(precision=P, recall=R, f1=F1,
                mae_s=float(np.mean(errors)) if errors else np.nan,
                tp=tp, fp=fp, fn=fn)


def evaluate(output, baseline, tol_s=1.0):
    inh_t = np.array([t for t, _ in output["inh_onset"][1:]])
    exh_t = np.array([t for t, _ in output["exh_onset"][1:]])
    inh_m = compute_metrics(inh_t, baseline["trough_times"], tol_s)
    exh_m = compute_metrics(exh_t, baseline["peak_times"],   tol_s)
    return dict(inh=inh_m, exh=exh_m,
                combined_f1=0.5 * inh_m["f1"] + 0.5 * exh_m["f1"])


def run_and_evaluate(cfg, signals, srates, baselines, verbose=False):
    results = {}; f1s = []
    for name, sig in signals.items():
        cfg.srate_hz = srates[name]
        output  = SimulatedRecorder(sig, cfg).run(verbose=verbose)
        metrics = evaluate(output, baselines[name], cfg.match_tolerance_s)
        results[name] = dict(output=output, metrics=metrics)
        f1s.append(metrics["combined_f1"])
        if verbose:
            m = metrics; n_art = sum(output["artifact_flags"])
            print(f"  {name}: F1={m['combined_f1']:.3f} "
                  f"| inh P={m['inh']['precision']:.2f} R={m['inh']['recall']:.2f} "
                  f"mae={m['inh']['mae_s']:.2f}s "
                  f"| exh P={m['exh']['precision']:.2f} R={m['exh']['recall']:.2f} "
                  f"mae={m['exh']['mae_s']:.2f}s "
                  f"| artifact={n_art} samples ({100*n_art/len(output['artifact_flags']):.1f}%)")
    return results, float(np.mean(f1s)) if f1s else 0.


def print_metrics_table(results):
    print(f"\n{'File':<36} {'F1':>6} {'inh-P':>7} {'inh-R':>7} "
          f"{'inh-MAE':>8} {'exh-P':>7} {'exh-R':>7} {'exh-MAE':>8}")
    print("-" * 90)
    for name, r in results.items():
        m = r["metrics"]; i, e = m["inh"], m["exh"]
        mi = f"{i['mae_s']:.3f}" if i['mae_s'] == i['mae_s'] else "  nan"
        me = f"{e['mae_s']:.3f}" if e['mae_s'] == e['mae_s'] else "  nan"
        print(f"  {name:<34} {m['combined_f1']:6.3f} "
              f"{i['precision']:7.3f} {i['recall']:7.3f} {mi:>8} "
              f"{e['precision']:7.3f} {e['recall']:7.3f} {me:>8}")


def report_delay(label, cfg, results):
    sr          = cfg.srate_hz
    deriv_delay = (cfg.deriv_tightness // 2) / sr
    inh_m = [r["metrics"]["inh"]["mae_s"] for r in results.values()
              if r["metrics"]["inh"]["mae_s"] == r["metrics"]["inh"]["mae_s"]]
    exh_m = [r["metrics"]["exh"]["mae_s"] for r in results.values()
              if r["metrics"]["exh"]["mae_s"] == r["metrics"]["exh"]["mae_s"]]
    mi = float(np.mean(inh_m)) if inh_m else float("nan")
    me = float(np.mean(exh_m)) if exh_m else float("nan")
    mm = float(np.mean(inh_m + exh_m)) if (inh_m or exh_m) else float("nan")
    active_scales = sorted(set([3, 5, cfg.deriv_tightness]))
    print(f"  [{label}] Delay breakdown:")
    print(f"    deriv window  : {deriv_delay:.3f} s  "
          f"(tightness={cfg.deriv_tightness}, scales voted={active_scales})")
    print(f"    detection MAE : inh={mi:.3f} s   exh={me:.3f} s   mean={mm:.3f} s")
    print(f"    total est.    : {deriv_delay + mm:.3f} s  (window + jitter)")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FILTER QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def _greedy_timing_mae(times_a, times_b, tol=3.0):
    if len(times_a) == 0 or len(times_b) == 0:
        return tol
    matched, used = [], set()
    for t in np.sort(times_a):
        diffs = np.abs(times_b - t)
        j = int(np.argmin(diffs))
        if diffs[j] <= tol and j not in used:
            matched.append(diffs[j]); used.add(j)
    return float(np.mean(matched)) if matched else tol


def filter_quality(sig, sr, low, high, order):
    sig = np.asarray(sig, dtype=float)
    try:
        f        = bandpass(fs=sr, low=low, high=high, order=order)
        filtered = f.filter_chunk(sig)
    except Exception:
        return -1.0
    corr = float(np.corrcoef(filtered, sig)[0, 1])
    if np.isnan(corr):
        return -1.0
    try:
        raw_nk, _ = nk.rsp_process(sig,      sampling_rate=sr)
        flt_nk, _ = nk.rsp_process(filtered, sampling_rate=sr)
        raw_peaks   = np.flatnonzero(raw_nk["RSP_Peaks"])   / sr
        raw_troughs = np.flatnonzero(raw_nk["RSP_Troughs"]) / sr
        flt_peaks   = np.flatnonzero(flt_nk["RSP_Peaks"])   / sr
        flt_troughs = np.flatnonzero(flt_nk["RSP_Troughs"]) / sr
        peak_mae   = _greedy_timing_mae(raw_peaks,   flt_peaks)
        trough_mae = _greedy_timing_mae(raw_troughs, flt_troughs)
        event_mae  = 0.5 * (peak_mae + trough_mae)
    except Exception:
        event_mae = 3.0
    raw_roughness  = np.std(np.diff(sig))
    filt_roughness = np.std(np.diff(filtered))
    smoothness = max(0.0, 1.0 - filt_roughness / (raw_roughness + 1e-9))
    return float(corr - ONSET_MAE_PENALTY * event_mae + SMOOTHNESS_BONUS * smoothness)


def stage1_filter_optuna(signals, srates):
    print("\n" + "=" * 60)
    print("STAGE 1 — FILTER SEARCH")
    print("=" * 60)
    print()
    print("Literature-informed search bounds")
    print("  RSP belt frequency range: 0.1–0.5 Hz at rest (6–30 bpm)")
    print("  Upper bound 0.8 Hz covers mild exertion (~48 bpm)")
    print("  Lower bound 0.04 Hz removes slow drift without cutting signal")
    print("  Butterworth order 2–4: order 2 = minimal group delay,")
    print("    order 4 = steeper roll-off at cost of ~0.15-0.2 s extra delay")
    print("  Constraining low < 0.25 Hz prevents narrow-notch overfitting")
    print("    (previous unconstrained run found low≈0.49 Hz — almost a notch)")
    print()
    print(f"  Search: low=[0.04,0.25]  high=[0.30,0.80]  order=[1,4]")
    print(f"  Objective: corr(filtered,raw) - {ONSET_MAE_PENALTY}*onset_mae"
          f" + {SMOOTHNESS_BONUS}*smoothness")
    print()

    def objective(trial):
        low   = trial.suggest_float("bp_low",   0.04, 0.25)
        high  = trial.suggest_float("bp_high",  0.30, 0.80)
        order = trial.suggest_int  ("bp_order", 1,    4)
        if high <= low + 0.05:
            return -1.0
        scores = [filter_quality(sig, srates[name], low, high, order)
                  for name, sig in signals.items()]
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))

    def _cb(study, trial):
        if trial.number % 20 == 0 or trial.number < 3:
            best = study.best_value if study.best_trial else 0.
            print(f"  [filter] trial {trial.number:4d}  "
                  f"value={trial.value:.4f}  best={best:.4f}")

    study.optimize(objective, n_trials=FILTER_TRIALS, callbacks=[_cb])
    bp = study.best_params
    print(f"\nBest filter score : {study.best_value:.4f}")
    print(f"  low={bp['bp_low']:.4f} Hz  high={bp['bp_high']:.4f} Hz  "
          f"order={bp['bp_order']}")
    return bp["bp_low"], bp["bp_high"], bp["bp_order"]


def save_filter_comparison(signals, srates, low, high, order, out_dir):
    for name, sig in signals.items():
        sr = srates[name]
        t  = np.arange(len(sig)) / sr
        f  = bandpass(fs=sr, low=low, high=high, order=order)
        filtered = f.filter_chunk(sig)

        fig, axes = plt.subplots(2, 1, figsize=(17, 6), sharex=True)
        fig.suptitle(f"{name}  |  [{low:.3f}–{high:.3f} Hz  order {order}]",
                     fontsize=11, fontweight="bold")
        axes[0].plot(t, sig,      color="#5B9BD5", lw=0.8, alpha=0.6, label="Raw")
        axes[0].plot(t, filtered, color="#1F3864", lw=1.3,            label="Filtered")
        axes[0].set_ylabel("Signal"); axes[0].legend(fontsize=7)
        axes[0].set_title("Raw vs Filtered", fontsize=9)

        residual = sig - filtered
        axes[1].plot(t, residual, color="#C00000", lw=0.7, alpha=0.7)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("Residual (removed jitter)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title(
            "Removed content — should look like random noise, no respiratory waves",
            fontsize=9)

        fname = out_dir / f"{name}_filter_check.png"
        fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
        print(f"  saved -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — WEIGHT / SCORING SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def make_cfg_from_trial(trial, low, high, order):
    cfg = Config()
    cfg.low   = low
    cfg.high  = high
    cfg.order = order

    cfg.deriv_tightness = 2 * trial.suggest_int("deriv_half", 1, MAX_DERIV_HALF) + 1

    cfg.inh_refrac_s = trial.suggest_float("inh_refrac_s", 0.3, 4.0)
    cfg.exh_refrac_s = trial.suggest_float("exh_refrac_s", 0.3, 4.0)

    # New tunable params from algorithmic improvements
    cfg.refrac_factor         = trial.suggest_float("refrac_factor",    0.2,  0.7)
    cfg.phase_override_factor = trial.suggest_float("phase_override",   1.0,  3.0)
    cfg.shape_ema_alpha       = trial.suggest_float("shape_ema_alpha",  0.05, 0.4)
    cfg.artifact_z_thresh     = trial.suggest_float("artifact_z",       2.5,  8.0)

    cfg.w1i        = trial.suggest_float("w1i",        0.0, 5.0)
    cfg.w2i        = trial.suggest_float("w2i",        0.0, 5.0)
    cfg.w3i        = trial.suggest_float("w3i",        0.0, 5.0)
    cfg.w4i        = trial.suggest_float("w4i",        0.0, 5.0)
    cfg.cutoff_inh = trial.suggest_float("cutoff_inh", 0.0, 8.0)
    cfg.w1e        = trial.suggest_float("w1e",        0.0, 5.0)
    cfg.w2e        = trial.suggest_float("w2e",        0.0, 5.0)
    cfg.w3e        = trial.suggest_float("w3e",        0.0, 5.0)
    cfg.cutoff_exh = trial.suggest_float("cutoff_exh", 0.0, 8.0)
    return cfg


def stage2_weight_optuna(signals, srates, baselines, low, high, order):
    print("\n" + "=" * 60)
    print("STAGE 2 — WEIGHT / SCORING SEARCH")
    print(f"  Filter locked: low={low:.4f}  high={high:.4f}  order={order}")
    print(f"  Calibration locked: initial=60 s  recalib=30 s  lookback=90 s")
    print(f"  MAX_DERIV_HALF={MAX_DERIV_HALF} → tightness ∈ {[2*h+1 for h in range(1,MAX_DERIV_HALF+1)]}")
    print(f"  New params in search: refrac_factor, phase_override,")
    print(f"    shape_ema_alpha, artifact_z_thresh")
    print("=" * 60)

    def objective(trial):
        try:
            cfg = make_cfg_from_trial(trial, low, high, order)
            local_bl = {}
            for name, sig in signals.items():
                cfg.srate_hz = srates[name]
                local_bl[name] = get_nk_baseline(sig, cfg)
            results, mean_f1 = run_and_evaluate(cfg, signals, srates, local_bl)
            deriv_delay_s = (cfg.deriv_tightness // 2) / cfg.srate_hz
            all_maes = [r["metrics"][p]["mae_s"]
                        for r in results.values() for p in ("inh", "exh")
                        if r["metrics"][p]["mae_s"] == r["metrics"][p]["mae_s"]]
            mean_mae_s = float(np.mean(all_maes)) if all_maes else 0.0
            return mean_f1 - DERIV_DELAY_PENALTY * deriv_delay_s - MAE_PENALTY * mean_mae_s
        except Exception:
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )

    def _cb(study, trial):
        if trial.number % 10 == 0 or trial.number < 5:
            best = study.best_value if study.best_trial else 0.
            print(f"  [weights] trial {trial.number:4d}  "
                  f"value={trial.value:.4f}  best={best:.4f}")

    study.optimize(objective, n_trials=WEIGHT_TRIALS, callbacks=[_cb])
    best = study.best_trial
    print(f"\nBest penalised score : {best.value:.4f}  (trial #{best.number})")
    print("  (= F1 - delay penalties; raw F1 shown in BEST CONFIG RUN below)")
    return best, study


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_comparison(name, sig, baseline, output, cfg, out_dir, tag=""):
    sr = cfg.srate_hz
    t  = np.arange(len(sig)) / sr
    sp = np.asarray(output["samples_processed"])
    af = np.asarray(output["artifact_flags"], dtype=bool)

    inh_t = np.array([x for x, _ in output["inh_onset"][1:]])
    inh_v = np.array([v for _, v in output["inh_onset"][1:]])
    exh_t = np.array([x for x, _ in output["exh_onset"][1:]])
    exh_v = np.array([v for _, v in output["exh_onset"][1:]])

    fig = plt.figure(figsize=(17, 11))
    gspec = gs.GridSpec(4, 1, hspace=0.48, height_ratios=[2, 2, 2, 0.6])
    fig.suptitle(f"{name}  |  {tag}", fontsize=11, fontweight="bold")

    ax0 = fig.add_subplot(gspec[0])
    ax0.plot(t, sig, color="#5B9BD5", alpha=0.45, lw=0.9, label="Raw")
    ax0.plot(t, sp,  color="#1F3864", lw=1.3, label="Pipeline filtered")
    ax0.set_ylabel("Signal"); ax0.legend(fontsize=7, loc="upper right")
    ax0.set_title("Raw  vs  Pipeline-filtered Signal", fontsize=9)

    ax1 = fig.add_subplot(gspec[1], sharex=ax0)
    ax1.plot(t, baseline["filtered"], color="#404040", lw=1)
    ax1.scatter(baseline["trough_times"], baseline["filtered"][baseline["trough_idx"]],
                color="#2E75B6", s=60, zorder=5, label="NK2 troughs (inh)")
    ax1.scatter(baseline["peak_times"],   baseline["filtered"][baseline["peak_idx"]],
                color="#C00000", s=60, zorder=5, label="NK2 peaks (exh)")
    ax1.set_ylabel("Signal"); ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("NeuroKit2 Ground Truth", fontsize=9)

    ax2 = fig.add_subplot(gspec[2], sharex=ax0)
    ax2.plot(t, sp, color="#808080", lw=0.9, alpha=0.6, label="Pipeline filtered")
    ax2.scatter(baseline["trough_times"], baseline["filtered"][baseline["trough_idx"]],
                color="#2E75B6", s=22, alpha=0.35, label="NK2 troughs")
    ax2.scatter(baseline["peak_times"],   baseline["filtered"][baseline["peak_idx"]],
                color="#C00000", s=22, alpha=0.35, label="NK2 peaks")
    if len(inh_t):
        ax2.scatter(inh_t, inh_v, color="#2E75B6", s=90, marker="^",
                    zorder=6, label="Detected inh")
    if len(exh_t):
        ax2.scatter(exh_t, exh_v, color="#C00000", s=90, marker="v",
                    zorder=6, label="Detected exh")
    ax2.set_ylabel("Signal")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("Detected (^v)  vs  NK2 Ground Truth (dots)", fontsize=9)

    # Artifact flag strip
    ax3 = fig.add_subplot(gspec[3], sharex=ax0)
    ax3.fill_between(t, 0, af.astype(float), color="#FF6B35", alpha=0.8, step="mid")
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(["ok", "art"], fontsize=6)
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Artifact flag", fontsize=8)

    fname = out_dir / f"{name}_{tag.replace(' ', '_').replace('/', '-')}.png"
    fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {fname}")


def save_optuna_plots(study, label, out_dir):
    trial_vals  = [t.value for t in study.trials if t.value is not None]
    if not trial_vals:
        return
    best_so_far = np.maximum.accumulate(trial_vals)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(f"Optuna — {label}", fontsize=11)
    axes[0].plot(trial_vals,  alpha=0.35, color="#5B9BD5", label="Trial score")
    axes[0].plot(best_so_far, color="#C00000", lw=2, label="Best so far")
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Score")
    axes[0].set_title("Optimisation History"); axes[0].legend()

    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys()); imps = list(importances.values())
        axes[1].barh(params[::-1], imps[::-1], color="#5B9BD5")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Parameter Importances (fANOVA)")
    except Exception as exc:
        axes[1].set_visible(False)
        print(f"  importance plot skipped: {exc}")

    fname = out_dir / f"optuna_{label.replace(' ', '_')}.png"
    fig.savefig(fname, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    signals = {}; srates = {}
    for path in NPZ_FILES:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: not found: {path}"); continue
        print(f"\n{p.name}")
        sig, sr = load_npz(str(p), srate_override=SRATE_HZ)
        signals[p.stem] = sig; srates[p.stem] = sr
        print(f"  -> {len(sig)} samples  ({len(sig)/sr:.1f} s  @  {sr} Hz)")
    if not signals:
        print("No files loaded."); sys.exit(1)

    # ── Stage 1: filter ───────────────────────────────────────────────────────
    best_low, best_high, best_order = stage1_filter_optuna(signals, srates)
    save_filter_comparison(signals, srates, best_low, best_high, best_order, out_dir)

    # ── NK2 baselines ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NK2 BASELINES  (best filter)")
    print("=" * 60)
    cfg_filt = Config()
    cfg_filt.low = best_low; cfg_filt.high = best_high; cfg_filt.order = best_order
    baselines = {}
    for name, sig in signals.items():
        cfg_filt.srate_hz = srates[name]
        bl = get_nk_baseline(sig, cfg_filt)
        baselines[name] = bl
        print(f"  {name}: peaks(exh)={len(bl['peak_times'])}  "
              f"troughs(inh)={len(bl['trough_times'])}")

    # ── Default config run ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEFAULT CONFIG RUN  (new pipeline, default weights)")
    print("=" * 60)
    results_default, f1_default = run_and_evaluate(
        cfg_filt, signals, srates, baselines, verbose=True)
    print(f"\nMean combined F1 (default): {f1_default:.4f}")
    print_metrics_table(results_default)
    report_delay("default", cfg_filt, results_default)
    for name, r in results_default.items():
        m = r["metrics"]
        save_comparison(name, signals[name], baselines[name], r["output"],
                        cfg_filt, out_dir,
                        tag=f"default_F1={m['combined_f1']:.3f}")

    # ── Stage 2: weights ──────────────────────────────────────────────────────
    best_trial, weight_study = stage2_weight_optuna(
        signals, srates, baselines, best_low, best_high, best_order)
    save_optuna_plots(weight_study, "stage2_weights", out_dir)

    # ── Best config run ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BEST CONFIG RUN")
    print("=" * 60)
    best_cfg = make_cfg_from_trial(best_trial, best_low, best_high, best_order)
    best_baselines = {}
    for name, sig in signals.items():
        best_cfg.srate_hz = srates[name]
        best_baselines[name] = get_nk_baseline(sig, best_cfg)

    best_results, best_f1 = run_and_evaluate(
        best_cfg, signals, srates, best_baselines, verbose=True)
    print(f"\nMean combined F1 (best): {best_f1:.4f}")
    print_metrics_table(best_results)
    report_delay("best", best_cfg, best_results)
    for name, r in best_results.items():
        m = r["metrics"]
        save_comparison(name, signals[name], best_baselines[name], r["output"],
                        best_cfg, out_dir,
                        tag=f"best_F1={m['combined_f1']:.3f}")

    # ── Final config ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TUNED CONFIG  (copy into config.py / Config())")
    print("=" * 60)
    for f in dataclasses.fields(best_cfg):
        print(f"  {f.name:<34s} = {getattr(best_cfg, f.name)!r}")

    print(f"\nAll plots saved to: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()