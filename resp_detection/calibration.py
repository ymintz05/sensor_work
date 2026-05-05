import neurokit2 as nk
import numpy as np
from config import Config


def _spread(vals):
    
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    return (q3 - q1) / 1.349


def _compute_cross_intervals(start_times, end_times):
    
    intervals = []
    for start_t in start_times:
        later = end_times[end_times > start_t]
        if len(later) > 0:
            intervals.append(later[0] - start_t)
    return np.array(intervals)


def _departure_slopes(sig, event_idx, mp, dt):
    """
    Compute the departure slope d_f at each confirmed event, using the same
    finite-difference arithmetic as the live detector.

    For a confirmed event at sample index k, the live detector would have seen:
        window = sig[k-mp : k+mp+1]  (size 2*mp+1 = deriv_tightness)
        candidate at position mp (i.e. sig[k])
        d_f = (window[-1] - window[mp]) / (dt * mp)
             = (sig[k+mp]  - sig[k])    / (dt * mp)

    Only events with enough look-ahead (k+mp < len(sig)) are included.

    Parameters
    ----------
    sig        : 1-D array, the processed signal
    event_idx  : array of confirmed event indices (troughs or peaks)
    mp         : deriv_tightness // 2
    dt         : 1 / srate_hz

    Returns
    -------
    slopes : 1-D array of d_f values (empty if none fit)
    """
    slopes = []
    for k in event_idx:
        if k + mp < len(sig):
            d_f = (sig[k + mp] - sig[k]) / (dt * mp)
            slopes.append(d_f)
    return np.array(slopes)


def _prominence_values(sig, trough_idx, peak_idx):
    """
    For each trough compute inh prominence = preceding_peak_value - trough_value.
    For each peak  compute exh prominence = peak_value - preceding_trough_value.

    Prominence is always >= 0 when events alternate cleanly.
    Events with no preceding counterpart are skipped.

    Returns
    -------
    inh_prominences, exh_prominences : 1-D arrays
    """
    inh_prom = []
    for ti in trough_idx:
        prior_peaks = peak_idx[peak_idx < ti]
        if len(prior_peaks) > 0:
            inh_prom.append(float(sig[prior_peaks[-1]] - sig[ti]))

    exh_prom = []
    for pi in peak_idx:
        prior_troughs = trough_idx[trough_idx < pi]
        if len(prior_troughs) > 0:
            exh_prom.append(float(sig[pi] - sig[prior_troughs[-1]]))

    return np.array(inh_prom), np.array(exh_prom)


# ─────────────────────────────────────────────────────────────────────────────
# Core calibration
# ─────────────────────────────────────────────────────────────────────────────

def _compute_calibration(raw_stream):
    """
    Run NK2 on raw_stream and extract all statistical priors needed by the
    live detector.

    Calibration tuple layout (18 values + last_inh + last_exh):
        0  exp_inh          median inh-to-inh interval (s)
        1  spr_inh          spread of inh-to-inh intervals
        2  exp_exh          median exh-to-exh interval (s)
        3  spr_exh          spread of exh-to-exh intervals
        4  exp_i2e          median inh-onset → exh-onset interval
        5  spr_i2e
        6  exp_e2i          median exh-onset → inh-onset interval
        7  spr_e2i
        8  exp_slope_inh    median d_f at confirmed troughs (should be > 0)
        9  spr_slope_inh    spread of departure slopes at troughs
       10  exp_slope_exh    median |d_f| at confirmed peaks  (d_f < 0 there,
                            stored as negative; score with -d_f in detector)
       11  spr_slope_exh    spread of departure slopes at peaks
       12  exp_prom_inh     median inh prominence (preceding_peak − trough)
       13  spr_prom_inh     spread of inh prominences
       14  exp_prom_exh     median exh prominence (peak − preceding_trough)
       15  spr_prom_exh     spread of exh prominences
       16  last_inh         (ts, val) of last confirmed trough in window
       17  last_exh         (ts, val) of last confirmed peak  in window
    """
    sig = np.asarray(raw_stream, dtype=float)

    try:
        stream_signals, _ = nk.rsp_process(sig, sampling_rate=Config.srate_hz)
    except Exception:
        return None

    # ── Event indices ─────────────────────────────────────────────────────────
    trough_idx = np.flatnonzero(stream_signals["RSP_Troughs"])   # inhalation onsets
    peak_idx   = np.flatnonzero(stream_signals["RSP_Peaks"])     # exhalation onsets

    if len(trough_idx) < 2 or len(peak_idx) < 2:
        return None

    times_inh = trough_idx / Config.srate_hz
    times_exh = peak_idx   / Config.srate_hz

    # ── Inhalation interval stats ─────────────────────────────────────────────
    ivs_inh      = np.diff(times_inh)
    exp_inh      = np.median(ivs_inh)
    spr_inh      = _spread(ivs_inh)

    # ── Exhalation interval stats ─────────────────────────────────────────────
    ivs_exh      = np.diff(times_exh)
    exp_exh      = np.median(ivs_exh)
    spr_exh      = _spread(ivs_exh)

    # ── Cross-phase timing ────────────────────────────────────────────────────
    iv_i2e = _compute_cross_intervals(times_inh, times_exh)
    iv_e2i = _compute_cross_intervals(times_exh, times_inh)
    if len(iv_i2e) < 2 or len(iv_e2i) < 2:
        return None

    exp_i2e = np.median(iv_i2e);  spr_i2e = _spread(iv_i2e)
    exp_e2i = np.median(iv_e2i);  spr_e2i = _spread(iv_e2i)

    # ── Departure slope stats ─────────────────────────────────────────────────
    # mp mirrors the live detector: candidate sits at deriv_tightness // 2
    mp = Config.deriv_tightness // 2
    dt = 1.0 / Config.srate_hz

    slopes_inh = _departure_slopes(sig, trough_idx, mp, dt)
    slopes_exh = _departure_slopes(sig, peak_idx,   mp, dt)

    # Need at least 2 valid slope measurements to compute spread
    if len(slopes_inh) < 2 or len(slopes_exh) < 2:
        return None

    # At a trough d_f should be positive (signal rising); store raw value.
    # At a peak   d_f should be negative (signal falling); store raw value
    # (detector will use -d_f so it can apply the same one-sided zero floor).
    exp_slope_inh = np.median(slopes_inh);  spr_slope_inh = _spread(slopes_inh)
    exp_slope_exh = np.median(slopes_exh);  spr_slope_exh = _spread(slopes_exh)

    # ── Prominence stats ──────────────────────────────────────────────────────
    inh_prom, exh_prom = _prominence_values(sig, trough_idx, peak_idx)

    if len(inh_prom) < 2 or len(exh_prom) < 2:
        return None

    exp_prom_inh = np.median(inh_prom);  spr_prom_inh = _spread(inh_prom)
    exp_prom_exh = np.median(exh_prom);  spr_prom_exh = _spread(exh_prom)

    # ── Seed events ───────────────────────────────────────────────────────────
    last_inh = (float(times_inh[-1]), float(sig[trough_idx[-1]]))
    last_exh = (float(times_exh[-1]), float(sig[peak_idx[-1]]))

    return (
        exp_inh,       spr_inh,        #  0,  1
        exp_exh,       spr_exh,        #  2,  3
        exp_i2e,       spr_i2e,        #  4,  5
        exp_e2i,       spr_e2i,        #  6,  7
        exp_slope_inh, spr_slope_inh,  #  8,  9
        exp_slope_exh, spr_slope_exh,  # 10, 11
        exp_prom_inh,  spr_prom_inh,   # 12, 13
        exp_prom_exh,  spr_prom_exh,   # 14, 15
        last_inh,                      # 16
        last_exh,                      # 17
    )



def initial_calibration(raw_stream):
    """Full calibration including seed events. Returns 18-element tuple or None."""
    return _compute_calibration(raw_stream)


def rolling_calibration(raw_stream):
    """
    Rolling recalibration — same stats as initial but drops seed events
    so the caller keeps its existing inh_onset / exh_onset lists.
    Returns a 16-element tuple (indices 0–15) or None.
    """
    calib = _compute_calibration(raw_stream)
    if calib is None:
        return None
    return calib[:16]   # drop last_inh (16) and last_exh (17)