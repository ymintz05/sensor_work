import math
import time
import numpy as np
import signal_processing
import calibration
from config import Config
from godirect import GoDirect
from pylsl import StreamInfo, StreamOutlet
import threading

# ═══════════════════════════════════════════════════════════════════════════
# Adaptive weighting
# ═══════════════════════════════════════════════════════════════════════════  

def adapt_weights_from_initial_calibration(
    cfg,
    calibration_vals,
    base_weights=None,
    blend=0.35,
):
    """
    Gently readjust hardcoded baseline weights after initial calibration.

    Uses feature reliability:
      - slope: log-space spread only
      - timing/prominence: relative spread = spread / expected

    Requires calibration_vals to be the 16-value tuple:
      expected/spread for inh, exh, i2e, e2i, slope inh/exh, prom inh/exh
    """

    (
        expected_inh,        spread_inh,
        expected_exh,        spread_exh,
        expected_inh_to_exh, spread_inh_to_exh,
        expected_exh_to_inh, spread_exh_to_inh,
        expected_slope_inh,  spread_slope_inh,
        expected_slope_exh,  spread_slope_exh,
        expected_prom_inh,   spread_prom_inh,
        expected_prom_exh,   spread_prom_exh,
    ) = calibration_vals

    def base(name):
        if base_weights is not None:
            return float(base_weights[name])
        return float(getattr(cfg, name))

    def relative_reliability(expected, spread, min_mult=0.6, max_mult=1.4):
        eps = 1e-6
        relative_spread = abs(spread) / (abs(expected) + eps)

        # relative_spread ~= 0.25 gives multiplier ~= 1
        mult = 0.25 / (relative_spread + eps)

        return float(np.clip(mult, min_mult, max_mult))

    def log_reliability(spread, min_mult=0.6, max_mult=1.4):
        eps = 1e-6

        # In log-space, smaller spread means more reliable.
        # spread ~= 0.25 gives multiplier ~= 1
        mult = 0.25 / (abs(spread) + eps)

        return float(np.clip(mult, min_mult, max_mult))

    def blend_weight(base_weight, multiplier):
        return float(base_weight * ((1.0 - blend) + blend * multiplier))

    # Slope calibration is log-space, so use log-space spread directly.
    slope_mult_inh = log_reliability(spread_slope_inh)
    slope_mult_exh = log_reliability(spread_slope_exh)

    # Timing is raw seconds, so relative spread is appropriate.
    timing_mult_inh = 0.5 * relative_reliability(
        expected_inh,
        spread_inh,
    ) + 0.5 * relative_reliability(
        expected_exh_to_inh,
        spread_exh_to_inh,
    )

    timing_mult_exh = 0.5 * relative_reliability(
        expected_exh,
        spread_exh,
    ) + 0.5 * relative_reliability(
        expected_inh_to_exh,
        spread_inh_to_exh,
    )

    # Prominence is raw amplitude, so relative spread is appropriate.
    prom_mult_inh = relative_reliability(expected_prom_inh, spread_prom_inh)
    prom_mult_exh = relative_reliability(expected_prom_exh, spread_prom_exh)

    # Apply gentle readjustment from hardcoded baseline weights.
    cfg.w1i = blend_weight(base("w1i"), slope_mult_inh)
    cfg.w2i = blend_weight(base("w2i"), timing_mult_inh)
    cfg.w3i = blend_weight(base("w3i"), prom_mult_inh)

    cfg.w1e = blend_weight(base("w1e"), slope_mult_exh)
    cfg.w2e = blend_weight(base("w2e"), timing_mult_exh)
    cfg.w3e = blend_weight(base("w3e"), prom_mult_exh)

    print("Initial calibration readjusted weights:")
    print(
        f"  INH: slope={cfg.w1i:.3f}, "
        f"timing={cfg.w2i:.3f}, "
        f"prominence={cfg.w3i:.3f}"
    )
    print(
        f"  EXH: slope={cfg.w1e:.3f}, "
        f"timing={cfg.w2e:.3f}, "
        f"prominence={cfg.w3e:.3f}"
    )

    print("Reliability multipliers:")
    print(
        f"  INH: slope={slope_mult_inh:.3f}, "
        f"timing={timing_mult_inh:.3f}, "
        f"prominence={prom_mult_inh:.3f}"
    )
    print(
        f"  EXH: slope={slope_mult_exh:.3f}, "
        f"timing={timing_mult_exh:.3f}, "
        f"prominence={prom_mult_exh:.3f}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# Recording Class
# ═══════════════════════════════════════════════════════════════════════════  

class VernierRespRecorder:

    def __init__(self, cfg: Config = Config()):

        self.base_weights = {
        "w1i": cfg.w1i,
        "w2i": cfg.w2i,
        "w3i": cfg.w3i,
        "w1e": cfg.w1e,
        "w2e": cfg.w2e,
        "w3e": cfg.w3e,
        }

        self.cfg    = cfg
        self.filter = None
        self.dt     = 1.0 / cfg.srate_hz

        self.device    = None
        self.streaming = False
        self.thread    = None
        self.outlet    = None

        # ── Raw data ──────────────────────────────────────────────────────────
        self.samples    = []
        self.timestamps = []
        self.t0         = None

        # ── Processed signal ──────────────────────────────────────────────────
        self.samples_processed = []

        # ── Detection outputs ─────────────────────────────────────────────────
        self.inh_onset        = []   # [(ts, val), ...]
        self.exh_onset        = []   # [(ts, val), ...]
        self.breath_status    = []   # [(ts, "inh"|"exh"), ...]
        self.total_peaktrough = []   # [(ts, val, "inh start"|"exh start"), ...]
        self.last_phase       = None

        # ── Derived metrics ───────────────────────────────────────────────────
        self.wavelength   = []
        self.RR           = []
        self.breath_depth = []

        # ── Calibration ───────────────────────────────────────────────────────
        self.calibration_vals         = None
        self.initial_calibration_done = False
        self.last_recalibration_time  = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Start
    # ═══════════════════════════════════════════════════════════════════════════

    def start(self):
        print("Connecting to Vernier USB device...")
        gd      = GoDirect(use_ble=False, use_usb=True)
        devices = gd.list_devices()
        if not devices:
            raise RuntimeError("No Vernier USB devices found")

        self.device = devices[0]
        if not self.device.open():
            raise RuntimeError("Failed to open Vernier device")

        print(f"Connected to: {self.device.name}")
        self.device.enable_sensors([1])

        info = StreamInfo(
            "VernieRespiration", "Respiration", 1,
            self.cfg.srate_hz, "float32", "vernier_resp"
        )
        info.desc().append_child_value("manufacturer", "Vernier")
        info.desc().append_child_value("model", "Go Direct Respiration Belt")
        info.desc().append_child_value("units", "Newtons")
        self.outlet = StreamOutlet(info)

        self.device.start()
        self.streaming = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"Streaming + recording started ({self.cfg.srate_hz:.0f} Hz)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Worker
    # ═══════════════════════════════════════════════════════════════════════════

    def _worker(self):
        last_print = time.time()

        while self.streaming:
            try:
                if self.device.read():
                    sensors = self.device.get_enabled_sensors()
                    for s in sensors:
                        if getattr(s, "value", None) is None:
                            continue

                        val = float(s.value)
                        ts_abs = time.time()

                        if self.t0 is None:
                            self.t0 = ts_abs

                        ts = ts_abs - self.t0

                        # ── Filter init ───────────────────────────────────────
                        if self.filter is None:
                            if self.cfg.filter_type == "l":
                                self.filter = signal_processing.lowpass(
                                    ts, val,
                                    min_cutoff=self.cfg.min_cutoff,
                                    beta=self.cfg.beta,
                                    d_cutoff=self.cfg.d_cutoff,
                                )
                            elif self.cfg.filter_type == "b":
                                self.filter = signal_processing.bandpass(
                                    fs=self.cfg.srate_hz,
                                    low=self.cfg.low,
                                    high=self.cfg.high,
                                    order=self.cfg.order,
                                )

                        # ── Filter ────────────────────────────────────────────
                        if self.cfg.filter_type == "l":
                            processed_val = (self.filter(ts, val)
                                             if len(self.timestamps) >= 2 else val)
                        elif self.cfg.filter_type == "b":
                            processed_val = self.filter(val)
                        else:
                            processed_val = val

                        self.samples.append(val)
                        self.timestamps.append(ts)
                        self.samples_processed.append(processed_val)

                        t_rel = ts

                        # ── Analysis ──────────────────────────────────────────
                        if self.initial_calibration_done:
                            self.analysis()

                        # ── Initial calibration ───────────────────────────────
                        if (not self.initial_calibration_done
                                and t_rel >= self.cfg.initial_calibration_delay_s):
                            calib = calibration.initial_calibration(self.samples_processed)
                            if calib is not None:
                                *calibration_vals, last_inh, last_exh = calib
                                self.calibration_vals = tuple(calibration_vals)

                                adapt_weights_from_initial_calibration(
                                    self.cfg,
                                    self.calibration_vals,
                                    base_weights=self.base_weights,
                                    blend=0.35,
                                )
                                self.inh_onset = [last_inh]
                                self.exh_onset = [last_exh]
                                self.last_phase = "inh" if last_inh[0] > last_exh[0] else "exh"
                                self.initial_calibration_done = True
                                self.last_recalibration_time = ts
                                print("Initial calibration complete")

                        # ── Rolling recalibration ─────────────────────────────
                        elif (self.initial_calibration_done
                              and ts - self.last_recalibration_time
                              >= self.cfg.recalibration_interval_s):
                            # look back 3x the recalib interval for stable stats
                            lookback_n = int(
                                3 * self.cfg.recalibration_interval_s * self.cfg.srate_hz
                            )
                            recent = self.samples_processed[-lookback_n:]
                            if recent:
                                calib = calibration.rolling_calibration(recent)
                                if calib is not None:
                                    self.calibration_vals        = calib
                                    self.last_recalibration_time = ts
                                    print(f"Recalibration complete  t={t_rel:.1f}s")

                        # ── LSL ───────────────────────────────────────────────
                        if self.outlet is not None:
                            self.outlet.push_sample([val])

                # Status print once per second
                now = time.time()
                if now - last_print >= 1.0 and self.samples:
                    dur   = (self.timestamps[-1] - self.timestamps[0]
                             if len(self.timestamps) > 1 else 0.0)
                    rate  = len(self.samples) / dur if dur > 0 else 0.0
                    phase = self.breath_status[-1][1] if self.breath_status else "N/A"
                    print(f"Samples: {len(self.samples)} | "
                          f"Last: {self.samples[-1]:.4f} N | "
                          f"~{rate:.1f} Hz | {phase}")
                    last_print = now

                time.sleep(self.dt)

            except Exception as e:
                print(f"Vernier worker error: {e}")
                break

    # ═══════════════════════════════════════════════════════════════════════════
    # Derivative
    # ═══════════════════════════════════════════════════════════════════════════

    def derivative(self):
        n = self.cfg.deriv_tightness
        if len(self.timestamps) < n:
            return None
        mp   = math.floor(n / 2)
        last = self.samples_processed[-n:]
        d_i  = (last[mp]  - last[0])  / (self.dt * mp)
        d_f  = (last[-1]  - last[mp]) / (self.dt * mp)
        return d_i, d_f

    # ═══════════════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════════════

    def analysis(self):

        # ── Early returns ─────────────────────────────────────────────────────
        if len(self.timestamps) == 0:
            return None
        if len(self.samples_processed) < self.cfg.deriv_tightness:
            return None
        if self.calibration_vals is None:
            return None
        if not self.inh_onset:
            return None
        if not self.exh_onset:
            return None
        deriv = self.derivative()
        if deriv is None:
            return None

        # ── Unpack calibration ────────────────────────────────────────────────
        (
            expected_inh,        spread_inh,
            expected_exh,        spread_exh,
            expected_inh_to_exh, spread_inh_to_exh,
            expected_exh_to_inh, spread_exh_to_inh,
            expected_slope_inh,  spread_slope_inh,
            expected_slope_exh,  spread_slope_exh,
            expected_prom_inh,   spread_prom_inh,
            expected_prom_exh,   spread_prom_exh,
        ) = self.calibration_vals

        # ── Window ────────────────────────────────────────────────────────────
        n               = self.cfg.deriv_tightness
        last_values     = self.samples_processed[-n:]
        last_timestamps = self.timestamps[-n:]

        inh_idx      = int(np.argmin(last_values))
        exh_idx      = int(np.argmax(last_values))
        inh_cand_val = last_values[inh_idx]
        inh_cand_ts  = last_timestamps[inh_idx]
        exh_cand_val = last_values[exh_idx]
        exh_cand_ts  = last_timestamps[exh_idx]

        last_inh_ts, last_inh_val = self.inh_onset[-1]
        last_exh_ts, last_exh_val = self.exh_onset[-1]

        # current_ts is the most recent sample — use for interval timing
        # so elapsed time isn't underestimated by the candidate's lag in the window
        current_ts       = last_timestamps[-1]
        since_last_inh_s = current_ts - last_inh_ts
        since_last_exh_s = current_ts - last_exh_ts

        ddx_approx_i, ddx_approx_f = deriv

        # ── Inflection gate ───────────────────────────────────────────────────
        is_inh_candidate = ddx_approx_i < 0 and ddx_approx_f > 0
        is_exh_candidate = ddx_approx_i > 0 and ddx_approx_f < 0

        can_detect_inh = self.last_phase in [None, "exh"]
        can_detect_exh = self.last_phase in [None, "inh"]

        # ── Departure slopes ──────────────────────────────────────────────────
        inh_dt    = last_timestamps[-1] - inh_cand_ts + 1e-6
        inh_slope = (last_values[-1] - inh_cand_val) / inh_dt

        exh_dt    = last_timestamps[-1] - exh_cand_ts + 1e-6
        exh_slope = (exh_cand_val - last_values[-1]) / exh_dt

        # ── Inhalation ────────────────────────────────────────────────────────
        if is_inh_candidate and can_detect_inh:

            # log-normal slope: slopes are positive and right-skewed
            # calibration stores log-space median and spread
            if inh_slope > 0:
                z = (np.log(inh_slope) - expected_slope_inh) / (spread_slope_inh + 1e-6)
                z_score_slope_inh = np.exp(-0.5 * z**2)
            else:
                z_score_slope_inh = 0.0

            # one-sided interval: being early is implausible, being late is normal
            _z_inh = (since_last_inh_s - expected_inh) / (spread_inh + 1e-6)
            z_score_inh = np.exp(-0.5 * (_z_inh / (1.0 if since_last_inh_s < expected_inh else 2.5))**2)

            # one-sided cross-interval: same asymmetric logic
            _gap_e2i = inh_cand_ts - last_exh_ts
            _z_e2i   = (_gap_e2i - expected_exh_to_inh) / (spread_exh_to_inh + 1e-6)
            z_score_inh_cross = np.exp(-0.5 * (_z_e2i / (1.0 if _gap_e2i < expected_exh_to_inh else 3.0))**2)

            # prominence: depth below last exh peak — clamped to zero to avoid
            # belt drift creating negative prominences that score symmetrically
            prominence_inh = max(0.0, last_exh_val - inh_cand_val)
            z = abs(prominence_inh - expected_prom_inh) / (spread_prom_inh + 1e-6)
            z_score_prom_inh = np.exp(-0.5 * z**2)

            timing_inh = (z_score_inh * z_score_inh_cross) ** 0.5

            if z_score_slope_inh >= self.cfg.floor_slope_inh:
                inh_candidacy_score = (
                    self.cfg.w1i * z_score_slope_inh +
                    self.cfg.w2i * timing_inh +
                    self.cfg.w3i * z_score_prom_inh
                )
                if inh_candidacy_score > self.cfg.cutoff_inh:
                    self.inh_onset.append((inh_cand_ts, inh_cand_val))
                    self.breath_status.append((inh_cand_ts, "inh"))
                    self.total_peaktrough.append((inh_cand_ts, inh_cand_val, "inh start"))
                    self.last_phase = "inh"

        # ── Exhalation ────────────────────────────────────────────────────────
        if is_exh_candidate and can_detect_exh:

            # log-normal slope
            if exh_slope > 0:
                z = (np.log(exh_slope) - expected_slope_exh) / (spread_slope_exh + 1e-6)
                z_score_slope_exh = np.exp(-0.5 * z**2)
            else:
                z_score_slope_exh = 0.0

            # one-sided interval
            _z_exh = (since_last_exh_s - expected_exh) / (spread_exh + 1e-6)
            z_score_exh = np.exp(-0.5 * (_z_exh / (1.0 if since_last_exh_s < expected_exh else 2.5))**2)

            # one-sided cross-interval
            _gap_i2e = exh_cand_ts - last_inh_ts
            _z_i2e   = (_gap_i2e - expected_inh_to_exh) / (spread_inh_to_exh + 1e-6)
            z_score_exh_cross = np.exp(-0.5 * (_z_i2e / (1.0 if _gap_i2e < expected_inh_to_exh else 3.0))**2)

            # prominence: height above last inh trough — clamped
            prominence_exh = max(0.0, exh_cand_val - last_inh_val)
            z = abs(prominence_exh - expected_prom_exh) / (spread_prom_exh + 1e-6)
            z_score_prom_exh = np.exp(-0.5 * z**2)

            timing_exh = (z_score_exh * z_score_exh_cross) ** 0.5

            if z_score_slope_exh >= self.cfg.floor_slope_exh:
                exh_candidacy_score = (
                    self.cfg.w1e * z_score_slope_exh +
                    self.cfg.w2e * timing_exh +
                    self.cfg.w3e * z_score_prom_exh
                )
                if exh_candidacy_score > self.cfg.cutoff_exh:
                    self.exh_onset.append((exh_cand_ts, exh_cand_val))
                    self.breath_status.append((exh_cand_ts, "exh"))
                    self.total_peaktrough.append((exh_cand_ts, exh_cand_val, "exh start"))
                    self.last_phase = "exh"