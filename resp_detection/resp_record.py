import math
import time
import numpy as np
import signal_processing, calibration
from config import Config
from godirect import GoDirect
from pylsl import StreamInfo, StreamOutlet
import threading

class VernierRespRecorder:
    def __init__(self, cfg: Config = Config()):
        
        self.cfg = cfg
        self.filter = None
        self.dt = 1.0 / self.cfg.srate_hz
    
        self.device = None
        self.streaming = False
        self.thread = None

        self.outlet = None  # LSL outlet (optional but enabled)

        # Recorded data
        self.samples = []
        self.timestamps = []  # local time.time() seconds
        # Derived
        self.breath_status = []
        self.total_peaktrough = []
        self.wavelength = []
        self.RR = []
        self.breath_depth = []
        self.inh_onset = []
        self.exh_onset = []
        self.last_phase = None

        # Processed Data
        self.samples_processed = []

        # Calibration Vals
        self.calibration_vals = None
        self.initial_calibration_done = False
        self.initial_calibration_delay_s = 30.0   # or whatever you want
        self.recalibration_interval_s = 10.0
        self.last_recalibration_time = None
        

    def start(self):
        print("Connecting to Vernier USB device...")
        gd = GoDirect(use_ble=False, use_usb=True)
        devices = gd.list_devices()
        if not devices:
            raise RuntimeError("No Vernier USB devices found")

        self.device = devices[0]
        if not self.device.open():
            raise RuntimeError("Failed to open Vernier device")

        print(f"Connected to: {self.device.name}")

        # Enable first sensor (your belt)
        self.device.enable_sensors([1])

        # Create LSL stream (1 channel @ 10 Hz)
        info = StreamInfo("VernieRespiration", "Respiration", 1, self.cfg.srate_hz, "float32", "vernier_resp")
        info.desc().append_child_value("manufacturer", "Vernier")
        info.desc().append_child_value("model", "Go Direct Respiration Belt")
        info.desc().append_child_value("units", "Newtons")
        self.outlet = StreamOutlet(info)

        self.device.start()
        self.streaming = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        print(f"Streaming + recording started ({self.cfg.srate_hz:.0f} Hz)")


    def _worker(self):
        count = 0
        last_print = time.time()

        while self.streaming:
            try:
                if self.device.read():
                    sensors = self.device.get_enabled_sensors()
                    for s in sensors:
                        if getattr(s, "value", None) is None:
                            continue

                        val = float(s.value)
                        ts = time.time()

                        # Initialize filter
                        if self.filter is None:
                            if self.cfg.filter_type == "l":
                                self.filter = signal_processing.lowpass(
                                    ts,
                                    val,
                                    min_cutoff=self.cfg.min_cutoff,
                                    beta=self.cfg.beta,
                                    d_cutoff=self.cfg.d_cutoff
                                )
                            elif self.cfg.filter_type == "b":
                                self.filter = signal_processing.bandpass(
                                    fs=self.cfg.srate_hz,
                                    low=self.cfg.low,
                                    high=self.cfg.high,
                                    order=self.cfg.order
                                )

                        # Record raw sample
                        self.samples.append(val)
                        self.timestamps.append(ts)

                        # Process sample
                        if self.cfg.filter_type == "l":
                            if len(self.timestamps) >= 2:
                                processed_val = self.filter(ts, val)
                            else:
                                processed_val = val
                        elif self.cfg.filter_type == "b":
                            processed_val = self.filter(val)
                        else:
                            processed_val = val

                        self.samples_processed.append(processed_val)

                        # Relative time from first timestamp
                        t_rel = ts - self.timestamps[0]

                        # Run analysis once enough data exists
                        if self.initial_calibration_done:
                            self.analysis()

                        # ---------------------------
                        # Initial calibration
                        # ---------------------------
                        if (
                            not self.initial_calibration_done
                            and t_rel >= self.initial_calibration_delay_s
                        ):
                            # use ALL processed data collected so far
                            calib = calibration.initial_calibration(self.samples_processed)

                            if calib is not None:
                                *calibration_vals, last_inh, last_exh = calib

                                self.calibration_vals = tuple(calibration_vals)
                                self.inh_onset = [last_inh]
                                self.exh_onset = [last_exh]
                                self.last_phase = "inh" if last_inh[0] > last_exh[0] else "exh"

                                self.initial_calibration_done = True
                                self.last_recalibration_time = ts
                                print("Initial calibration complete")

                        # ---------------------------
                        # Recalibration every 10 sec
                        # ---------------------------
                        elif self.initial_calibration_done:
                            if ts - self.last_recalibration_time >= self.recalibration_interval_s:
                                window_n = int(self.recalibration_interval_s * self.cfg.srate_hz)
                                recent_processed = self.samples_processed[-window_n:]

                                if len(recent_processed) > 0:
                                    calib = calibration.rolling_calibration(recent_processed)

                                    if calib is not None:
                                        self.calibration_vals = calib
                                        self.last_recalibration_time = ts
                                        print("Recalibration complete")

                        # Push to LSL
                        if self.outlet is not None:
                            self.outlet.push_sample([val])

                        count += 1

                now = time.time()
                if now - last_print >= 1.0 and self.samples:
                    dur = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0
                    rate = len(self.samples) / dur if dur > 0 else 0.0
                    phase = self.breath_status[-1][1] if self.breath_status else "N/A"
                    print(f"Samples: {len(self.samples)} | Last: {self.samples[-1]:.4f} N | ~{rate:.1f} Hz | {phase}")
                    last_print = now

                time.sleep(self.dt)

            except Exception as e:
                print(f"Vernier worker error: {e}")
                break


    def derivative(self):

        samples = self.samples_processed

        if (len(self.timestamps) >= self.cfg.deriv_tightness):
            
            mp = math.floor(self.cfg.deriv_tightness/2)

            last_values = samples[-self.cfg.deriv_tightness:]
            initial_diff = last_values[mp] - last_values[0]
            final_diff = last_values[self.cfg.deriv_tightness - 1] - last_values[mp]

            derivative_approx_init =  initial_diff/(self.dt * mp)
            derivative_approx_final =  final_diff/(self.dt * mp)

            return (derivative_approx_init, derivative_approx_final)
        else:
            return


    def analysis(self):

        # -------------------------
        # Early Returns
        # -------------------------

        if len(self.timestamps) == 0:
            return None

        if len(self.samples_processed) < self.cfg.deriv_tightness:
            return None

        if self.calibration_vals is None:
            return None

        if len(self.inh_onset) == 0:
            return None

        if len(self.exh_onset) == 0:
            return None

        deriv = self.derivative()
        if deriv is None:
            return None

        # -------------------------
        # Derive Vals 
        # -------------------------

        samples = self.samples_processed

        (
            expected_inh,
            spread_inh,
            expected_exh,
            spread_exh,
            expected_inh_to_exh,
            spread_inh_to_exh,
            expected_exh_to_inh,
            spread_exh_to_inh,
            expected_trough_val,
            spread_trough_val
        ) = self.calibration_vals

        midpt = math.floor(self.cfg.deriv_tightness / 2)
        last_values = samples[-self.cfg.deriv_tightness:]
        last_timestamps = self.timestamps[-self.cfg.deriv_tightness:]

        candidate_val = last_values[midpt]
        candidate_ts = last_timestamps[midpt]

        last_inh_ts, last_inh_val = self.inh_onset[-1]
        since_last_inh_s = candidate_ts - last_inh_ts

        last_exh_ts, last_exh_val = self.exh_onset[-1]
        since_last_exh_s = candidate_ts - last_exh_ts

        ddx_approx_i, ddx_approx_f = deriv

        # -------------------------
        # Gating 
        # -------------------------

        is_inh_candidate = ddx_approx_i < 0 and ddx_approx_f > 0
        is_exh_candidate = ddx_approx_i > 0 and ddx_approx_f < 0

        can_detect_inh = self.last_phase in [None, "exh"]
        can_detect_exh = self.last_phase in [None, "inh"]


        # -------------------------
        # Scoring 
        # -------------------------

        inh_shape_score = ddx_approx_f - ddx_approx_i
        exh_shape_score = ddx_approx_i - ddx_approx_f

        if is_inh_candidate and can_detect_inh:

            # z-score last inh-inh
            z = abs(since_last_inh_s - expected_inh) / (spread_inh + 1e-6)
            z_score_inh = np.exp(-0.5 * z**2)

            # z-score exh-inh cross for inh
            z = abs(since_last_exh_s - expected_exh_to_inh) / (spread_exh_to_inh + 1e-6)
            z_score_inh_cross = np.exp(-0.5 * z**2)

            # z-score trough depth 
            z = abs(candidate_val - expected_trough_val) / (spread_trough_val + 1e-6)
            z_score_trough_val = np.exp(-0.5 * z**2)

            # summative scoring
            inh_candidacy_score = (
                w1i * inh_shape_score +
                w2i * z_score_inh +
                w3i * z_score_inh_cross +
                w4i * z_score_trough_val 
            )

            if inh_candidacy_score > cutoff_inh:
                self.inh_onset.append((candidate_ts, candidate_val))
                self.last_phase = "inh"

        if is_exh_candidate and can_detect_exh:

            # z-score last exh-exh 
            z = abs(since_last_exh_s - expected_exh) / (spread_exh + 1e-6)
            z_score_exh = np.exp(-0.5 * z**2)

            # z-score inh-exh cross for exh 
            z = abs(since_last_inh_s - expected_inh_to_exh) / (spread_inh_to_exh + 1e-6)
            z_score_exh_cross = np.exp(-0.5 * z**2)

            # summative scoring 
            exh_candidacy_score = (
                w1e * exh_shape_score +
                w2e * z_score_exh +
                w3e * z_score_exh_cross
            )

            if exh_candidacy_score > cutoff_exh:
                self.exh_onset.append((candidate_ts, candidate_val))
                self.last_phase = "exh"

