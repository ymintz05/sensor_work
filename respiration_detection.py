#!/usr/bin/env python3
"""
Minimal Vernier Respiration (USB) -> LSL + live print + save NPZ
"""

import time
import threading
import numpy as np
from datetime import datetime
import math
import tkinter as tk
from dataclasses import dataclass

from godirect import GoDirect
from pylsl import StreamInfo, StreamOutlet

#============# Changeable Settings #============#

@dataclass
class Config:

    RR_refrac_sensitivity: float = 0.1 #in sec
    deriv_tightness: int = 3 #MIN 3
    calib_window: float = 10   #in sec
    srate_hz: float = 10
    process: bool = True

    # Preprocessing #
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0



#===============================================#
 

#========== Data Preprocessing ==========#
    
def smoothing_factor(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
        return a * x + (1 - a) * x_prev

class OneEuroFilter:
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

#========== Respiration Recorder ==========#
    
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

        # Processed Data
        self.samples_processed = []
        

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

                        # Initialize filter class #
                        if self.filter is None:
                            self.filter = OneEuroFilter(
                                ts,
                                val,
                                min_cutoff=self.cfg.min_cutoff,
                                beta=self.cfg.beta,
                                d_cutoff=self.cfg.d_cutoff
                            )
                    
                        # Record 
                        self.samples.append(val)
                        self.timestamps.append(ts)

                        # Preprocess 
                        if (len(self.timestamps)) >= 2:
                            self.samples_processed.append(self.filter(ts,val))
                        elif (len(self.timestamps) < 2):
                            self.samples_processed.append(val)
                        

                        #Analysis
                        self.analysis()

                        # Push to LSL
                        if self.outlet is not None:
                            self.outlet.push_sample([val])

                        count += 1

                # Print once per second (lightweight)
                now = time.time()
                if now - last_print >= 1.0 and self.samples:
                    # show last value + rough rate
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

        samples = self.samples

        if (len(self.timestamps) >= self.cfg.deriv_tightness):
            
            mp = math.floor(self.cfg.deriv_tightness/2)

            last_values = samples[-self.cfg.deriv_tightness:]
            last_timestamps = self.timestamps[-self.cfg.deriv_tightness:]
            initial_diff = last_values[mp] - last_values[0]
            final_diff = last_values[self.cfg.deriv_tightness - 1] - last_values[mp]

            derivative_approx_init =  initial_diff/(self.dt * mp)
            derivative_approx_final =  final_diff/(self.dt * mp)

            return (derivative_approx_init, derivative_approx_final)
        else:
            return

        
    #def calibration(self):
        

    def analysis(self):
        
        is_exh_candidate = is_inh_candidate = is_first_inst = is_refrac_period = False
        
        # if first instance
        if len(self.total_peaktrough)==0: 
            is_first_inst = True

        samples = self.samples
        
        # validate sample n to derive vals
        if (len(self.timestamps) >= self.cfg.deriv_tightness):
            
            mp = math.floor(self.cfg.deriv_tightness/2)
            last_values = samples[-self.cfg.deriv_tightness:]
            last_timestamps = self.timestamps[-self.cfg.deriv_tightness:]

            derivative_approx_init, derivative_approx_final = self.derivative()

            
            # prevents counting during refraction period
            if (
                (len(self.total_peaktrough) > 0) and 
                (last_timestamps[mp] - self.total_peaktrough[-1][0] < self.cfg.RR_refrac_sensitivity)
                ): 
                is_refrac_period = True  
            
            # inhalation, exhalation criteria
            if derivative_approx_init > 0 and derivative_approx_final < 0:
                is_exh_candidate = True
            if derivative_approx_init < 0 and derivative_approx_final > 0:
                is_inh_candidate = True
    

            # breath capture logic
            if (
                is_first_inst or 
                not is_refrac_period
                ):

                if (is_exh_candidate):
                    self.total_peaktrough.append((last_timestamps[mp], last_values[mp], "exh start"))

                if (is_inh_candidate):
                    self.total_peaktrough.append((last_timestamps[mp], last_values[mp], "inh start")) 

                    # wavelength capture
                    if (
                        ((len(self.total_peaktrough) >= 3)) and  (self.total_peaktrough[-3][2] == "inh start") and 
                        (self.total_peaktrough[-2][2] == "exh start")
                        ):

                        self.wavelength.append(((last_timestamps[mp], last_values[mp]),(self.total_peaktrough[-3][0], self.total_peaktrough[-3][1])))

        # breath status stream
        if(len(self.total_peaktrough) > 0):
            if (self.total_peaktrough[-1][2] == "exh start"):
                self.breath_status.append(((self.timestamps[-1]), "exh")) 

            if (self.total_peaktrough[-1][2] == "inh start"):
                self.breath_status.append(((self.timestamps[-1]), "inh"))
        else:
            self.breath_status.append(((self.timestamps[-1]), "N/A"))


    # def derived_values(self):
    #     # RR
    #     if len(self.total_peaktrough) < 2:
    #         return
    #     if len(self.timestamps) < 1:
    #         return

    #     duration = self.timestamps[-1] - self.timestamps[0]
    #     RR = len(self.wavelength) / duration
    #     self.RR.append((self.timestamps[-1], RR))

    #     # Breath Depth 
    #     if (self.total_peaktrough[-1][2] == "exh start"):
    #         self.breath_depth.append((self.total_peaktrough[-1][0], abs(self.total_peaktrough[-1][1] - self.total_peaktrough[-2][1])))


    # placeholder, to be replaced with VNS setup
    def start_phase_window(self, poll_ms=100):

        self.root = tk.Tk()
        self.root.title("Respiration Phase")
        self.root.geometry("300x200")

        self.label = tk.Label(self.root, text="WAITING", font=("Helvetica", 24))
        self.label.pack(expand=True, fill="both")

        def tick():
            
            phase = "N/A"

            if self.breath_status:
                phase = self.breath_status[-1][1]

            if phase == "inh":
                self.root.configure(bg="dodgerblue")
                self.label.configure(text="INHALE", bg="dodgerblue")
            elif phase == "exh":
                self.root.configure(bg="tomato")
                self.label.configure(text="EXHALE", bg="tomato")
            else:
                self.root.configure(bg="gray")
                self.label.configure(text="WAITING", bg="gray")

            self.root.after(poll_ms, tick)

        tick()
        self.root.mainloop()

    def stop_and_save(self, prefix="vernier_resp"):
        print("Stopping...")
        self.streaming = False
        if self.thread:
            self.thread.join(timeout=2)

        # Clean up device
        try:
            if self.device:
                self.device.stop()
                self.device.close()
        except Exception:
            pass

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.npz"

        samples = np.array(self.samples, dtype=np.float32)
        samples_processed = np.array(self.samples_processed, dtype=np.float32)
        time = np.array(self.timestamps, dtype=np.float64)
        peaktrough = np.array(self.total_peaktrough, dtype=object)
        breath_status = np.array(self.breath_status, dtype=object)

        np.savez(
            filename, 
            data=samples, 
            data2=samples_processed,
            timestamps=time, 
            srate_hz=self.cfg.srate_hz, 
            peaktrough=peaktrough, 
            breath_status=breath_status,
            )
        print(f" Saved: {filename}")
        print(f"Total samples: {len(samples)}")
        if len(self.timestamps) > 1:
            dur = self.timestamps[-1] - self.timestamps[0]
            print(f"Duration: {dur:.1f} s | Avg rate: {len(samples)/dur:.2f} Hz")
        return filename


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", "-d", type=float, default=60.0, help="recording duration in seconds")
    p.add_argument("--output", "-o", type=str, default="vernier_resp", help="output file prefix")
    args = p.parse_args()

    rec = VernierRespRecorder()
    rec.start()

    rec.root = tk.Tk()
    rec.root.title("Respiration Phase")
    rec.root.geometry("300x200")

    rec.label = tk.Label(rec.root, text="WAITING", font=("Helvetica", 24))
    rec.label.pack(expand=True, fill="both")

    def tick():
        phase = rec.breath_status[-1][1] if rec.breath_status else "N/A"

        if phase == "inh":
            rec.root.configure(bg="dodgerblue")
            rec.label.configure(text="INHALE", bg="dodgerblue")
        elif phase == "exh":
            rec.root.configure(bg="tomato")
            rec.label.configure(text="EXHALE", bg="tomato")
        else:
            rec.root.configure(bg="gray")
            rec.label.configure(text="WAITING", bg="gray")

        rec.root.after(100, tick)

    def stop_everything():
        rec.stop_and_save(prefix=args.output)
        rec.root.destroy()

    tick()
    rec.root.after(int(args.duration * 1000), stop_everything)

    try:
        rec.root.mainloop()
    except KeyboardInterrupt:
        stop_everything()


if __name__ == "__main__":
    main()