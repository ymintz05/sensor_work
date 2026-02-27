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

from godirect import GoDirect
from pylsl import StreamInfo, StreamOutlet


class VernierRespRecorder:
    def __init__(self, srate_hz=10):
        self.srate_hz = float(srate_hz)
        self.dt = 1.0 / self.srate_hz

        self.device = None
        self.streaming = False
        self.thread = None

        self.outlet = None  # LSL outlet (optional but enabled)

        # Recorded data
        self.samples = []
        self.timestamps = []  # local time.time() seconds

        #peak-trough analysis
        
        self.RR_refrac_sensitivity = 0.1 #in sec
        self.breath_status = []
        self.deriv_tightness = 3
        self.total_peaktrough = []
    
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
        info = StreamInfo("VernieRespiration", "Respiration", 1, self.srate_hz, "float32", "vernier_resp")
        info.desc().append_child_value("manufacturer", "Vernier")
        info.desc().append_child_value("model", "Go Direct Respiration Belt")
        info.desc().append_child_value("units", "Newtons")
        self.outlet = StreamOutlet(info)

        self.device.start()
        self.streaming = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        print(f"Streaming + recording started ({self.srate_hz:.0f} Hz)")

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

                        # Record
                        self.samples.append(val)
                        self.timestamps.append(ts)

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

    def analysis(self):

        if (len(self.timestamps) >= self.deriv_tightness):

            mp = math.floor(self.deriv_tightness/2)

            last_values = self.samples[-self.deriv_tightness:]
            last_timestamps = self.timestamps[-self.deriv_tightness:]
            initial_diff = last_values[mp] - last_values[0]
            final_diff = last_values[self.deriv_tightness - 1] - last_values[mp]

            derivative_approx_init =  initial_diff/self.dt 
            derivative_approx_final =  final_diff/self.dt
                
            if (len(self.total_peaktrough)==0) or (last_timestamps[mp] - self.total_peaktrough[-1][0] > self.RR_refrac_sensitivity):
                if (derivative_approx_init > 0 and derivative_approx_final < 0):
                    self.total_peaktrough.append((last_timestamps[mp], last_values[mp], "exh start"))
                if (derivative_approx_init < 0 and derivative_approx_final > 0):
                    self.total_peaktrough.append((last_timestamps[mp], last_values[mp], "inh start")) 

        if(len(self.total_peaktrough)>0):
            if (self.total_peaktrough[-1][2] == "exh start"):
                self.breath_status.append(((self.timestamps[-1]), "exh"))

            if (self.total_peaktrough[-1][2] == "inh start"):
                self.breath_status.append(((self.timestamps[-1]), "inh"))
        else:
            self.breath_status.append(((self.timestamps[-1]), "N/A"))

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

        data = np.array(self.samples, dtype=np.float32)
        t = np.array(self.timestamps, dtype=np.float64)
        peaktrough = np.array(self.total_peaktrough, dtype=object)
        breath_status = np.array(self.breath_status, dtype=object)

        np.savez(filename, data=data, timestamps=t, srate_hz=self.srate_hz, peaktrough=peaktrough, breath_status=breath_status,)
        print(f" Saved: {filename}")
        print(f"Total samples: {len(data)}")
        if len(t) > 1:
            dur = t[-1] - t[0]
            print(f"Duration: {dur:.1f} s | Avg rate: {len(data)/dur:.2f} Hz")
        return filename


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", "-d", type=float, default=60.0, help="recording duration in seconds")
    p.add_argument("--srate", "-r", type=float, default=10.0, help="target sample rate (Hz)")
    p.add_argument("--output", "-o", type=str, default="vernier_resp", help="output file prefix")
    args = p.parse_args()

    rec = VernierRespRecorder(srate_hz=args.srate)
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