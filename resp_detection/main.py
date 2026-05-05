import argparse
import tkinter as tk

from respiration_detection import VernierRespRecorder
from ui import LiveTraceUI
from config import Config

def main():
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

    trace_ui = LiveTraceUI(rec, plot_window_s=15.0, update_ms=100)
    trace_ui.start()

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
        trace_ui.close()
        rec.stop_and_save(prefix=args.output)
        rec.root.destroy()

    tick()
    rec.root.after(int(args.duration * 1000), stop_everything)

    try:
        rec.root.mainloop()
    except KeyboardInterrupt:
        stop_everything()
