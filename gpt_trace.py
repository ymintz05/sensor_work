import tkinter as tk


class LiveTraceUI:
    def __init__(
        self,
        recorder,
        plot_window_s=15.0,
        update_ms=100,
        width=900,
        height=320,
        use_processed=True,
    ):
        self.rec = recorder
        self.plot_window_s = plot_window_s
        self.update_ms = update_ms
        self.width = width
        self.height = height
        self.use_processed = use_processed

        self.window = None
        self.canvas = None
        self.info_label = None
        self.after_id = None
        self.running = False

    def start(self):
        if self.rec.root is None:
            raise RuntimeError("Recorder root window must exist before starting LiveTraceUI")

        self.window = tk.Toplevel(self.rec.root)
        self.window.title("Respiration Trace")
        self.window.geometry(f"{self.width}x{self.height + 60}")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(
            self.window,
            width=self.width,
            height=self.height,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=False)

        self.info_label = tk.Label(
            self.window,
            text="Waiting for data...",
            font=("Helvetica", 11),
            anchor="w",
            justify="left"
        )
        self.info_label.pack(fill="x", padx=8, pady=6)

        self.running = True
        self.refresh()

    def refresh(self):
        if not self.running:
            return

        if self.window is None or not self.window.winfo_exists():
            self.running = False
            return

        self._draw()
        self.after_id = self.rec.root.after(self.update_ms, self.refresh)

    def _draw(self):
        self.canvas.delete("all")

        width = self.width
        height = self.height
        margin = 20

        self.canvas.create_rectangle(0, 0, width, height, fill="black", outline="")
        self.canvas.create_line(margin, height // 2, width - margin, height // 2, fill="#333333")

        if not self.rec.timestamps:
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="Waiting for data...",
                fill="white",
                font=("Helvetica", 16)
            )
            self.info_label.configure(text="Waiting for data...")
            return

        y_source = self.rec.samples_processed if (
            self.use_processed and len(self.rec.samples_processed) == len(self.rec.timestamps)
        ) else self.rec.samples

        if len(y_source) < 2:
            self.info_label.configure(text="Collecting samples...")
            return

        latest_t = self.rec.timestamps[-1]
        t_min = latest_t - self.plot_window_s

        start_idx = 0
        total_n = len(self.rec.timestamps)
        while start_idx < total_n and self.rec.timestamps[start_idx] < t_min:
            start_idx += 1

        x_ts = self.rec.timestamps[start_idx:]
        y_vals = y_source[start_idx:]

        if len(x_ts) < 2:
            self.info_label.configure(text="Collecting plot window...")
            return

        y_min = min(y_vals)
        y_max = max(y_vals)

        if abs(y_max - y_min) < 1e-9:
            y_min -= 1.0
            y_max += 1.0

        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        coords = []
        for t, y in zip(x_ts, y_vals):
            x_norm = (t - x_ts[0]) / max((x_ts[-1] - x_ts[0]), 1e-9)
            y_norm = (y - y_min) / (y_max - y_min)

            x_pix = margin + x_norm * plot_w
            y_pix = height - margin - y_norm * plot_h

            coords.extend([x_pix, y_pix])

        if len(coords) >= 4:
            self.canvas.create_line(
                *coords,
                fill="#00ff88",
                width=2,
                smooth=False
            )

        for ts_pt, val_pt, label in self.rec.total_peaktrough[-20:]:
            if ts_pt < x_ts[0]:
                continue

            x_norm = (ts_pt - x_ts[0]) / max((x_ts[-1] - x_ts[0]), 1e-9)
            y_norm = (val_pt - y_min) / (y_max - y_min)

            x_pix = margin + x_norm * plot_w
            y_pix = height - margin - y_norm * plot_h

            color = "tomato" if label == "exh start" else "dodgerblue"
            self.canvas.create_oval(
                x_pix - 3, y_pix - 3, x_pix + 3, y_pix + 3,
                fill=color, outline=color
            )

        current_phase = self.rec.breath_status[-1][1] if self.rec.breath_status else "N/A"

        self.canvas.create_text(
            12, 12,
            anchor="nw",
            text=f"Phase: {current_phase}",
            fill="white",
            font=("Helvetica", 12, "bold")
        )

        self.canvas.create_text(
            12, 30,
            anchor="nw",
            text=f"Window: last {self.plot_window_s:.1f}s",
            fill="white",
            font=("Helvetica", 11)
        )

        self.info_label.configure(
            text=(
                f"Showing {len(y_vals)} samples | "
                f"y-range: {y_min:.4f} to {y_max:.4f} | "
                f"source: {'processed' if y_source is self.rec.samples_processed else 'raw'}"
            )
        )

    def close(self):
        self.running = False

        if self.after_id is not None:
            try:
                self.rec.root.after_cancel(self.after_id)
            except Exception:
                pass

        if self.window is not None and self.window.winfo_exists():
            self.window.destroy()