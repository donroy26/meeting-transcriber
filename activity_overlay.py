"""
activity_overlay.py - Small floating audio activity meter.

Runs Tkinter in its own thread. The capture thread sends rough mixed-audio levels;
the overlay displays them as moving bars while recording.
"""

from __future__ import annotations

import queue
import random
import sys
import threading
import tkinter as tk
import traceback
from collections import deque
from typing import Any

# Colors for the liquid-fill meter
_GREEN = (46, 189, 118, 255)        # liquid
_GREEN_CREST = (118, 226, 166, 255)  # lighter wave crest
_SILHOUETTE = (58, 62, 66, 255)      # unfilled part of the disc


class ActivityOverlay:
    def __init__(self) -> None:
        # Bounded: set_level pushes ~16x/s and drops on a full queue, so if the
        # Tk thread ever dies the queue stops growing instead of leaking for the
        # rest of the session. 512 is far above what show/hide/stop/get_notes
        # (which still block on put) ever need.
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=512)
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._root: tk.Tk | None = None  # macOS only: Tk lives on the main thread

    def start(self) -> None:
        if sys.platform == "darwin":
            # Cocoa is not thread-safe: Tk must own the main thread. Build the
            # windows here (on the caller's thread); main() runs mainloop().
            if self._root is None:
                self._root = self._build()
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="activity-overlay")
        self._thread.start()
        self._started.wait(timeout=2)

    def mainloop(self) -> None:
        """macOS: run Tk on the calling (main) thread. Blocks until stop()."""
        if self._root is None:
            self._root = self._build()
        self._root.mainloop()

    def _put_control(self, command: str, value: Any = None) -> None:
        """Queue a control message. Bounded wait: if the Tk thread has died the
        queue fills with level samples and never drains, and the recording
        thread must not be the thing that blocks on it."""
        try:
            self._queue.put((command, value), timeout=2)
        except queue.Full:
            print(
                f"[overlay] WARNING: overlay queue full; dropped '{command}' "
                "(the overlay thread is not draining it).",
                file=sys.stderr,
            )

    def show(self) -> None:
        self.start()
        self._put_control("show")

    def hide(self) -> None:
        self._put_control("hide")

    def stop(self) -> None:
        self._put_control("stop")

    def set_level(self, level: float) -> None:
        try:
            self._queue.put_nowait(("level", max(0.0, min(1.0, float(level)))))
        except Exception:
            pass

    def get_notes(self) -> dict[str, str]:
        response: queue.Queue[dict[str, str]] = queue.Queue(maxsize=1)
        self._put_control("get_notes", response)
        try:
            return response.get(timeout=2)
        except queue.Empty:
            return {"attendees": "", "operator_notes": ""}

    def _run(self) -> None:
        try:
            self._build().mainloop()
        except Exception as exc:
            # Make overlay-thread death visible in the log instead of silent.
            print(f"[overlay] ERROR: overlay thread died: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            self._started.set()  # never leave start() blocked on a dead thread

    def _build(self) -> tk.Tk:
        """Create the meter and notes windows and schedule the pump. Returns the
        root; the caller runs ``mainloop()`` on whichever thread owns Tk."""
        root = tk.Tk()
        root.title("Meeting Transcriber")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.88)
        root.configure(bg="#151515")

        width = 188
        height = 54
        canvas = tk.Canvas(
            root,
            width=width,
            height=height,
            bg="#151515",
            highlightthickness=1,
            highlightbackground="#3a3a3a",
        )
        canvas.pack()

        try:
            screen_width = root.winfo_screenwidth()
            root.geometry(f"{width}x{height}+{screen_width - width - 24}+80")
        except Exception:
            root.geometry(f"{width}x{height}+24+80")

        levels: deque[float] = deque([0.0] * 28, maxlen=28)
        visible = False
        root.withdraw()

        # Liquid-fill meter: a disc fills bottom-up with green; the surface
        # sloshes harder when the audio is louder. Falls back to the classic
        # bars if numpy/PIL is unavailable.
        MARK_SIZE = 44
        mark_alpha = None
        try:
            import numpy as _np
            from PIL import Image as _Image
            from PIL import ImageTk as _ImageTk

            _yy, _xx = _np.mgrid[:MARK_SIZE, :MARK_SIZE]
            _c = (MARK_SIZE - 1) / 2
            mark_alpha = (_xx - _c) ** 2 + (_yy - _c) ** 2 <= _c ** 2
        except Exception as exc:
            print(f"[overlay] Liquid meter unavailable ({exc}); using bars.", file=sys.stderr)

        anim = {
            "eased": 0.0,
            "target": 0.0,
            "photo": None,
            # 1-D water surface: per-column height deviation + velocity.
            # Surface tension pulls each column toward its neighbors so
            # random splash impulses ripple outward and settle naturally.
            "h": None if mark_alpha is None else _np.zeros(MARK_SIZE),
            "v": None if mark_alpha is None else _np.zeros(MARK_SIZE),
        }
        _rng = random.Random()

        def _step_water(level: float) -> None:
            h, v = anim["h"], anim["v"]
            # tension toward neighbors (wrap around)
            laplacian = (_np.roll(h, 1) + _np.roll(h, -1)) / 2.0 - h
            v += laplacian * 0.35
            v *= 0.93           # damping: splashes die out
            h += v
            h *= 0.985          # slow settle toward flat
            # random splash impulses, scaled by audio level
            if level > 0.02 and _rng.random() < 0.25 + level * 0.6:
                x = _rng.randrange(MARK_SIZE)
                strength = (0.6 + _rng.random() * 1.8) * level * 3.2
                v[x] -= strength
                if x + 1 < MARK_SIZE:
                    v[x + 1] -= strength * 0.5
                if x - 1 >= 0:
                    v[x - 1] -= strength * 0.5
            _np.clip(h, -7.0, 7.0, out=h)

        def _render_meter() -> None:
            level = max(0.04, anim["eased"])  # tiny puddle even in silence
            s = MARK_SIZE
            ys = _np.arange(s)[:, None]
            surface = (1.0 - level) * (s + 2) - 1 + anim["h"][None, :]
            fill = ys >= surface
            crest = fill & (ys < surface + 2.2)

            out = _np.zeros((s, s, 4), dtype=_np.uint8)
            out[mark_alpha] = _SILHOUETTE
            out[mark_alpha & fill] = _GREEN
            out[mark_alpha & crest] = _GREEN_CREST

            anim["photo"] = _ImageTk.PhotoImage(_Image.fromarray(out))
            canvas.create_image(10, 5, image=anim["photo"], anchor="nw")

        notes = tk.Toplevel(root)
        notes.title("Meeting Notes")
        notes.attributes("-topmost", True)
        notes.configure(bg="#202020")
        notes.protocol("WM_DELETE_WINDOW", notes.withdraw)

        try:
            screen_width = root.winfo_screenwidth()
            notes.geometry(f"380x360+{screen_width - 404}+144")
        except Exception:
            notes.geometry("380x360+24+144")

        attendees_label = tk.Label(
            notes,
            text="Attendees",
            bg="#202020",
            fg="#f4f4f5",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        attendees_label.pack(fill="x", padx=10, pady=(10, 2))

        attendees_entry = tk.Entry(
            notes,
            bg="#111111",
            fg="#f4f4f5",
            insertbackground="#f4f4f5",
            relief="flat",
            font=("Segoe UI", 10),
        )
        attendees_entry.pack(fill="x", padx=10, pady=(0, 10))

        notes_header = tk.Frame(notes, bg="#202020")
        notes_header.pack(fill="x", padx=10, pady=(0, 2))

        notes_label = tk.Label(
            notes_header,
            text="Notes to add",
            bg="#202020",
            fg="#f4f4f5",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        notes_label.pack(side="left", fill="x", expand=True)

        todo_button = tk.Button(
            notes_header,
            text="To-do",
            bg="#2f2f2f",
            fg="#f4f4f5",
            activebackground="#3f3f3f",
            activeforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9),
            padx=8,
            pady=2,
        )
        todo_button.pack(side="right")

        notes_text = tk.Text(
            notes,
            bg="#111111",
            fg="#f4f4f5",
            insertbackground="#f4f4f5",
            relief="flat",
            font=("Segoe UI", 10),
            wrap="word",
            undo=True,
        )
        notes_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def mark_todo_line() -> None:
            notes_text.focus_set()
            line_start = notes_text.index("insert linestart")
            line_end = notes_text.index("insert lineend")
            line = notes_text.get(line_start, line_end)
            stripped = line.lstrip()

            if stripped.startswith("- [ ] TODO:"):
                return

            if line.strip():
                notes_text.insert(line_start, "- [ ] TODO: ")
            else:
                notes_text.delete(line_start, line_end)
                notes_text.insert(line_start, "- [ ] TODO: ")
                notes_text.mark_set("insert", f"{line_start}+12c")

        todo_button.configure(command=mark_todo_line)
        notes.withdraw()
        self._started.set()

        def redraw() -> None:
            canvas.delete("all")
            if mark_alpha is not None:
                # Animate: ease toward the latest level, let the water slosh.
                anim["eased"] += (anim["target"] - anim["eased"]) * 0.25
                _step_water(anim["eased"])
                _render_meter()
                canvas.create_text(
                    66, 20, anchor="w", text="REC",
                    fill="#ff4b4b", font=("Segoe UI", 10, "bold"),
                )
                canvas.create_oval(102, 15, 112, 25, fill="#ff3030", outline="")
                return

            # Fallback: classic bars
            canvas.create_text(
                12, 12, anchor="w", text="REC",
                fill="#ff4b4b", font=("Segoe UI", 9, "bold"),
            )
            canvas.create_oval(44, 8, 54, 18, fill="#ff3030", outline="")
            base_y = 42
            bar_w = 4
            gap = 2
            start_x = 12
            for idx, value in enumerate(levels):
                x = start_x + idx * (bar_w + gap)
                bar_h = max(2, int(value * 30))
                color = "#32d583" if value < 0.72 else "#ffd166"
                canvas.create_rectangle(x, base_y - bar_h, x + bar_w, base_y, fill=color, outline="")

        def pump() -> None:
            # This loop is the overlay's heartbeat. It must ALWAYS reschedule
            # itself — a single uncaught exception here would otherwise stop
            # the pump permanently and the overlay would silently go dead.
            nonlocal visible
            try:
                while True:
                    command, value = self._queue.get_nowait()
                    if command == "show":
                        visible = True
                        attendees_entry.delete(0, "end")
                        notes_text.delete("1.0", "end")
                        # Windows quirk: borderless topmost windows sometimes
                        # fail to re-map after withdraw/deiconify. Re-assert
                        # style, position, and topmost every show.
                        root.deiconify()
                        root.overrideredirect(True)
                        root.attributes("-topmost", True)
                        try:
                            sw = root.winfo_screenwidth()
                            root.geometry(f"{width}x{height}+{sw - width - 24}+80")
                        except tk.TclError:
                            pass
                        root.lift()
                        notes.deiconify()
                        notes.attributes("-topmost", True)
                        notes.lift()
                        attendees_entry.focus_set()
                        print("[overlay] show: meter+notes mapped", flush=True)
                    elif command == "hide":
                        visible = False
                        levels.clear()
                        levels.extend([0.0] * 28)
                        anim["eased"] = 0.0
                        anim["target"] = 0.0
                        if anim["h"] is not None:
                            anim["h"][:] = 0.0
                            anim["v"][:] = 0.0
                        root.withdraw()
                        notes.withdraw()
                    elif command == "level" and value is not None:
                        levels.append(float(value))
                        anim["target"] = float(value)
                    elif command == "get_notes":
                        try:
                            value.put_nowait(
                                {
                                    "attendees": attendees_entry.get().strip(),
                                    "operator_notes": notes_text.get("1.0", "end").strip(),
                                }
                            )
                        except queue.Full:
                            pass
                    elif command == "stop":
                        root.destroy()
                        return
            except queue.Empty:
                pass
            except tk.TclError:
                # Window torn down underneath us; end the pump.
                return
            except Exception as exc:
                print(f"[overlay] WARNING: pump error (continuing): {exc}", file=sys.stderr)

            try:
                if visible:
                    redraw()
            except Exception as exc:
                print(f"[overlay] WARNING: redraw error (continuing): {exc}", file=sys.stderr)
            root.after(50, pump)

        root.after(50, pump)
        return root
