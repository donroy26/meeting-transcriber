"""
activity_overlay.py - Small floating audio activity meter.

Runs Tkinter in its own thread. The capture thread sends rough mixed-audio levels;
the overlay displays them as moving bars while recording.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import deque
from typing import Any


class ActivityOverlay:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="activity-overlay")
        self._thread.start()
        self._started.wait(timeout=2)

    def show(self) -> None:
        self.start()
        self._queue.put(("show", None))

    def hide(self) -> None:
        self._queue.put(("hide", None))

    def stop(self) -> None:
        self._queue.put(("stop", None))

    def set_level(self, level: float) -> None:
        try:
            self._queue.put_nowait(("level", max(0.0, min(1.0, float(level)))))
        except Exception:
            pass

    def get_notes(self) -> dict[str, str]:
        response: queue.Queue[dict[str, str]] = queue.Queue(maxsize=1)
        self._queue.put(("get_notes", response))
        try:
            return response.get(timeout=2)
        except queue.Empty:
            return {"attendees": "", "operator_notes": ""}

    def _run(self) -> None:
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
            canvas.create_text(
                12,
                12,
                anchor="w",
                text="REC",
                fill="#ff4b4b",
                font=("Segoe UI", 9, "bold"),
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
            nonlocal visible
            try:
                while True:
                    command, value = self._queue.get_nowait()
                    if command == "show":
                        visible = True
                        attendees_entry.delete(0, "end")
                        notes_text.delete("1.0", "end")
                        root.deiconify()
                        root.lift()
                        notes.deiconify()
                        notes.lift()
                        attendees_entry.focus_set()
                    elif command == "hide":
                        visible = False
                        levels.clear()
                        levels.extend([0.0] * 28)
                        root.withdraw()
                        notes.withdraw()
                    elif command == "level" and value is not None:
                        levels.append(float(value))
                    elif command == "get_notes":
                        value.put(
                            {
                                "attendees": attendees_entry.get().strip(),
                                "operator_notes": notes_text.get("1.0", "end").strip(),
                            }
                        )
                    elif command == "stop":
                        root.destroy()
                        return
            except queue.Empty:
                pass

            if visible:
                redraw()
            root.after(50, pump)

        root.after(50, pump)
        root.mainloop()
