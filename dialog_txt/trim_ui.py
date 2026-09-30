from __future__ import annotations

import queue
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import messagebox, ttk

from .audio_edit import (audio_duration, clock_offset, mixed_peaks, parse_duration,
                         restore_session, session_start, time_text, trim_backup, trim_session)
from .config import MIX_FILE_NAME, MIC_FILE_NAME, DESKTOP_FILE_NAME
from .storage import session_title


class MixedPlayer:
    """Streaming output with a seekable playhead; never loads a whole meeting."""
    def __init__(self, session: Path, events: queue.Queue):
        self.session = session
        self.events = events
        self.stop_event = threading.Event()
        self.thread = None
        self.position = 0.0
        self.generation = 0

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.thread = None
        self.generation += 1

    def play(self, position: float, end: float):
        self.stop()
        self.position = position
        self.stop_event.clear()
        generation = self.generation

        def run():
            sources = []
            try:
                mix = self.session / MIX_FILE_NAME
                paths = [mix] if mix.exists() else [self.session / n for n in
                                                    (MIC_FILE_NAME, DESKTOP_FILE_NAME)
                                                    if (self.session / n).exists()]
                sources = [sf.SoundFile(str(path)) for path in paths]
                rate = sources[0].samplerate
                if any(source.samplerate != rate for source in sources):
                    raise ValueError("Track sample rates differ")
                for source in sources:
                    source.seek(min(source.frames, round(position * rate)))
                current = round(position * rate)
                last = round(end * rate)
                with sd.OutputStream(samplerate=rate, channels=1, dtype="float32", blocksize=2048) as stream:
                    while not self.stop_event.is_set() and current < last:
                        count = min(2048, last - current)
                        mixed = np.zeros((count, 1), dtype=np.float32)
                        read_any = False
                        for source in sources:
                            data = source.read(count, dtype="float32", always_2d=True)
                            if len(data):
                                read_any = True
                                mixed[:len(data), 0] += data.mean(axis=1) / len(sources)
                        if not read_any:
                            break
                        stream.write(mixed.clip(-1, 1))
                        current += count
                        self.position = current / rate
            except Exception as exc:
                self.events.put(("play_error", generation, str(exc)))
            finally:
                for source in sources:
                    source.close()
                self.events.put(("play_done", generation))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


class TrimWindow(tk.Toplevel):
    def __init__(self, app, session: Path):
        super().__init__(app)
        self.app = app
        self.session = session
        self.tr = app._tr
        self.duration = audio_duration(session)
        self.started = session_start(session)
        self.events = queue.Queue()
        self.player = MixedPlayer(session, self.events)
        self.waveform_cancel = threading.Event()
        self.detail_cancel = threading.Event()
        self.waveform_threads = []
        self.busy = False
        self.closed = False
        self.detail_generation = 0
        self.detail_after = None
        self.poll_after = None
        self.overview = np.zeros(1200)
        self.detail = np.zeros(1200)
        self.view_start = 0.0
        self.view_end = self.duration
        self.cutoff = self.duration
        self.cursor = self.duration
        self.edited_field = None
        self.updating_fields = False
        self.updating_seek = False
        self.buttons = []
        self.title(self.tr("trim_title") + " — " + session_title(session))
        self.geometry("920x560")
        self.minsize(860, 540)
        self.transient(app)
        self.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        self.info = ttk.Label(body, wraplength=900,
                              text=self.tr("trim_info", duration=time_text(self.duration),
                                            start=self.started.isoformat(sep=" ", timespec="seconds") if self.started else "—")
                              + "  •  " + self.tr("trim_backup_help"))
        self.info.pack(anchor=tk.W)
        self.message = self.info
        legend = ttk.Frame(body)
        legend.pack(fill=tk.X, pady=(4, 6))
        ttk.Label(legend, text=self.tr("trim_help")).pack(side=tk.LEFT, padx=(0, 12))
        for color, key in (("#ffffff", "trim_legend_cursor"),
                           ("#ffbe63", "trim_legend_cut"),
                           ("#d9535f", "trim_legend_tail")):
            chip = tk.Canvas(legend, width=10, height=10, background=color,
                             highlightthickness=1, highlightbackground="#657080")
            chip.pack(side=tk.LEFT, padx=(5, 3))
            ttk.Label(legend, text=self.tr(key)).pack(side=tk.LEFT)
        self.overview_canvas = tk.Canvas(body, height=70, background="#18212c", highlightthickness=0, takefocus=True)
        self.overview_canvas.pack(fill=tk.X)
        self.overview_canvas.bind("<Button-1>", self._overview_click)
        self.overview_canvas.bind("<B1-Motion>", self._overview_click)
        self.overview_canvas.bind("<Configure>", lambda e: self._draw())
        zoom_row = ttk.Frame(body)
        zoom_row.pack(fill=tk.X, pady=6)
        ttk.Label(zoom_row, text=self.tr("trim_zoom")).pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="30 s")
        zoom = ttk.Combobox(zoom_row, textvariable=self.zoom_var, values=("30 s", "120 s", "600 s", self.tr("trim_all")),
                            state="readonly", width=12)
        zoom.pack(side=tk.LEFT, padx=8)
        zoom.bind("<<ComboboxSelected>>", lambda e: self._request_detail())
        self.range_label = ttk.Label(zoom_row)
        self.range_label.pack(side=tk.LEFT)
        self.detail_canvas = tk.Canvas(body, height=105, background="#18212c", highlightthickness=0, takefocus=True)
        self.detail_canvas.pack(fill=tk.BOTH, expand=True)
        self.detail_canvas.bind("<Button-1>", self._detail_click)
        self.detail_canvas.bind("<B1-Motion>", self._detail_click)
        self.detail_canvas.bind("<Configure>", lambda e: self._draw())
        self.wheel_remainder = 0.0
        for canvas in (self.overview_canvas, self.detail_canvas):
            canvas.bind("<MouseWheel>", self._wheel)
            canvas.bind("<Button-4>", self._wheel)
            canvas.bind("<Button-5>", self._wheel)
        ttk.Label(zoom_row, text=self.tr("trim_keys")).pack(side=tk.RIGHT)
        playback = ttk.Frame(body)
        playback.pack(fill=tk.X, pady=(8, 4))
        self.play_button = self._button(playback, text=self.tr("trim_play"), command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT)
        self.preview_button = self._button(playback, text=self.tr("trim_preview"), command=self._preview)
        self.preview_button.pack(side=tk.LEFT, padx=6)
        self.cut_button = self._button(playback, text=self.tr("trim_cut_here"), command=self._cut_here)
        self.cut_button.pack(side=tk.LEFT)
        steps = ttk.Frame(playback)
        steps.pack(side=tk.LEFT, padx=8)
        for delta in (-30, -5, -1, 1, 5, 30):
            self._button(steps, text=f"{delta:+}", width=4,
                         command=lambda d=delta: self._nudge(d)).pack(side=tk.LEFT, padx=(0, 3))
        self.position_label = ttk.Label(playback)
        self.position_label.pack(side=tk.RIGHT)
        self.seek = ttk.Scale(body, from_=0, to=self.duration, command=self._seek, takefocus=False)
        self.seek.pack(fill=tk.X, pady=(2, 6))
        inputs = ttk.LabelFrame(body, text=self.tr("trim_end_group"), padding=6)
        inputs.pack(fill=tk.X, pady=(0, 6))
        inputs.columnconfigure(3, weight=1)
        ttk.Label(inputs, text=self.tr("trim_duration")).grid(row=0, column=0, sticky=tk.W)
        self.duration_var = tk.StringVar(value=time_text(self.duration))
        duration_entry = self.duration_entry = ttk.Entry(inputs, textvariable=self.duration_var, width=22)
        duration_entry.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))
        duration_entry.bind("<Return>", lambda e: self._from_duration())
        self._button(inputs, text=self.tr("trim_set"), command=self._from_duration).grid(row=0, column=2, padx=(6, 0))
        ttk.Label(inputs, text=self.tr("trim_clock")).grid(row=1, column=0, sticky=tk.W, pady=(3, 0))
        self.clock_var = tk.StringVar()
        self.clock_entry = ttk.Entry(inputs, textvariable=self.clock_var, width=22)
        self.clock_entry.grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(3, 0))
        self.clock_entry.bind("<Return>", lambda e: self._from_clock())
        self.clock_button = self._button(inputs, text=self.tr("trim_set"), command=self._from_clock)
        self.clock_button.grid(row=1, column=2, padx=(6, 0), pady=(3, 0))
        if self.started is None:
            self.clock_entry.configure(state=tk.DISABLED)
            self.clock_button.configure(state=tk.DISABLED)
        self.summary = ttk.Label(inputs)
        self.summary.grid(row=0, column=3, sticky=tk.W, padx=(12, 0))
        self.remove_summary = ttk.Label(inputs)
        self.remove_summary.grid(row=1, column=3, sticky=tk.W, padx=(12, 0), pady=(3, 0))
        actions = ttk.Frame(body)
        actions.pack(fill=tk.X, pady=(2, 0))
        self.apply_button = self._button(actions, text=self.tr("trim_apply"), command=self._apply)
        self.apply_button.pack(side=tk.RIGHT)
        self.undo_button = self._button(actions, text=self.tr("trim_undo"), command=self._undo,
                                      state=tk.NORMAL if trim_backup(session) else tk.DISABLED)
        self.undo_button.pack(side=tk.LEFT)
        self._button(actions, text=self.tr("trim_close"), command=self.close).pack(side=tk.RIGHT, padx=8)
        self.duration_var.trace_add("write", lambda *args: self._input_changed("duration"))
        self.clock_var.trace_add("write", lambda *args: self._input_changed("clock"))
        self._set_cutoff(self.duration)
        self.grab_set()
        self._install_keys()
        self.detail_canvas.focus_set()
        self._worker("overview", lambda: mixed_peaks(session, 0, self.duration, cancel=self.waveform_cancel))
        self._request_detail()
        self._poll()

    def _button(self, parent, **options):
        button = ttk.Button(parent, takefocus=False, **options)
        self.buttons.append(button)
        # Some Windows themes focus a button on mouse down despite takefocus=0.
        button.bind("<FocusIn>", lambda event: self.detail_canvas.focus_set())
        return button

    def _install_keys(self):
        # Prepend our tag so Space wins over Entry, Button and Scale class bindings.
        self.key_tag = f"TrimKeys{id(self)}"
        def tag(widget):
            widget.bindtags((self.key_tag,) + widget.bindtags())
            for child in widget.winfo_children():
                tag(child)
        tag(self)
        self.bind_class(self.key_tag, "<space>", self._space)
        for modifier, step in (("", 1), ("Shift-", 5), ("Control-", 30)):
            for key, direction in (("Left", -1), ("Right", 1)):
                self.bind_class(self.key_tag, f"<{modifier}{key}>",
                                lambda event, delta=step * direction: self._arrow(delta))

    def _space(self, event=None):
        self._toggle_play()
        return "break"

    def _arrow(self, delta):
        self._nudge(delta)
        return "break"

    def _wheel(self, event):
        if self.busy:
            return "break"
        if getattr(event, "num", None) in (4, 5):
            ticks = 1 if event.num == 4 else -1
        else:
            self.wheel_remainder += event.delta / 120
            ticks = int(self.wheel_remainder)
            self.wheel_remainder -= ticks
        if ticks:
            step = 30 if event.state & 0x4 else 5 if event.state & 0x1 else 1
            self._nudge(-ticks * step)
        return "break"

    def _clock_text(self, value):
        if self.started is None:
            return ""
        end = self.started + timedelta(seconds=value)
        if self.duration >= 86400:
            return end.isoformat(timespec="milliseconds")
        return end.strftime("%H:%M:%S.%f")[:-3]

    def _typed_cutoff(self):
        if self.edited_field == "duration":
            return parse_duration(self.duration_var.get())
        if self.edited_field == "clock" and self.started:
            return clock_offset(self.clock_var.get(), self.started, self.duration)
        return self.cutoff

    def _worker(self, kind, operation, generation=None):
        def run():
            try:
                if kind == "saved":
                    # Release Windows file handles before replacing any tracks.
                    for thread in self.waveform_threads:
                        thread.join()
                self.events.put((kind, generation, operation()))
            except Exception as exc:
                self.events.put(("error", kind, str(exc)))
        thread = threading.Thread(target=run, daemon=True)
        if kind in ("overview", "detail"):
            self.waveform_threads = [t for t in self.waveform_threads if t.is_alive()]
            self.waveform_threads.append(thread)
        thread.start()

    def _draw_canvas(self, canvas, peaks, start, end):
        canvas.delete("all")
        width, height = max(1, canvas.winfo_width()), canvas.winfo_height()
        middle = height / 2
        canvas.create_line(0, middle, width, middle, fill="#435063")
        if len(peaks):
            for x in range(width):
                left = x * len(peaks) // width
                right = max(left + 1, (x + 1) * len(peaks) // width)
                amplitude = min(1, float(np.max(peaks[left:right]))) * (middle - 12)
                canvas.create_line(x, middle - amplitude, x, middle + amplitude, fill="#5dd5bf")
        span = max(end - start, 0.001)
        cut_x = (self.cutoff - start) / span * width
        if cut_x < width:
            canvas.create_rectangle(max(0, cut_x), 0, width, height, fill="#d9535f", stipple="gray50", outline="")
        if 0 <= cut_x <= width:
            canvas.create_line(cut_x, 0, cut_x, height, fill="#ffbe63", width=2)
        play_x = (self.cursor - start) / span * width
        if 0 <= play_x <= width:
            canvas.create_line(play_x, 0, play_x, height, fill="white", width=2, tags="playhead")
        canvas.create_text(5, height - 8, text=time_text(start), fill="#d5dce5", anchor=tk.W)
        canvas.create_text(width - 5, height - 8, text=time_text(end), fill="#d5dce5", anchor=tk.E)

    def _draw(self):
        self._draw_canvas(self.overview_canvas, self.overview, 0, self.duration)
        self._draw_canvas(self.detail_canvas, self.detail, self.view_start, self.view_end)

    def _request_detail(self):
        if self.closed or self.busy:
            return
        self.detail_after = None
        self.detail_cancel.set()
        self.detail_cancel = threading.Event()
        cancel = self.detail_cancel
        self.detail_generation += 1
        generation = self.detail_generation
        span = self.duration if self.zoom_var.get() == self.tr("trim_all") else float(self.zoom_var.get().split()[0])
        span = min(span, self.duration)
        self.view_start = max(0, min(self.cursor - span / 2, self.duration - span))
        self.view_end = self.view_start + span
        self.detail = np.zeros(1200)
        self.range_label.configure(text=f"{time_text(self.view_start)} — {time_text(self.view_end)}")
        self._draw()
        first, last = self.view_start, self.view_end
        self._worker("detail", lambda: mixed_peaks(self.session, first, last, cancel=cancel), generation)

    def _input_changed(self, field):
        if self.updating_fields:
            return
        self.edited_field = field
        try:
            value = self._typed_cutoff()
            valid = not self.busy and 0 < value < self.duration
        except (ValueError, TypeError, OverflowError):
            valid = False
        self.apply_button.configure(state=tk.NORMAL if valid else tk.DISABLED)

    def _set_cutoff(self, value, recenter=True):
        self.cutoff = max(0, min(self.duration, value))
        self.updating_fields = True
        try:
            self.duration_var.set(time_text(self.cutoff))
            self.clock_var.set(self._clock_text(self.cutoff))
        finally:
            self.updating_fields = False
        self.edited_field = None
        self.summary.configure(text=self.tr("trim_keep", keep=time_text(self.cutoff)))
        self.remove_summary.configure(text=self.tr("trim_remove", remove=time_text(self.duration - self.cutoff)))
        self.apply_button.configure(state=tk.NORMAL if not self.busy and 0 < self.cutoff < self.duration else tk.DISABLED)
        self._draw()
        if recenter:
            if self.detail_after:
                self.after_cancel(self.detail_after)
            self.detail_after = self.after(180, self._request_detail)

    def _select(self, value, recenter):
        if self.busy:
            return
        self._move_cursor(value, recenter)
        self._set_cutoff(value, recenter)

    def _move_cursor(self, value, recenter=True):
        if self.busy:
            return
        playing = self.player.thread is not None and self.player.thread.is_alive()
        self._pause()
        self.cursor = max(0, min(self.duration, value))
        self._update_seek()
        self._update_position()
        self._draw()
        if recenter:
            if self.detail_after:
                self.after_cancel(self.detail_after)
            self.detail_after = self.after(120, self._request_detail)
        if playing and self.cursor < self.duration:
            self._play()

    def _update_seek(self):
        self.updating_seek = True
        try:
            self.seek.set(self.cursor)
        finally:
            self.updating_seek = False

    def _update_position(self):
        self.position_label.configure(text=self.tr("trim_position", position=time_text(self.cursor),
                                                  clock=self._clock_text(self.cursor)))

    def _sync_playhead(self):
        self._update_seek()
        self._update_position()
        # Update only the cursor while playing. Rebuilding the entire waveform
        # every tick is unnecessary, and pause must paint the final position now.
        for canvas, start, end in ((self.overview_canvas, 0, self.duration),
                                   (self.detail_canvas, self.view_start, self.view_end)):
            canvas.delete("playhead")
            if start <= self.cursor <= end:
                x = (self.cursor - start) / max(end - start, .001) * canvas.winfo_width()
                canvas.create_line(x, 0, x, canvas.winfo_height(), fill="white", width=2, tags="playhead")

    def _nudge(self, delta):
        position = self.player.position if self.player.thread is not None else self.cursor
        self._move_cursor(position + delta)

    def _cut_here(self):
        if not self.busy:
            if self.player.thread is not None:
                self.cursor = self.player.position
            self._set_cutoff(self.cursor, False)

    def _overview_click(self, event):
        self.detail_canvas.focus_set()
        self._move_cursor(max(0, min(1, event.x / max(1, self.overview_canvas.winfo_width()))) * self.duration, True)

    def _detail_click(self, event):
        fraction = max(0, min(1, event.x / max(1, self.detail_canvas.winfo_width())))
        self.detail_canvas.focus_set()
        self._move_cursor(self.view_start + fraction * (self.view_end - self.view_start), False)

    def _from_duration(self):
        try:
            self._validate_selection(parse_duration(self.duration_var.get()))
        except (ValueError, OverflowError) as exc:
            self._input_error(exc)

    def _from_clock(self):
        if self.started:
            try:
                self._validate_selection(clock_offset(self.clock_var.get(), self.started, self.duration))
            except (ValueError, TypeError, OverflowError) as exc:
                self._input_error(exc)

    def _validate_selection(self, value):
        if not 0 < value <= self.duration:
            raise ValueError(self.tr("trim_outside"))
        self._select(value, True)

    def _input_error(self, exc):
        messagebox.showerror(self.tr("trim_title"), self.tr("trim_input_error", error=exc), parent=self)

    def _seek(self, value):
        if self.busy or self.updating_seek:
            return
        # Programmatic Scale updates need not interrupt ongoing playback.
        if abs(float(value) - self.cursor) < 0.06:
            return
        self._move_cursor(float(value))

    def _toggle_play(self):
        if self.busy:
            return
        if self.player.thread is not None and self.player.thread.is_alive():
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self.busy:
            if self.cursor >= self.duration:
                self.cursor = 0
            self.player.play(self.cursor, self.duration)
            self.play_button.configure(text=self.tr("trim_pause"))
            self._sync_playhead()
            if not self.view_start <= self.cursor < self.view_end:
                self._request_detail()

    def _pause(self):
        was_playing = self.player.thread is not None
        self.player.stop()
        if was_playing:
            self.cursor = self.player.position
        self.play_button.configure(text=self.tr("trim_play"))
        self._sync_playhead()

    def _preview(self):
        if not self.busy:
            self.cursor = max(0, self.cutoff - 5)
            self.player.play(self.cursor, min(self.duration, self.cutoff + 5))
            self.play_button.configure(text=self.tr("trim_pause"))
            self._sync_playhead()
            if not self.view_start <= self.cursor < self.view_end:
                self._request_detail()

    def _apply(self):
        if self.busy:
            return
        # The field edited last is authoritative, including pasted date/time.
        try:
            self._validate_selection(self._typed_cutoff())
        except (ValueError, TypeError, OverflowError) as exc:
            self._input_error(exc)
            return
        if not 0 < self.cutoff < self.duration:
            return
        self._begin_edit()
        cutoff = self.cutoff
        self._worker("saved", lambda: trim_session(self.session, cutoff))

    def _undo(self):
        if not self.busy:
            self._begin_edit()
            self._worker("saved", lambda: restore_session(self.session))

    def _begin_edit(self):
        self._pause()
        self.busy = True
        self.waveform_cancel.set()
        self.detail_cancel.set()
        self.detail_generation += 1
        if self.detail_after:
            self.after_cancel(self.detail_after)
            self.detail_after = None
        for button in self.buttons:
            button.configure(state=tk.DISABLED)
        self.message.configure(text=self.tr("trim_saving"))

    def _poll(self):
        if self.closed:
            return
        old_cursor = self.cursor
        while True:
            try:
                kind, generation, *payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "overview":
                self.overview = payload[0]
                self._draw()
            elif kind == "detail" and generation == self.detail_generation:
                self.detail = payload[0]
                self._draw()
            elif kind == "saved":
                self.busy = False
                self.app._refresh_recordings()
                self.app._set_status(self.tr("trim_saved"))
                self.close()
                return
            elif kind == "error":
                if generation == "saved":
                    self.busy = False
                    self.waveform_cancel.clear()
                    for button in self.buttons:
                        button.configure(state=tk.NORMAL)
                    if self.started is None:
                        self.clock_button.configure(state=tk.DISABLED)
                    self._set_cutoff(self.cutoff, False)
                    self.undo_button.configure(state=tk.NORMAL if trim_backup(self.session) else tk.DISABLED)
                    self.play_button.configure(state=tk.NORMAL)
                    self.preview_button.configure(state=tk.NORMAL)
                self.message.configure(text=payload[0])
            elif kind == "play_error" and generation == self.player.generation:
                self.message.configure(text=payload[0])
            elif kind == "play_done" and generation == self.player.generation:
                self.cursor = self.player.position
                self.player.stop()
                self.play_button.configure(text=self.tr("trim_play"))
        if self.player.thread:
            self.cursor = self.player.position
        self._sync_playhead()
        if old_cursor != self.cursor:
            if not self.view_start <= self.cursor < self.view_end and not self.busy:
                self._request_detail()
        self.poll_after = self.after(50, self._poll)

    def close(self, restart_monitor=True):
        if self.busy:
            return False
        self.closed = True
        self.player.stop()
        self.waveform_cancel.set()
        self.detail_cancel.set()
        for thread in self.waveform_threads:
            thread.join()
        if self.detail_after:
            self.after_cancel(self.detail_after)
        if self.poll_after:
            self.after_cancel(self.poll_after)
        self.grab_release()
        for key in ("<space>", "<Left>", "<Right>", "<Shift-Left>", "<Shift-Right>", "<Control-Left>", "<Control-Right>"):
            self.unbind_class(self.key_tag, key)
        self.destroy()
        self.app.trim_window = None
        if restart_monitor:
            self.app._start_idle_level_monitor(restart=True)
        return True
