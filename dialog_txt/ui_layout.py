from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from .localization import UI_LANGUAGE_CODES
from .settings import ALLOWED_COMPUTE_TYPES
from .transcription_backends import ALLOWED_DEVICES, ALLOWED_TRANSCRIPTION_LIBRARIES


class HoverTooltip:
    def __init__(
        self,
        widget: tk.Widget,
        text_getter: Callable[[], str],
        *,
        delay_ms: int = 350,
    ) -> None:
        self.widget = widget
        self.text_getter = text_getter
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tooltip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")
        self.widget.bind("<Destroy>", self._on_destroy, add="+")

    def _on_enter(self, _event=None) -> None:
        self._cancel_pending()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event=None) -> None:
        self._cancel_pending()
        self._hide()

    def _on_destroy(self, _event=None) -> None:
        self._cancel_pending()
        self._hide()

    def _cancel_pending(self) -> None:
        if self._after_id is None:
            return
        self.widget.after_cancel(self._after_id)
        self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tooltip_window is not None:
            return
        text = self.text_getter().strip()
        if not text:
            return
        x, y = self.widget.winfo_pointerxy()
        tooltip_window = tk.Toplevel(self.widget)
        tooltip_window.wm_overrideredirect(True)
        try:
            tooltip_window.attributes("-topmost", True)
        except tk.TclError:
            pass
        tooltip_window.wm_geometry(f"+{x + 14}+{y + 14}")
        label = tk.Label(
            tooltip_window,
            text=text,
            justify=tk.LEFT,
            wraplength=360,
            bg="#fffbe6",
            fg="#202020",
            relief=tk.SOLID,
            bd=1,
            padx=6,
            pady=4,
        )
        label.pack()
        self._tooltip_window = tooltip_window

    def _hide(self) -> None:
        if self._tooltip_window is None:
            return
        self._tooltip_window.destroy()
        self._tooltip_window = None


def build_ui(app) -> None:
    style = ttk.Style(app)
    style.configure("Treeview", rowheight=19)
    style.configure("Compact.Horizontal.TProgressbar", thickness=8)

    top = ttk.Frame(app, padding=6)
    top.pack(fill=tk.BOTH, expand=True)

    app.controls_box = ttk.LabelFrame(top, text=app._tr("group_recording"), padding=6)
    app.controls_box.pack(fill=tk.X)
    app.controls_box.columnconfigure(0, weight=1)

    mic_row = ttk.Frame(app.controls_box)
    mic_row.grid(row=0, column=0, sticky=tk.EW)
    mic_row.columnconfigure(3, weight=1)
    app.mic_label = ttk.Label(mic_row, text=app._tr("label_microphone"))
    app.mic_label.grid(row=0, column=0, sticky=tk.W)
    app.mic_combo = ttk.Combobox(mic_row, state="readonly", width=34)
    app.mic_combo.grid(row=0, column=1, sticky=tk.W, padx=(2, 0))
    app.mic_combo.bind("<<ComboboxSelected>>", app._on_mic_selected)

    app.refresh_mic_button = ttk.Button(mic_row, text="↻", width=3, command=app._refresh_microphones)
    app.refresh_mic_button.grid(row=0, column=2, padx=(4, 0), sticky=tk.W)

    app.ui_language_frame = ttk.Frame(mic_row)
    app.ui_language_frame.grid(row=0, column=3)
    app.ui_language_label = ttk.Label(app.ui_language_frame, text=app._tr("label_ui_language"))
    app.ui_language_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
    app.ui_language_combo = ttk.Combobox(
        app.ui_language_frame,
        state="readonly",
        width=4,
        values=list(UI_LANGUAGE_CODES.keys()),
        textvariable=app.ui_language_code_var,
    )
    app.ui_language_combo.grid(row=0, column=1, sticky=tk.W)
    app.ui_language_combo.bind("<<ComboboxSelected>>", app._on_ui_language_selected)

    app.record_button = tk.Button(
        mic_row,
        text=app._tr("record_start"),
        command=app._toggle_recording,
        bg="#1f8b4c",
        fg="white",
        activebackground="#176a38",
        activeforeground="white",
        disabledforeground="#d8d8d8",
        relief=tk.FLAT,
        bd=0,
        padx=12,
        pady=5,
    )
    app.record_button.grid(row=0, column=4, sticky=tk.E)

    status_row = ttk.Frame(app.controls_box)
    status_row.grid(row=1, column=0, sticky=tk.EW, pady=(4, 0))
    status_row.columnconfigure(0, weight=1)

    app.status_label = ttk.Label(status_row, text="")
    app.status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

    app.pending_short_name_label = ttk.Label(status_row, text=app._tr("label_next_short_name"))
    app.pending_short_name_label.grid(row=0, column=1, sticky=tk.E, padx=(0, 4))
    app.pending_short_name_entry = ttk.Entry(
        status_row,
        width=24,
        textvariable=app.pending_short_name_var,
    )
    app.pending_short_name_entry.grid(row=0, column=2, sticky=tk.E)
    app._update_status_line()

    levels_row = ttk.Frame(app.controls_box)
    levels_row.grid(row=2, column=0, sticky=tk.EW, pady=(2, 0))
    levels_row.columnconfigure(1, weight=1)
    levels_row.columnconfigure(4, weight=1)

    app.level_mic_label = ttk.Label(levels_row, text=app._tr("label_level_mic"))
    app.level_mic_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
    app.mic_level = ttk.Progressbar(
        levels_row,
        mode="determinate",
        maximum=100,
        style="Compact.Horizontal.TProgressbar",
    )
    app.mic_level.grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))

    app.level_desktop_label = ttk.Label(levels_row, text=app._tr("label_level_desktop"))
    app.level_desktop_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 4))
    app.desktop_level = ttk.Progressbar(
        levels_row,
        mode="determinate",
        maximum=100,
        style="Compact.Horizontal.TProgressbar",
    )
    app.desktop_level.grid(row=0, column=4, sticky=tk.EW)

    app.transcribe_box = ttk.LabelFrame(top, text=app._tr("group_transcription"), padding=6)
    app.transcribe_box.pack(fill=tk.X, pady=(6, 0))
    app.transcribe_box.columnconfigure(1, weight=1)

    primary_row = ttk.Frame(app.transcribe_box)
    primary_row.grid(row=0, column=0, columnspan=4, sticky=tk.EW)
    primary_row.columnconfigure(2, weight=1)
    primary_row.columnconfigure(4, weight=1)
    app.auto_transcribe_check = ttk.Checkbutton(
        primary_row,
        text=app._tr("auto_transcribe_after_record"),
        variable=app.auto_transcribe_var,
    )
    app.auto_transcribe_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
    app.self_label_label = ttk.Label(primary_row, text=app._tr("label_speaker_mic"))
    app.self_label_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 4))
    app.self_label_entry = ttk.Entry(primary_row, textvariable=app.self_label_var)
    app.self_label_entry.grid(row=0, column=2, sticky=tk.EW, padx=(0, 8))
    app.other_label_label = ttk.Label(primary_row, text=app._tr("label_speaker_desktop"))
    app.other_label_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 4))
    app.other_label_entry = ttk.Entry(primary_row, textvariable=app.other_label_var)
    app.other_label_entry.grid(row=0, column=4, sticky=tk.EW)

    backend_row = ttk.Frame(app.transcribe_box)
    backend_row.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
    app.library_label = ttk.Label(backend_row, text=app._tr("label_library"))
    app.library_label.grid(row=0, column=0, sticky=tk.W)
    app.library_combo = ttk.Combobox(
        backend_row,
        state="readonly",
        width=14,
        values=list(ALLOWED_TRANSCRIPTION_LIBRARIES),
        textvariable=app.transcription_library_var,
    )
    app.library_combo.grid(row=0, column=1, sticky=tk.W, padx=(3, 8))
    app.library_combo.bind("<<ComboboxSelected>>", app._on_transcription_library_selected)
    app.model_label = ttk.Label(backend_row, text=app._tr("label_model"))
    app.model_label.grid(row=0, column=2, sticky=tk.W)
    app.model_combo = ttk.Combobox(
        backend_row,
        state="readonly",
        width=14,
        values=[],
        textvariable=app.model_var,
    )
    app.model_combo.grid(row=0, column=3, sticky=tk.W, padx=(3, 8))
    app.device_label = ttk.Label(backend_row, text=app._tr("label_device"))
    app.device_label.grid(row=0, column=4, sticky=tk.W)
    app.device_combo = ttk.Combobox(
        backend_row,
        state="readonly",
        width=5,
        values=list(ALLOWED_DEVICES),
        textvariable=app.device_var,
    )
    app.device_combo.grid(row=0, column=5, sticky=tk.W, padx=(3, 8))
    app.compute_label = ttk.Label(backend_row, text=app._tr("label_compute"))
    app.compute_label.grid(row=0, column=6, sticky=tk.W)
    app.compute_type_combo = ttk.Combobox(
        backend_row,
        state="readonly",
        width=8,
        values=list(ALLOWED_COMPUTE_TYPES),
        textvariable=app.compute_type_var,
    )
    app.compute_type_combo.grid(row=0, column=7, sticky=tk.W)

    options_row = ttk.Frame(app.transcribe_box)
    options_row.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))
    app.vad_filter_check = ttk.Checkbutton(
        options_row,
        text=app._tr("vad_filter"),
        variable=app.vad_filter_var,
    )
    app.vad_filter_check.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
    app.include_timestamps_check = ttk.Checkbutton(
        options_row,
        text=app._tr("include_timestamps"),
        variable=app.include_timestamps_var,
    )
    app.include_timestamps_check.grid(row=0, column=1, sticky=tk.W, padx=(0, 8))
    app.transcribe_mix_track_check = ttk.Checkbutton(
        options_row,
        text=app._tr("transcribe_mix_track"),
        variable=app.transcribe_mix_track_var,
    )
    app.transcribe_mix_track_check.grid(row=0, column=2, sticky=tk.W, padx=(0, 8))
    app.language_label = ttk.Label(options_row, text=app._tr("label_language"))
    app.language_label.grid(row=0, column=3, sticky=tk.W)
    app.language_combo = ttk.Combobox(
        options_row,
        width=5,
        values=["ru", "en", "auto"],
        textvariable=app.language_var,
    )
    app.language_combo.grid(row=0, column=4, sticky=tk.W, padx=(3, 8))
    app.beam_label = ttk.Label(options_row, text=app._tr("label_beam"))
    app.beam_label.grid(row=0, column=5, sticky=tk.W)
    app.beam_spinbox = ttk.Spinbox(
        options_row,
        from_=1,
        to=10,
        width=4,
        textvariable=app.beam_size_var,
    )
    app.beam_spinbox.grid(row=0, column=6, sticky=tk.W)

    app.progress_label_title = ttk.Label(app.transcribe_box, text=app._tr("label_progress"))
    app.progress_label_title.grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
    app.progress = ttk.Progressbar(app.transcribe_box, mode="determinate", maximum=100)
    app.progress.grid(row=3, column=1, sticky=tk.EW, pady=(6, 0), padx=(6, 0))
    app.progress_label = ttk.Label(app.transcribe_box, text="0%")
    app.progress_label.grid(row=3, column=2, sticky=tk.W, padx=(6, 0), pady=(6, 0))

    app.cancel_transcribe_button = ttk.Button(
        app.transcribe_box,
        text=app._tr("cancel"),
        command=app._cancel_transcription,
        state=tk.DISABLED,
    )
    app.cancel_transcribe_button.grid(row=3, column=3, sticky=tk.E, padx=(6, 0), pady=(6, 0))

    app.recordings_box = ttk.LabelFrame(top, text=app._tr("group_recordings"), padding=6)
    app.recordings_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    recordings_actions = ttk.Frame(app.recordings_box)
    recordings_actions.pack(fill=tk.X, pady=(0, 4))
    app.transcribe_selected_button = ttk.Button(
        recordings_actions,
        text=app._tr("btn_queue_selected"),
        command=app._transcribe_selected,
    )
    app.transcribe_selected_button.grid(row=0, column=0, sticky=tk.W)
    app.queue_untranscribed_button = ttk.Button(
        recordings_actions,
        text=app._tr("btn_queue_untranscribed"),
        command=app._queue_untranscribed,
    )
    app.queue_untranscribed_button.grid(row=0, column=1, sticky=tk.W, padx=(4, 0))
    app.refresh_recordings_button = ttk.Button(
        recordings_actions,
        text=app._tr("btn_refresh_list"),
        command=app._refresh_recordings,
    )
    app.refresh_recordings_button.grid(row=0, column=2, sticky=tk.W, padx=(4, 0))
    app.open_folder_button = ttk.Button(
        recordings_actions,
        text=app._tr("btn_open_folder"),
        command=app._open_selected_folder,
    )
    app.open_folder_button.grid(row=0, column=3, sticky=tk.W, padx=(4, 0))
    app.delete_selected_button = ttk.Button(
        recordings_actions,
        text=app._tr("btn_delete_selected"),
        command=app._delete_selected_recordings,
    )
    app.delete_selected_button.grid(row=0, column=4, sticky=tk.W, padx=(4, 0))

    columns = ("session", "short_name", "duration", "audio", "txt", "queue")
    app.recordings_tree = ttk.Treeview(app.recordings_box, columns=columns, show="headings")
    app.recordings_tree.heading("session", text=app._tr("col_session"))
    app.recordings_tree.heading("short_name", text=app._tr("col_short_name"))
    app.recordings_tree.heading("duration", text=app._tr("col_duration"))
    app.recordings_tree.heading("audio", text=app._tr("col_audio"))
    app.recordings_tree.heading("txt", text=app._tr("col_txt"))
    app.recordings_tree.heading("queue", text=app._tr("col_queue"))
    app.recordings_tree.column("session", width=132, anchor=tk.CENTER, stretch=False)
    app.recordings_tree.column("short_name", width=185, anchor=tk.W, stretch=True)
    app.recordings_tree.column("duration", width=100, anchor=tk.CENTER, stretch=False)
    app.recordings_tree.column("audio", width=72, anchor=tk.CENTER, stretch=False)
    app.recordings_tree.column("txt", width=48, anchor=tk.CENTER, stretch=False)
    app.recordings_tree.column("queue", width=55, anchor=tk.CENTER, stretch=False)
    app.recordings_tree.pack(fill=tk.BOTH, expand=True)
    app.recordings_tree.bind("<<TreeviewSelect>>", app._on_recording_selected)
    app.recordings_tree.bind("<Double-1>", app._on_recording_double_click)
    app.recordings_tree.bind("<Delete>", app._on_recording_delete_key)
    app.recordings_tree.bind("<F2>", app._on_recording_rename_key)

    app.log_box = ttk.LabelFrame(top, text=app._tr("group_log"), padding=6)
    app.log_box.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
    app.log_text = ScrolledText(app.log_box, height=4, wrap=tk.WORD, state=tk.DISABLED)
    app.log_text.pack(fill=tk.BOTH, expand=True)
    _install_settings_tooltips(app)


def _install_settings_tooltips(app) -> None:
    app._tooltips = []
    tooltip_targets: list[tuple[str, tuple[tk.Widget, ...]]] = [
        ("tooltip_auto_transcribe", (app.auto_transcribe_check,)),
        ("tooltip_speaker_mic", (app.self_label_label, app.self_label_entry)),
        ("tooltip_speaker_desktop", (app.other_label_label, app.other_label_entry)),
        ("tooltip_library", (app.library_label, app.library_combo)),
        ("tooltip_model", (app.model_label, app.model_combo)),
        ("tooltip_device", (app.device_label, app.device_combo)),
        ("tooltip_compute", (app.compute_label, app.compute_type_combo)),
        ("tooltip_vad_filter", (app.vad_filter_check,)),
        ("tooltip_include_timestamps", (app.include_timestamps_check,)),
        ("tooltip_transcribe_mix_track", (app.transcribe_mix_track_check,)),
        ("tooltip_language", (app.language_label, app.language_combo)),
        ("tooltip_beam", (app.beam_label, app.beam_spinbox)),
    ]
    for translation_key, widgets in tooltip_targets:
        for widget in widgets:
            app._tooltips.append(
                HoverTooltip(widget, lambda key=translation_key: app._tr(key))
            )


def apply_localization(app, refresh_data: bool = False) -> None:
    app.controls_box.configure(text=app._tr("group_recording"))
    app.mic_label.configure(text=app._tr("label_microphone"))
    app.ui_language_label.configure(text=app._tr("label_ui_language"))
    app.pending_short_name_label.configure(text=app._tr("label_next_short_name"))
    app.level_mic_label.configure(text=app._tr("label_level_mic"))
    app.level_desktop_label.configure(text=app._tr("label_level_desktop"))
    app.transcribe_box.configure(text=app._tr("group_transcription"))
    app.auto_transcribe_check.configure(text=app._tr("auto_transcribe_after_record"))
    app.self_label_label.configure(text=app._tr("label_speaker_mic"))
    app.other_label_label.configure(text=app._tr("label_speaker_desktop"))
    app.include_timestamps_check.configure(text=app._tr("include_timestamps"))
    app.transcribe_mix_track_check.configure(text=app._tr("transcribe_mix_track"))
    app.library_label.configure(text=app._tr("label_library"))
    app.device_label.configure(text=app._tr("label_device"))
    app.model_label.configure(text=app._tr("label_model"))
    app.language_label.configure(text=app._tr("label_language"))
    app.beam_label.configure(text=app._tr("label_beam"))
    app.compute_label.configure(text=app._tr("label_compute"))
    app.vad_filter_check.configure(text=app._tr("vad_filter"))
    app.progress_label_title.configure(text=app._tr("label_progress"))
    app.cancel_transcribe_button.configure(text=app._tr("cancel"))
    app.recordings_box.configure(text=app._tr("group_recordings"))
    app.transcribe_selected_button.configure(text=app._tr("btn_queue_selected"))
    app.queue_untranscribed_button.configure(text=app._tr("btn_queue_untranscribed"))
    app.refresh_recordings_button.configure(text=app._tr("btn_refresh_list"))
    app.open_folder_button.configure(text=app._tr("btn_open_folder"))
    app.delete_selected_button.configure(text=app._tr("btn_delete_selected"))
    app.recordings_tree.heading("session", text=app._tr("col_session"))
    app.recordings_tree.heading("short_name", text=app._tr("col_short_name"))
    app.recordings_tree.heading("duration", text=app._tr("col_duration"))
    app.recordings_tree.heading("audio", text=app._tr("col_audio"))
    app.recordings_tree.heading("txt", text=app._tr("col_txt"))
    app.recordings_tree.heading("queue", text=app._tr("col_queue"))
    app.log_box.configure(text=app._tr("group_log"))
    app._set_record_button_style(is_recording=app.recorder is not None)
    app._update_status_line()
    app._on_recording_selected()
    if refresh_data:
        app._refresh_microphones()
        app._refresh_recordings()


def set_record_button_style(app, is_recording: bool) -> None:
    if is_recording:
        app.record_button.configure(
            text=app._tr("record_stop"),
            bg="#b23b3b",
            activebackground="#8c2f2f",
        )
        return
    app.record_button.configure(
        text=app._tr("record_start"),
        bg="#1f8b4c",
        activebackground="#176a38",
    )


def set_recording_ui_state(app, is_recording: bool) -> None:
    app._set_record_button_style(is_recording)
    app.record_button.configure(state=tk.NORMAL)
    app.mic_combo.configure(state=tk.DISABLED if is_recording else "readonly")
    app.ui_language_combo.configure(state=tk.DISABLED if is_recording else "readonly")
    app.refresh_mic_button.configure(state=tk.DISABLED if is_recording else tk.NORMAL)


def set_transcription_ui_state(app, is_running: bool) -> None:
    sessions = app._selected_sessions()
    can_queue_any = any(app._session_audio_ready(session) for session in sessions)
    app.transcribe_selected_button.configure(state=tk.NORMAL if can_queue_any else tk.DISABLED)
    app.queue_untranscribed_button.configure(state=tk.NORMAL)
    app.self_label_entry.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.other_label_entry.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.auto_transcribe_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.library_combo.configure(state=tk.DISABLED if is_running else "readonly")
    app.device_combo.configure(state=tk.DISABLED if is_running else "readonly")
    app.model_combo.configure(state=tk.DISABLED if is_running else "readonly")
    app.language_combo.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.beam_spinbox.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.vad_filter_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.compute_type_combo.configure(state=tk.DISABLED if is_running else "readonly")
    app.include_timestamps_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.transcribe_mix_track_check.configure(state=tk.DISABLED if is_running else tk.NORMAL)
    app.cancel_transcribe_button.configure(state=tk.NORMAL if is_running else tk.DISABLED)
    selected_sessions = app._selected_sessions()
    app.refresh_recordings_button.configure(state=tk.NORMAL)
    app.open_folder_button.configure(state=tk.NORMAL if len(selected_sessions) == 1 else tk.DISABLED)
    app.delete_selected_button.configure(
        state=tk.DISABLED if is_running or app.transcription_queue or not selected_sessions else tk.NORMAL
    )
    if not is_running:
        app._sync_transcription_settings_ui()


def update_status_line(app) -> None:
    app.status_label.configure(text=f"{app._tr('status_prefix')}: {app.status_text}")
