"""Windows integration smoke test. Uses synthetic audio and a temporary app home."""
import ctypes
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    with tempfile.TemporaryDirectory() as home:
        os.environ["DIALOG_TXT_HOME"] = home
        import numpy as np
        import soundfile as sf
        from dialog_txt.ui import App
        from dialog_txt.storage import write_session_metadata

        with patch.object(App, "_refresh_microphones"), patch.object(App, "_start_idle_level_monitor"), \
                patch.object(App, "_stop_idle_level_monitor"), patch.object(App, "_save_app_settings"):
            app = App()
            app.withdraw()
            try:
                assert app.system_events.hwnd, "Windows notification registration failed"
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
                user32.SendMessageW.restype = ctypes.c_ssize_t
                for message, code, reason in ((0x02B1, 7, "lock"), (0x0218, 4, "sleep")):
                    fake = SimpleNamespace(stop_event=threading.Event(), stop=lambda: None, raise_if_failed=lambda: None)
                    app.recorder = fake
                    app.recording_started_at = time.time() - 10
                    app.active_session_dir = None
                    user32.SendMessageW(app.system_events.hwnd, message, code, 0)
                    assert fake.stop_event.is_set(), "Capture must stop in native callback"
                    app._poll_events()
                    assert app.recorder is None
                    assert app.status_text == app._tr("system_stop_" + reason)
                user32.SendMessageW(app.system_events.hwnd, 0x02B1, 8, 0)
                app._poll_events()
                assert app.recorder is None, "Unlock must not resume recording"

                session = Path(home) / "recordings" / "2026-09-30_23-59-50"
                session.mkdir()
                rate = 8000
                samples = np.sin(np.arange(30 * rate) * .1).astype("float32") * .3
                for name in ("mic.ogg", "desktop.ogg", "mix.ogg"):
                    sf.write(str(session / name), samples, rate, format="OGG", subtype="VORBIS")
                write_session_metadata(session, {"created_at": "2026-09-30T23:59:50", "duration_seconds": 30})
                app._refresh_recordings()
                app.recordings_tree.selection_set(str(session))
                app._trim_selected_recording()
                editor = app.trim_window
                assert editor is not None
                editor.transient("")
                editor.deiconify()
                app.update_idletasks()
                editor.geometry("860x540")
                app.update()
                for button in editor.buttons:
                    assert not int(button["takefocus"])
                    assert button.winfo_ismapped(), button["text"]
                    assert button.winfo_rooty() + button.winfo_height() <= editor.winfo_rooty() + editor.winfo_height()
                    assert button.winfo_rootx() + button.winfo_width() <= editor.winfo_rootx() + editor.winfo_width()
                assert editor.winfo_reqheight() <= 540, editor.winfo_reqheight()
                for language in ("ru", "en"):
                    app.ui_language = language
                    app._apply_localization()
                    app.update_idletasks()
                    row = app.trim_selected_button.master
                    assert row.winfo_reqwidth() < 530, (language, row.winfo_reqwidth())
                    assert app.trim_selected_button.grid_info()["row"] == 0
                assert editor.cursor == editor.cutoff == 30
                actual_player = editor.player
                class PlayerStub:
                    thread = None
                    position = 0
                    def play(self, position, end):
                        self.position = position
                        self.thread = SimpleNamespace(is_alive=lambda: True)
                    def stop(self):
                        self.thread = None
                editor.player = PlayerStub()
                editor._nudge(-5)
                assert editor.cursor == 25 and editor.cutoff == 30
                editor.overview_canvas.event_generate("<MouseWheel>", delta=120)
                assert editor.cursor == 24 and editor.cutoff == 30
                editor.detail_canvas.event_generate("<MouseWheel>", delta=-120, state=1)
                assert editor.cursor == 29 and editor.cutoff == 30
                editor.detail_canvas.event_generate("<MouseWheel>", delta=120, state=4)
                assert editor.cursor == 0
                editor._move_cursor(25)
                editor.detail_canvas.focus_force()
                editor.detail_canvas.event_generate("<Left>")
                assert editor.cursor == 24
                editor.detail_canvas.event_generate("<Shift-Left>")
                assert editor.cursor == 19
                editor.detail_canvas.event_generate("<Control-Right>")
                assert editor.cursor == 30
                editor._nudge(-5)
                editor._cut_here()
                assert editor.cutoff == 25
                editor.duration_entry.focus_force()
                text = editor.duration_var.get()
                editor.duration_entry.event_generate("<space>")
                assert editor.player.thread is not None
                assert editor.duration_var.get() == text
                editor.duration_entry.event_generate("<space>")
                assert editor.player.thread is None
                editor.play_button.invoke()
                assert editor.player.thread is not None
                editor.play_button.focus_force()
                app.update()
                assert editor.focus_get() is editor.detail_canvas
                editor.detail_canvas.event_generate("<space>")
                assert editor.player.thread is None
                editor._move_cursor(12)
                editor._request_detail()
                editor._play()
                editor.player.position = 14.25
                editor.after_cancel(editor.poll_after)
                editor._poll()
                assert editor.cursor == 14.25
                assert float(editor.seek.get()) == 14.25
                marker = editor.detail_canvas.find_withtag("playhead")
                assert marker
                expected_x = (14.25 - editor.view_start) / (editor.view_end - editor.view_start) * editor.detail_canvas.winfo_width()
                assert abs(editor.detail_canvas.coords(marker[0])[0] - expected_x) < .01
                editor.player.position = 14.625
                editor._pause()
                assert editor.cursor == 14.625
                assert float(editor.seek.get()) == 14.625
                editor._nudge(-1)
                assert editor.cursor == 13.625
                editor._cut_here()
                assert editor.cutoff == 13.625
                editor.player = actual_player
                editor.duration_var.set("10")
                assert str(editor.apply_button["state"]) == "normal"
                editor._from_duration()
                # The requested height must fit the default window size; all
                # controls should remain reachable at normal font/DPI settings.
                assert editor.winfo_reqheight() <= 540, editor.winfo_reqheight()
                editor.clock_var.set("00:00:10.125")
                editor._from_clock()
                assert editor.cutoff == 20.125
                editor.duration_var.set("00:00:12.500")
                editor._from_duration()
                assert editor.cutoff == 12.5
                editor.duration_var.set("00:00:10.250")
                editor._apply()  # Freshly typed field must work without pressing Select.
                deadline = time.monotonic() + 10
                while app.trim_window is not None and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                assert app.trim_window is None, "Trim worker did not finish"
                assert sf.info(str(session / "mix.ogg")).frames == 82000
                app._trim_selected_recording()
                editor = app.trim_window
                editor._undo()
                deadline = time.monotonic() + 10
                while app.trim_window is not None and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                assert app.trim_window is None
                assert sf.info(str(session / "mix.ogg")).frames == 240000
                print("Windows native lock/suspend, no resume, editor input, background trim and undo: OK")
            finally:
                if app.trim_window:
                    app.trim_window.close(restart_monitor=False)
                app.system_events.close()
                app._release_win32_icon_handles()
                app.destroy()


if __name__ == "__main__":
    main()
