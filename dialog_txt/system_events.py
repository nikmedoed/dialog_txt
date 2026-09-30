"""Windows lock/suspend notifications, independent of call applications."""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

WM_WTSSESSION_CHANGE = 0x02B1
WM_POWERBROADCAST = 0x0218
WTS_SESSION_LOCK = 7
PBT_APMSUSPEND = 4


def stop_reason(message: int, event: int) -> str | None:
    if message == WM_WTSSESSION_CHANGE and event == WTS_SESSION_LOCK:
        return "lock"
    if message == WM_POWERBROADCAST and event == PBT_APMSUSPEND:
        return "sleep"
    return None


class WindowsSessionEvents:
    def __init__(self, widget, callback):
        self.hwnd = None
        self.callback = callback
        self._procedure = None
        self.power_registration = None
        if sys.platform != "win32":
            return
        widget.update_idletasks()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32 = user32
        user32.RegisterSuspendResumeNotification.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        user32.RegisterSuspendResumeNotification.restype = wintypes.HANDLE
        user32.UnregisterSuspendResumeNotification.argtypes = [wintypes.HANDLE]
        user32.UnregisterSuspendResumeNotification.restype = wintypes.BOOL
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        hwnd = user32.GetAncestor(widget.winfo_id(), 2)  # GA_ROOT: Tk's wrapper HWND
        self.comctl = ctypes.WinDLL("comctl32", use_last_error=True)
        self.wts = ctypes.WinDLL("wtsapi32", use_last_error=True)
        result_type = ctypes.c_ssize_t
        subclass_type = ctypes.WINFUNCTYPE(result_type, wintypes.HWND, wintypes.UINT,
                                          wintypes.WPARAM, wintypes.LPARAM,
                                          ctypes.c_size_t, ctypes.c_size_t)
        self.comctl.SetWindowSubclass.argtypes = [wintypes.HWND, subclass_type,
                                                  ctypes.c_size_t, ctypes.c_size_t]
        self.comctl.SetWindowSubclass.restype = wintypes.BOOL
        self.comctl.RemoveWindowSubclass.argtypes = [wintypes.HWND, subclass_type, ctypes.c_size_t]
        self.comctl.RemoveWindowSubclass.restype = wintypes.BOOL
        self.comctl.DefSubclassProc.argtypes = [wintypes.HWND, wintypes.UINT,
                                               wintypes.WPARAM, wintypes.LPARAM]
        self.comctl.DefSubclassProc.restype = result_type
        self.wts.WTSRegisterSessionNotification.argtypes = [wintypes.HWND, wintypes.DWORD]
        self.wts.WTSRegisterSessionNotification.restype = wintypes.BOOL
        self.wts.WTSUnRegisterSessionNotification.argtypes = [wintypes.HWND]
        self.wts.WTSUnRegisterSessionNotification.restype = wintypes.BOOL

        def procedure(window, message, wparam, lparam, subclass_id, data):
            reason = stop_reason(message, wparam)
            if reason:
                # Only signal threads and enqueue here; Tk work runs through the
                # normal event poll. Capture stops even if Tk cannot run until resume.
                try:
                    self.callback(reason)
                except Exception:
                    import traceback
                    traceback.print_exc()
            return self.comctl.DefSubclassProc(window, message, wparam, lparam)

        self._procedure = subclass_type(procedure)
        if not self.comctl.SetWindowSubclass(hwnd, self._procedure, 1, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = hwnd
        if not self.wts.WTSRegisterSessionNotification(hwnd, 0):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error
        self.power_registration = user32.RegisterSuspendResumeNotification(hwnd, 0)
        if not self.power_registration:
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.power_registration:
            self.user32.UnregisterSuspendResumeNotification(self.power_registration)
            self.power_registration = None
        if self.hwnd:
            self.wts.WTSUnRegisterSessionNotification(self.hwnd)
            self.comctl.RemoveWindowSubclass(self.hwnd, self._procedure, 1)
            self.hwnd = None
