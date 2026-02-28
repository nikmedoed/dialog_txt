from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import tkinter as tk


def set_windows_app_user_model_id(app_id: str = "DialogTxt.App") -> None:
    if not sys.platform.startswith("win"):
        return
    # For source/pythonw launches, an explicit AppUserModelID can make the taskbar
    # prefer the shortcut's cached shell icon instead of the live WM_SETICON handles.
    # Keep the explicit AppID for frozen builds only, where the executable owns the icon.
    if not getattr(sys, "frozen", False):
        return
    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


class WindowMixin:
    def _apply_window_icon(self) -> None:
        icon_ico_path, icon_png_path = self._resolve_icon_paths()

        if sys.platform.startswith("win") and icon_ico_path is not None:
            try:
                self.iconbitmap(default=str(icon_ico_path))
            except tk.TclError:
                pass
            # Apply explicit small/big icons to avoid Windows sticking to 16px resource.
            self.after(0, lambda p=icon_ico_path: self._apply_win32_icon_handles(p))
            # Some Tk builds create/re-parent the native window after idle; re-apply once.
            self.after(250, lambda p=icon_ico_path: self._apply_win32_icon_handles(p))
            # A late pass helps when style/theme code recreates native handles.
            self.after(1200, lambda p=icon_ico_path: self._apply_win32_icon_handles(p))
            return

        # Keep PhotoImage reference to avoid garbage collection.
        self._icon_image = None
        photo_candidates = [candidate for candidate in (icon_png_path, icon_ico_path) if candidate]
        for candidate in photo_candidates:
            try:
                self._icon_image = tk.PhotoImage(file=str(candidate))
                self.iconphoto(True, self._icon_image)
                break
            except tk.TclError:
                self._icon_image = None

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_win32_icon_handles(self, icon_ico_path: Path) -> None:
        if not sys.platform.startswith("win"):
            return
        try:
            user32 = ctypes.windll.user32
            user32.LoadImageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            user32.SendMessageW.restype = ctypes.c_void_p
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int
            user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            user32.GetAncestor.restype = ctypes.c_void_p

            set_class_icon = getattr(user32, "SetClassLongPtrW", None)
            if set_class_icon is None:
                set_class_icon = getattr(user32, "SetClassLongW", None)
            if set_class_icon is not None:
                set_class_icon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
                set_class_icon.restype = ctypes.c_void_p

            hwnd = ctypes.c_void_p(self.winfo_id())
            if not hwnd.value:
                return
            ga_root = 2
            root_hwnd_value = user32.GetAncestor(hwnd, ga_root)
            root_hwnd = ctypes.c_void_p(root_hwnd_value) if root_hwnd_value else hwnd

            image_icon = 1
            lr_loadfromfile = 0x00000010
            wm_seticon = 0x0080
            icon_small = 0
            icon_big = 1
            icon_small2 = 2
            gclp_hicon = -14
            gclp_hiconsm = -34
            sm_cxicon = 11
            sm_cyicon = 12
            sm_cxsmicon = 49
            sm_cysmicon = 50

            # Ask for a large source icon and let Windows downscale from a richer raster.
            big_w = max(256, int(user32.GetSystemMetrics(sm_cxicon) or 32))
            big_h = max(256, int(user32.GetSystemMetrics(sm_cyicon) or 32))
            # Ask for at least a 32px source for the "small" icon as well. Windows can
            # scale that down for the caption area, but it avoids locking onto the
            # softest 16px raster when rendering taskbar-adjacent surfaces.
            small_w = max(32, int(user32.GetSystemMetrics(sm_cxsmicon) or 16))
            small_h = max(32, int(user32.GetSystemMetrics(sm_cysmicon) or 16))

            hicon_big = user32.LoadImageW(
                None,
                str(icon_ico_path),
                image_icon,
                big_w,
                big_h,
                lr_loadfromfile,
            )
            hicon_small = user32.LoadImageW(
                None,
                str(icon_ico_path),
                image_icon,
                small_w,
                small_h,
                lr_loadfromfile,
            )

            new_handles: list[int] = []
            if hicon_big:
                user32.SendMessageW(root_hwnd, wm_seticon, ctypes.c_void_p(icon_big), hicon_big)
                if hwnd.value != root_hwnd.value:
                    user32.SendMessageW(hwnd, wm_seticon, ctypes.c_void_p(icon_big), hicon_big)
                if set_class_icon is not None:
                    set_class_icon(root_hwnd, gclp_hicon, hicon_big)
                    if hwnd.value != root_hwnd.value:
                        set_class_icon(hwnd, gclp_hicon, hicon_big)
                new_handles.append(int(hicon_big))
            if hicon_small:
                user32.SendMessageW(root_hwnd, wm_seticon, ctypes.c_void_p(icon_small), hicon_small)
                user32.SendMessageW(root_hwnd, wm_seticon, ctypes.c_void_p(icon_small2), hicon_small)
                if hwnd.value != root_hwnd.value:
                    user32.SendMessageW(hwnd, wm_seticon, ctypes.c_void_p(icon_small), hicon_small)
                    user32.SendMessageW(hwnd, wm_seticon, ctypes.c_void_p(icon_small2), hicon_small)
                if set_class_icon is not None:
                    set_class_icon(root_hwnd, gclp_hiconsm, hicon_small)
                    if hwnd.value != root_hwnd.value:
                        set_class_icon(hwnd, gclp_hiconsm, hicon_small)
                new_handles.append(int(hicon_small))

            if not new_handles:
                return

            old_handles = self._win32_icon_handles
            self._win32_icon_handles = new_handles
            self._destroy_win32_icon_handles(old_handles)
        except Exception:
            return

    def _release_win32_icon_handles(self) -> None:
        if not self._win32_icon_handles:
            return
        old_handles = self._win32_icon_handles
        self._win32_icon_handles = []
        self._destroy_win32_icon_handles(old_handles)

    @staticmethod
    def _destroy_win32_icon_handles(handles: list[int]) -> None:
        if not handles or not sys.platform.startswith("win"):
            return
        try:
            user32 = ctypes.windll.user32
            user32.DestroyIcon.argtypes = [ctypes.c_void_p]
            user32.DestroyIcon.restype = ctypes.c_bool
            for handle in handles:
                if handle:
                    user32.DestroyIcon(ctypes.c_void_p(handle))
        except Exception:
            pass

    def _resolve_icon_paths(self) -> tuple[Path | None, Path | None]:
        ico_candidates: list[Path] = []
        png_candidates: list[Path] = []
        if getattr(sys, "frozen", False):
            base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
            ico_candidates.extend(
                [
                    base_dir / "icon.ico",
                    base_dir / "docs" / "icon.ico",
                ]
            )
            png_candidates.extend(
                [
                    base_dir / "icon.png",
                    base_dir / "docs" / "icon.png",
                ]
            )

        project_root = Path(__file__).resolve().parent.parent.parent
        ico_candidates.extend(
            [
                project_root / "docs" / "icon.ico",
                project_root / "icon.ico",
            ]
        )
        png_candidates.extend(
            [
                project_root / "docs" / "icon.png",
                project_root / "icon.png",
            ]
        )
        return self._first_existing(ico_candidates), self._first_existing(png_candidates)
