from __future__ import annotations

from ..localization import UI_LANGUAGE_CODES, resolve_ui_language, tr
from ..ui_layout import apply_localization, build_ui


class LocalizationMixin:
    def _build_ui(self) -> None:
        build_ui(self)

    @staticmethod
    def _resolve_ui_language(value: str | None) -> str:
        return resolve_ui_language(value)

    def _tr(self, key: str, **kwargs) -> str:
        return tr(self.ui_language, key, **kwargs)

    def _on_ui_language_selected(self, _event=None) -> None:
        selected_code = self.ui_language_code_var.get().strip().upper()
        target_language = UI_LANGUAGE_CODES.get(selected_code, resolve_ui_language(None))
        if target_language == self.ui_language:
            return
        self.ui_language = target_language
        self._apply_localization(refresh_data=True)
        self._save_app_settings()

    def _apply_localization(self, refresh_data: bool = False) -> None:
        apply_localization(self, refresh_data=refresh_data)
