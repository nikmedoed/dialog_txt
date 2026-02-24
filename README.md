# Dialog to TXT

Project codename/package: `dialog-txt`

## English

`Dialog to TXT` is a Python desktop app (Tkinter) for dual-track dialogue recording and Whisper transcription with speaker labels.

### What it does

- Records separate tracks:
  - `mic.ogg` (your microphone)
  - `desktop.ogg` (system audio via loopback)
  - `mix.ogg` (ready-to-listen mixed track)
- Stores each session in `recordings/<timestamp>/`
- Transcribes both tracks with selectable backend: `faster-whisper` or `openai-whisper` (`whisper`)
- Merges lines into `transcript.txt` with speaker attribution and optional timestamps

Transcript example:

```txt
[00:01:23] [Interlocutor] Hi, can you hear me well?
```

### Requirements

- Python 3.10+
- Windows, macOS, or Linux
- For desktop/system-audio capture:
  - Windows: works via WASAPI loopback
  - Linux (PulseAudio/PipeWire): works via monitor/loopback sources
  - macOS: requires a virtual loopback input (for example BlackHole/Soundflower/Loopback)
- Optional NVIDIA GPU + CUDA (if unavailable, transcription falls back to CPU)

### Installation

With `uv` (recommended):

```bash
uv sync
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .
```

### Run

```bash
python main.py
```

### Usage

1. Select a microphone.
2. Click `Start recording`.
3. Click `Stop recording`.
4. If auto-transcription is enabled, it starts automatically.
5. Otherwise select a session and click `Transcribe selected recording`.

### Settings

File: `app_settings.json`

Main options:

- `last_microphone`
- `speaker_self`, `speaker_other`
- `auto_transcribe_after_record`
- `transcription_library` (`faster-whisper` by default, or `whisper`)
- `whisper_model`:
  - for `faster-whisper`: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`
  - for `whisper`: `tiny(.en)`, `base(.en)`, `small(.en)`, `medium(.en)`, `large(-v1/-v2/-v3)`, `turbo`
- `whisper_device` (`auto`, `cpu`, `gpu`)
- `whisper_language` (`ru`, `en`, `auto`, etc.)
- `whisper_beam_size` (1..10)
- `whisper_vad_filter` (`true/false`)
- `whisper_compute_type` (`float16`, `int8_float16`, `int8`, applies to `faster-whisper`)
- `include_timestamps` (`true/false`, default `false`)

### Data layout

Each session is stored in its own folder:

```txt
recordings/
  2026-02-23_13-31-38/
    mic.ogg
    desktop.ogg
    mix.ogg
    transcript.txt
    meta.json
```

`meta.json` stores timestamps, devices, duration, and speaker labels.

### Limitations

- Transcription on CPU is noticeably slower than on CUDA GPU.
- On macOS, system-audio capture requires an external virtual loopback device.
- On Linux, desktop capture depends on available monitor sources in PulseAudio/PipeWire.

### Troubleshooting

- No microphones listed: click `Refresh` and verify input devices in OS settings.
- Desktop capture error on macOS: install/select a virtual loopback input (BlackHole/Soundflower/Loopback).
- Desktop capture error on Linux: check that monitor sources are exposed by PulseAudio/PipeWire.
- Missing selected transcription library: the app will offer one-click install via `pip`.
- Transcription error: if CUDA is unavailable, use `compute_type=int8` for CPU mode.

---

## Русский

`Dialog to TXT` — desktop-приложение на Python (Tkinter) для записи диалога в 2 дорожки и получения текстовой расшифровки через Whisper с разметкой спикеров.

### Что делает

- Пишет отдельно:
  - `mic.ogg` (ваш микрофон)
  - `desktop.ogg` (звук системы через loopback)
  - `mix.ogg` (готовый микс для быстрого прослушивания)
- Сохраняет сессию в `recordings/<timestamp>/`
- Транскрибирует обе дорожки с выбором backend: `faster-whisper` или `openai-whisper` (`whisper`)
- Объединяет реплики в `transcript.txt` с подписью спикера и опциональными таймкодами

Пример строки в транскрипте:

```txt
[00:01:23] [Собеседник] Привет, как слышно?
```

### Требования

- Python 3.10+
- Windows, macOS или Linux
- Для записи desktop/system audio:
  - Windows: через WASAPI loopback
  - Linux (PulseAudio/PipeWire): через monitor/loopback-источники
  - macOS: требуется virtual loopback-устройство (например BlackHole/Soundflower/Loopback)
- NVIDIA GPU + CUDA не обязателен: без CUDA транскрибация автоматически перейдёт на CPU

### Установка

Через `uv` (рекомендуется):

```bash
uv sync
```

Или через `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .
```

### Запуск

```bash
python main.py
```

### Как пользоваться

1. Выберите микрофон.
2. Нажмите `Начать запись`.
3. Нажмите `Остановить запись`.
4. Если включена авто-транскрибация, она стартует автоматически.
5. Иначе выберите сессию и нажмите `Транскрибировать выбранную запись`.

### Настройки

Файл: `app_settings.json`

Доступные параметры:

- `last_microphone`
- `speaker_self`, `speaker_other`
- `auto_transcribe_after_record`
- `transcription_library` (`faster-whisper` по умолчанию, либо `whisper`)
- `whisper_model`:
  - для `faster-whisper`: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`
  - для `whisper`: `tiny(.en)`, `base(.en)`, `small(.en)`, `medium(.en)`, `large(-v1/-v2/-v3)`, `turbo`
- `whisper_device` (`auto`, `cpu`, `gpu`)
- `whisper_language` (`ru`, `en`, `auto` и др.)
- `whisper_beam_size` (1..10)
- `whisper_vad_filter` (`true/false`)
- `whisper_compute_type` (`float16`, `int8_float16`, `int8`, используется для `faster-whisper`)
- `include_timestamps` (`true/false`, по умолчанию `false`)

### Структура данных

Каждая запись хранится в отдельной папке:

```txt
recordings/
  2026-02-23_13-31-38/
    mic.ogg
    desktop.ogg
    mix.ogg
    transcript.txt
    meta.json
```

`meta.json` содержит дату/время, устройства, длительность и подписи спикеров.

### Ограничения

- На CPU транскрибация работает заметно медленнее, чем на CUDA GPU.
- На macOS запись системного звука требует внешнее virtual loopback-устройство.
- На Linux desktop-захват зависит от monitor-источников PulseAudio/PipeWire.

### Диагностика

- Нет микрофона в списке: нажмите `Обновить`, проверьте системные устройства ввода.
- Ошибка desktop-захвата на macOS: установите/выберите virtual loopback-вход (BlackHole/Soundflower/Loopback).
- Ошибка desktop-захвата на Linux: проверьте, что PulseAudio/PipeWire публикует monitor-источники.
- Выбранная библиотека транскрибации не установлена: приложение предложит доустановить её через `pip`.
- Ошибка транскрибации: если CUDA недоступна, используйте `compute_type=int8` для CPU-режима.
