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
- Transcribes both tracks with `faster-whisper`
- Merges lines into `transcript.txt` with speaker attribution and optional timestamps

Transcript example:

```txt
[00:01:23] [Interlocutor] Hi, can you hear me well?
```

### Requirements

- Python 3.10+
- Windows (uses `os.startfile`, loopback capture via `soundcard`)
- NVIDIA GPU + CUDA (transcriber currently runs with `device="cuda"`)

### Installation

With `uv` (recommended):

```powershell
uv sync
```

Or with `pip`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Run

```powershell
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
- `whisper_model` (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`)
- `whisper_language` (`ru`, `en`, `auto`, etc.)
- `whisper_beam_size` (1..10)
- `whisper_vad_filter` (`true/false`)
- `whisper_compute_type` (`float16`, `int8_float16`, `int8`)
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

- Current transcription pipeline expects a CUDA-capable GPU.
- System-audio recording requires a working loopback source for the default output device.

### Troubleshooting

- No microphones listed: click `Refresh` and verify input devices in OS settings.
- Desktop capture error: check default output device and loopback support.
- Transcription error: verify CUDA/driver compatibility and selected `compute_type`.

---

## Русский

`Dialog to TXT` — desktop-приложение на Python (Tkinter) для записи диалога в 2 дорожки и получения текстовой расшифровки через Whisper с разметкой спикеров.

### Что делает

- Пишет отдельно:
  - `mic.ogg` (ваш микрофон)
  - `desktop.ogg` (звук системы через loopback)
  - `mix.ogg` (готовый микс для быстрого прослушивания)
- Сохраняет сессию в `recordings/<timestamp>/`
- Транскрибирует обе дорожки (`faster-whisper`)
- Объединяет реплики в `transcript.txt` с подписью спикера и опциональными таймкодами

Пример строки в транскрипте:

```txt
[00:01:23] [Собеседник] Привет, как слышно?
```

### Требования

- Python 3.10+
- Windows (используется `os.startfile`, loopback через `soundcard`)
- NVIDIA GPU + CUDA (в коде транскрибации модель запускается с `device="cuda"`)

### Установка

Через `uv` (рекомендуется):

```powershell
uv sync
```

Или через `pip`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Запуск

```powershell
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
- `whisper_model` (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`)
- `whisper_language` (`ru`, `en`, `auto` и др.)
- `whisper_beam_size` (1..10)
- `whisper_vad_filter` (`true/false`)
- `whisper_compute_type` (`float16`, `int8_float16`, `int8`)
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

- Текущая реализация транскрибатора ожидает CUDA GPU.
- Для корректной записи системного звука нужен доступный loopback-источник устройства вывода.

### Диагностика

- Нет микрофона в списке: нажмите `Обновить`, проверьте системные устройства ввода.
- Ошибка desktop-захвата: проверьте устройство вывода по умолчанию и поддержку loopback.
- Ошибка транскрибации: проверьте совместимость CUDA/драйвера и выбранный `compute_type`.
