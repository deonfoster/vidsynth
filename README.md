# Vidsynth

## How to run (macOS)

This app is a PyQt6 GUI. The main entrypoint is `vidz.py`.

1. Install system dependency:

   ```bash
   brew install ffmpeg
   ```

2. Create a fresh venv with a Python that still includes `audioop` (3.9–3.12 recommended):

   ```bash
   /usr/bin/python3 -m venv .venv
   ```

3. Install dependencies:

   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   .venv/bin/python vidz.py
   ```

Notes:
- `pydub` uses `ffmpeg` for decoding audio files (mp3, etc). Ensure `ffmpeg` is on your `PATH`.
