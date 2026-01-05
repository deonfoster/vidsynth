import sys
import os
import time
import math
import json
import uuid
import subprocess
import shutil
import numpy as np
import librosa
import mido 
from pydub import AudioSegment

# --- IMPORTS ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QVBoxLayout, QPushButton, QSlider,
                             QFileDialog, QHBoxLayout, QComboBox, QScrollArea,
                             QSpinBox, QRadioButton, QButtonGroup, QFrame,
                             QGraphicsView, QGraphicsScene, QDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox, QCheckBox, QSizePolicy) 
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtMultimedia import (QMediaPlayer, QAudioOutput, QMediaDevices)
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import (QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, 
                          QRectF, QPointF, QSizeF, QRect, QObject)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QPolygonF, QFont, QCursor, QAction, QKeySequence

# --- STYLING ---
DARK_THEME = """
QWidget { background-color: #121212; color: #ffffff; font-family: 'Arial', sans-serif; font-size: 13px; }
QMainWindow, QScrollArea { background-color: #121212; border: none; }
QLabel { color: #bbbbbb; font-weight: 600; background-color: transparent; padding: 0px; }
QPushButton { background-color: #2a2a2a; color: #ffffff; border: 1px solid #444; padding: 6px; border-radius: 3px; font-weight: bold; }
QPushButton:hover { background-color: #3d3d3d; border: 1px solid #666; }
QPushButton:pressed { background-color: #00CCFF; color: #000000; }
QPushButton:checked { background-color: #00CCFF; color: #000000; border: 1px solid #ffffff; }
QPushButton[active="true"] { border: 2px solid #ffffff; background-color: #444; }
QPushButton[sync="true"]:checked { background-color: #FF0055; color: white; border: 1px solid white; }
/* MUTE BUTTON */
QPushButton[mute="true"] { background-color: #222; color: #666; border: 1px solid #444; font-size: 11px; max-height: 20px; }
QPushButton[mute="true"]:checked { background-color: #FF0000; color: white; border: 1px solid #FF5555; }
QPushButton[loop="true"] { min-width: 30px; font-size: 11px; padding: 4px; } 
QRadioButton { color: #cccccc; font-weight: bold; }
QRadioButton::indicator { width: 12px; height: 12px; border-radius: 6px; border: 1px solid #555; background-color: #222; }
QRadioButton::indicator:checked { background-color: #00CCFF; border: 2px solid white; }
QCheckBox { color: #cccccc; font-weight: bold; spacing: 5px; }
QComboBox { background-color: #222; border: 1px solid #444; padding: 4px; }
/* SLIDERS */
QSlider::groove:vertical { border: 1px solid #333; width: 6px; background: #1a1a1a; margin: 0 10px; border-radius: 3px; }
QSlider::handle:vertical { background: #555; border: 1px solid #777; height: 14px; width: 20px; margin: 0 -8px; border-radius: 2px; }
QSlider::handle:vertical:hover { background: #00CCFF; border: 1px solid #fff; }
QSlider::sub-page:vertical { background: #444; border-radius: 3px; }
QSlider::groove:horizontal { border: 1px solid #333; height: 8px; background: #1a1a1a; margin: 2px 0; border-radius: 4px; }
QSlider::handle:horizontal { background: #00CCFF; border: 1px solid #00CCFF; width: 24px; height: 24px; margin: -9px 0; border-radius: 12px; }
QSlider::sub-page:horizontal { background: #444; border-radius: 4px; }
/* TABLE */
QTableWidget { background-color: #1a1a1a; gridline-color: #333; border: 1px solid #444; }
QHeaderView::section { background-color: #2a2a2a; padding: 4px; border: 1px solid #444; }
QTableWidget::item:selected { background-color: #00CCFF; color: black; }
"""

KEY_MAP = {'a': (0, 0, "#FF0055"), 's': (0, 1, "#00CCFF"), 'd': (1, 0, "#00FF66"), 'f': (1, 1, "#FFAA00")}

# ==========================================
# 1. WORKERS & HELPERS
# ==========================================

class MidiWorker(QThread):
    message_received = pyqtSignal(object) 
    def __init__(self):
        super().__init__()
        self.input_port = None; self.port_name = None; self.running = False
    def set_port(self, name):
        if self.input_port: self.input_port.close()
        self.port_name = name
    def run(self):
        self.running = True
        if not self.port_name: return
        try:
            with mido.open_input(self.port_name) as port:
                self.input_port = port
                while self.running:
                    for msg in port.iter_pending(): self.message_received.emit(msg)
                    time.sleep(0.001) 
        except: pass
    def stop(self): self.running = False; self.wait()

class AudioAnalysisWorker(QThread):
    finished = pyqtSignal(str, QPixmap, float, int, object, int, str)
    def __init__(self, key, filepath, width, height, color_hex, gen_id):
        super().__init__()
        self.key, self.filepath, self.width, self.height = key, filepath, width, height; self.bg_color, self.gen_id = QColor(color_hex), gen_id
    def run(self):
        try:
            if self.isInterruptionRequested(): return
            audio_full = AudioSegment.from_file(self.filepath)
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            clean_name = os.path.basename(self.filepath).replace(" ", "_")
            wav_path = os.path.join(temp_dir, f"{clean_name}_base.wav")
            if not os.path.exists(wav_path): audio_full.export(wav_path, format="wav")
            duration_ms = len(audio_full); audio_vis = audio_full[:60000] if duration_ms > 60000 else audio_full
            raw_samples = np.array(audio_full.get_array_of_samples())
            sample_rate = audio_full.frame_rate
            vis_samples = np.array(audio_vis.set_channels(1).set_frame_rate(11025).get_array_of_samples())
            tempo, _ = librosa.beat.beat_track(y=vis_samples.astype(np.float32)/32768.0, sr=11025)
            bpm = float(tempo.item()) if isinstance(tempo, np.ndarray) else float(round(tempo, 2))
            draw_samples = vis_samples[::150]
            pixmap = QPixmap(self.width, self.height); pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap); painter.setPen(QPen(self.bg_color.darker(150), 1)); center_y = self.height / 2; step = len(draw_samples) / self.width
            for x in range(self.width):
                if self.isInterruptionRequested(): return
                idx = int(x * step)
                if idx < len(draw_samples): h = abs(draw_samples[idx]) * (self.height * 0.9) / 32768.0; painter.drawLine(x, int(center_y - h/2), x, int(center_y + h/2))
            painter.end()
            if not self.isInterruptionRequested(): self.finished.emit(self.key, pixmap, bpm, duration_ms, raw_samples, sample_rate, wav_path)
        except:
            if not self.isInterruptionRequested(): self.finished.emit(self.key, QPixmap(), 120.0, 0, None, 44100, "")

class RubberBandWorker(QThread):
    finished = pyqtSignal(str, str, float)
    def __init__(self, key, original_wav, tempo_ratio):
        super().__init__(); self.key, self.original_wav, self.tempo_ratio = key, original_wav, tempo_ratio
    def run(self):
        try:
            if not os.path.exists(self.original_wav) or self.tempo_ratio <= 0: return
            unique_id = uuid.uuid4().hex[:8]; base, ext = os.path.splitext(self.original_wav); out_path = f"{base}_st_{self.tempo_ratio:.2f}_{unique_id}{ext}"
            if shutil.which("rubberband") is None: return
            subprocess.run(["rubberband", "-q", "realtime", "-t", str(1.0/self.tempo_ratio), self.original_wav, out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.finished.emit(self.key, out_path, self.tempo_ratio)
        except: pass

# ==========================================
# 2. CORE COMPONENTS (UPDATED)
# ==========================================

class VJDeck:
    def __init__(self, name, video_item):
        self.name = name; self.video_item = video_item
        self.current_filepath = None; self.base_wav_path = None
        self.is_looping = True 
        self.player = QMediaPlayer(); self.player.setVideoOutput(self.video_item); self.player.setLoops(QMediaPlayer.Loops.Infinite) 
        self.player.mediaStatusChanged.connect(self.on_media_status)
        self.video_audio = QAudioOutput(); self.player.setAudioOutput(self.video_audio); self.video_audio.setVolume(0) 
        self.audio_player = QMediaPlayer(); self.main_output = QAudioOutput(); self.audio_player.setAudioOutput(self.main_output); self.audio_player.setLoops(QMediaPlayer.Loops.Infinite)
        self.cue_player = QMediaPlayer(); self.cue_output = QAudioOutput(); self.cue_player.setAudioOutput(self.cue_output); self.cue_player.setLoops(QMediaPlayer.Loops.Infinite)
        self.cue_active = False; self.raw_samples = None; self.sample_rate = 44100; self.target_volume = 1.0; self.playback_rate = 1.0
        self.fade_level = 1.0; self.fade_timer = QTimer(); self.fade_timer.setInterval(10); self.fade_timer.timeout.connect(self._process_fade)

    def set_loop_mode(self, looping):
        self.is_looping = looping; loop_const = QMediaPlayer.Loops.Infinite if looping else QMediaPlayer.Loops.Once
        self.player.setLoops(loop_const); self.audio_player.setLoops(loop_const); self.cue_player.setLoops(loop_const)

    def on_media_status(self, status):
        if not self.is_looping and status == QMediaPlayer.MediaStatus.EndOfMedia: self.audio_player.stop(); self.cue_player.stop()

    def load_video(self, filepath): self.current_filepath = filepath; self.player.setSource(QUrl.fromLocalFile(filepath))
    def load_base_audio(self, wav_path): self.base_wav_path = wav_path; self.swap_audio(wav_path, reset_rate_to_video=True)
    def swap_audio(self, path, reset_rate_to_video=False):
        pos = self.player.position(); playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState; url = QUrl.fromLocalFile(path)
        self.audio_player.setSource(url); self.cue_player.setSource(url)
        loop_const = QMediaPlayer.Loops.Infinite if self.is_looping else QMediaPlayer.Loops.Once
        self.audio_player.setLoops(loop_const); self.cue_player.setLoops(loop_const)
        if reset_rate_to_video:
            self.audio_player.setPlaybackRate(self.playback_rate); self.cue_player.setPlaybackRate(self.playback_rate)
            self.audio_player.setPosition(pos); self.cue_player.setPosition(pos)
        else:
            self.audio_player.setPlaybackRate(1.0); self.cue_player.setPlaybackRate(1.0)
            mapped_pos = int(pos / self.playback_rate); self.audio_player.setPosition(mapped_pos); self.cue_player.setPosition(mapped_pos)
        if playing: self.audio_player.play(); self.cue_player.play() if self.cue_active else None
    def has_media(self): return self.player.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia
    def set_audio_data(self, samples, rate): self.raw_samples = samples; self.sample_rate = rate
    def find_zero_crossing(self, target_ms):
        if self.raw_samples is None: return target_ms
        idx = int((target_ms / 1000.0) * self.sample_rate); idx -= idx % 2 
        win = int(0.02 * self.sample_rate); s = max(0, idx - win); e = min(len(self.raw_samples), idx + win)
        if s >= e: return target_ms
        return int(((s + np.argmin(np.abs(self.raw_samples[s:e]))) / self.sample_rate) * 1000.0)
    
    def trigger(self, pos):
        self.main_output.setMuted(True); self.cue_output.setMuted(True) if self.cue_active else None
        safe_pos = self.find_zero_crossing(pos); self.player.setPosition(safe_pos)
        audio_pos = safe_pos
        if self.audio_player.playbackRate() == 1.0 and self.playback_rate != 1.0: audio_pos = int(safe_pos / self.playback_rate)
        self.audio_player.setPosition(audio_pos); self.cue_player.setPosition(audio_pos) if self.cue_active else None
        self.player.play(); self.audio_player.play(); self.cue_player.play() if self.cue_active else None
        self.fade_level = 0.0; self.main_output.setVolume(0); self.main_output.setMuted(False)
        if self.cue_active: self.cue_output.setVolume(0); self.cue_output.setMuted(False)
        if not self.fade_timer.isActive(): self.fade_timer.start()

    # --- NEW: UPDATE VIDEO OPACITY WITH VOLUME ---
    def _process_fade(self):
        self.fade_level += 0.1
        if self.fade_level >= 1.0: self.fade_level = 1.0; self.fade_timer.stop()
        
        current_vol = self.target_volume * self.fade_level
        self.main_output.setVolume(current_vol)
        self.video_item.setOpacity(current_vol) # TIE VIDEO TO AUDIO
        
        if self.cue_active: self.cue_output.setVolume(1.0 * self.fade_level)

    def set_volume(self, vol): 
        self.target_volume = vol
        if not self.fade_timer.isActive(): 
            self.main_output.setVolume(vol)
            self.video_item.setOpacity(vol) # TIE VIDEO TO AUDIO

    def set_cue_active(self, active):
        self.cue_active = active
        if active: self.cue_output.setVolume(1.0); self.cue_player.setPosition(self.audio_player.position()); self.cue_player.play() if self.audio_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState else None
        else: self.cue_output.setVolume(0)
    def play(self): 
        if not self.is_looping and self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia: self.seek(0)
        self.player.play(); self.audio_player.play(); self.cue_player.play() if self.cue_active else None
    def pause(self): self.player.pause(); self.audio_player.pause(); self.cue_player.pause()
    def seek(self, pos): 
        self.player.setPosition(pos)
        a_pos = int(pos / self.playback_rate) if (self.audio_player.playbackRate() == 1.0 and self.playback_rate != 1.0) else pos
        self.audio_player.setPosition(a_pos); self.cue_player.setPosition(a_pos) if self.cue_active else None
    def position(self): return self.player.position()
    def duration(self): return self.player.duration()
    def playbackState(self): return self.player.playbackState()
    def setPlaybackRate(self, rate): 
        self.playback_rate = rate; self.player.setPlaybackRate(rate)
        if self.base_wav_path and self.audio_player.playbackRate() == 1.0: self.swap_audio(self.base_wav_path, reset_rate_to_video=True)
        self.audio_player.setPlaybackRate(rate); self.cue_player.setPlaybackRate(rate)
    def set_main_output(self, device): self.main_output.setDevice(device)
    def set_cue_output(self, device): self.cue_output.setDevice(device)

class InteractiveWaveform(QLabel):
    def __init__(self, key_char, color, parent_app):
        super().__init__()
        self.key_char, self.parent_app = key_char, parent_app
        self.setAcceptDrops(True); self.setMouseTracking(True); self.base_color = QColor(color)
        self.setFixedSize(160, 100); self.setStyleSheet(f"border: 2px solid {color}; border-radius: 4px; background-color: #222;")
        self.filename = "[Empty]"; self.bpm_text = ""; self.waveform_pixmap = None
        self.playhead_x = 0; self.is_deck_a = False; self.is_deck_b = False
        self.loading = False; self.hotcues = {}; self.track_duration = 0
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.waveform_pixmap: painter.drawPixmap(0, 0, self.waveform_pixmap)
        if self.parent_app.active_edit_track == self.key_char:
            painter.setPen(QPen(QColor("#FFFFFF"), 3)); painter.drawRect(self.rect().adjusted(2,2,-2,-2))
        else:
            painter.setPen(QPen(self.base_color, 2)); painter.drawRect(self.rect().adjusted(1,1,-1,-1))
        painter.setPen(QColor("white")); font = painter.font(); font.setBold(True); font.setPointSize(10); painter.setFont(font)
        label = f"TRACK {self.key_char.upper()}\n{self.filename}{status}\n{self.bpm_text}" if (status := " (...)" if self.loading else "") else f"TRACK {self.key_char.upper()}\n{self.filename}\n{self.bpm_text}"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, label); painter.end()
    def mousePressEvent(self, event): self.parent_app.select_track_for_edit(self.key_char)
    def dragEnterEvent(self, event): event.accept() if event.mimeData().hasUrls() else event.ignore()
    def dropEvent(self, event): self.parent_app.load_track(self.key_char, [u.toLocalFile() for u in event.mimeData().urls()][0])
    def set_data(self, pixmap, bpm, duration): self.waveform_pixmap = pixmap; self.bpm_text = f"{bpm} BPM"; self.track_duration = duration; self.loading = False; self.update()
    def update_playhead(self, ratio): self.playhead_x = int(ratio * self.width()); self.update()
    def set_loading(self): self.loading = True; self.update()

# ==========================================
# 3. WIDGETS
# ==========================================

class MidiConfigDialog(QDialog):
    def __init__(self, midi_worker, current_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MIDI Config"); self.resize(500, 400)
        self.worker = midi_worker; self.midi_map = current_map; self.learning_row = -1
        self.worker.message_received.connect(self.on_midi_message)
        layout = QVBoxLayout(self)
        self.c_dev = QComboBox(); inputs = mido.get_input_names(); self.c_dev.addItems(inputs)
        if self.worker.port_name in inputs: self.c_dev.setCurrentText(self.worker.port_name)
        self.c_dev.currentTextChanged.connect(self.change_device); layout.addWidget(self.c_dev)
        self.table = QTableWidget(); self.table.setColumnCount(4); self.table.setHorizontalHeaderLabels(["Action", "Type", "Val", "Learn"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch); self.table.verticalHeader().setVisible(False); layout.addWidget(self.table)
        self.populate_table()
        btn_close = QPushButton("Done"); btn_close.clicked.connect(self.accept); layout.addWidget(btn_close)
    def change_device(self, name): self.worker.stop(); self.worker.set_port(name); self.worker.start()
    def populate_table(self):
        self.table.setRowCount(0)
        for action, binding in sorted(self.midi_map.items()):
            row = self.table.rowCount(); self.table.insertRow(row); self.table.setItem(row, 0, QTableWidgetItem(action))
            if binding: msg_type = "CC" if binding['type'] == 'control_change' else "NOTE"; val = str(binding['val'])
            else: msg_type = "-"; val = "-"
            self.table.setItem(row, 1, QTableWidgetItem(msg_type)); self.table.setItem(row, 2, QTableWidgetItem(val))
            btn = QPushButton("LEARN"); btn.clicked.connect(lambda _, r=row: self.start_learn(r)); self.table.setCellWidget(row, 3, btn)
    def start_learn(self, row):
        self.learning_row = row; self.table.cellWidget(row, 3).setText("WAIT...")
    def on_midi_message(self, msg):
        if self.learning_row == -1 or msg.type not in ['note_on', 'control_change']: return
        action_name = self.table.item(self.learning_row, 0).text()
        binding = {'type': msg.type, 'val': msg.control if msg.type == 'control_change' else msg.note, 'channel': msg.channel}
        self.midi_map[action_name] = binding; self.learning_row = -1; self.populate_table()

class HotkeyEditor(QDialog):
    def __init__(self, current_bindings, parent=None):
        super().__init__(parent); self.setWindowTitle("Keys"); self.resize(300, 400)
        self.bindings = current_bindings.copy(); self.waiting_for_key = None
        layout = QVBoxLayout(self); self.table = QTableWidget(); self.table.setColumnCount(2); self.table.setHorizontalHeaderLabels(["Action", "Key"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self.start_rebinding); layout.addWidget(self.table); self.populate_table()
        btn_box = QHBoxLayout(); btn_save = QPushButton("Save"); btn_save.clicked.connect(self.accept); btn_box.addWidget(btn_save); layout.addLayout(btn_box)
    def populate_table(self):
        self.table.setRowCount(len(self.bindings))
        for i, (action, key_code) in enumerate(self.bindings.items()):
            self.table.setItem(i, 0, QTableWidgetItem(action)); key_name = QKeySequence(key_code).toString(); 
            self.table.setItem(i, 1, QTableWidgetItem(key_name))
    def start_rebinding(self, row, col): self.waiting_for_key = self.table.item(row, 0).text(); self.table.setEnabled(False); self.setFocus()
    def keyPressEvent(self, event):
        if self.waiting_for_key: self.bindings[self.waiting_for_key] = event.key(); self.waiting_for_key = None; self.table.setEnabled(True); self.populate_table()
        else: super().keyPressEvent(event)
    def get_bindings(self): return self.bindings

class LoopBar(QWidget):
    def __init__(self, parent_sequencer):
        super().__init__(); self.sequencer = parent_sequencer; self.setFixedHeight(25); self.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        self.setMouseTracking(True); self.dragging = False; self.drag_start_x = 0; self.start_step_cache = 0
    def paintEvent(self, event):
        painter = QPainter(self); w = self.width(); h = self.height(); step_w = w / 64.0; painter.fillRect(self.rect(), QColor("#111"))
        start_x = self.sequencer.loop_start * step_w; loop_w = self.sequencer.loop_length * step_w
        bar_rect = QRectF(start_x, 2, loop_w, h - 4); painter.setBrush(QColor("#00CCFF")); painter.setPen(Qt.PenStyle.NoPen); painter.drawRoundedRect(bar_rect, 2, 2)
        painter.setPen(QColor("black")); painter.setFont(QFont("Arial", 9, QFont.Weight.Bold)); label = f"{self.sequencer.loop_length}"; painter.drawText(bar_rect, Qt.AlignmentFlag.AlignCenter, label)
    def mousePressEvent(self, event):
        step_w = self.width() / 64.0; start_x = self.sequencer.loop_start * step_w; loop_w = self.sequencer.loop_length * step_w; bar_rect = QRectF(start_x, 0, loop_w, self.height())
        if bar_rect.contains(event.position()): self.dragging = True; self.drag_start_x = event.position().x(); self.start_step_cache = self.sequencer.loop_start; self.setCursor(Qt.CursorShape.SizeHorCursor)
        else: new_start = int(event.position().x() / step_w) - (self.sequencer.loop_length // 2); self.sequencer.set_loop_window(new_start, self.sequencer.loop_length); self.update()
    def mouseMoveEvent(self, event):
        if self.dragging: delta_steps = int((event.position().x() - self.drag_start_x) / (self.width() / 64.0)); self.sequencer.set_loop_window(self.start_step_cache + delta_steps, self.sequencer.loop_length); self.update()
    def mouseReleaseEvent(self, event): self.dragging = False; self.setCursor(Qt.CursorShape.ArrowCursor)

class PianoRollSequencer(QWidget):
    def __init__(self, parent_app):
        super().__init__(); self.parent_app = parent_app; self.setMinimumHeight(200); self.setStyleSheet("background-color: #080808; border: 1px solid #333;")
        self.points = {}; self.selection = set(); self.current_step = 0; self.steps = 64; self.loop_start = 0; self.loop_length = 64
        self.mode = "IDLE"; self.drag_start_pos = QPointF(); self.last_mouse_pos = QPointF(); self.marquee_rect = QRectF(); self.move_snapshot = {}; self.clean_slate_points = {}
        self.undo_stack = []; self.redo_stack = []; self.state_at_press = {}; self.setMouseTracking(True); self.setFocusPolicy(Qt.FocusPolicy.ClickFocus) 
    def quantize_selection(self, grid=4):
        if not self.selection: return
        self.push_to_undo(self.points.copy()); new_points = self.points.copy(); new_selection = set(); moves = []
        for step in list(self.selection):
            if step in new_points: target = max(0, min(round(step / grid) * grid, 63)); moves.append((step, target, new_points[step]))
        for o, n, v in moves: 
            if o in new_points: del new_points[o]
        for o, n, v in moves: new_points[n] = v; new_selection.add(n)
        self.points = new_points; self.selection = new_selection; self.update(); self.parent_app.save_curve_data()
    def push_to_undo(self, state):
        self.undo_stack.append(state); 
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)
        self.redo_stack.clear()
    def perform_undo(self):
        if not self.undo_stack: return
        self.redo_stack.append(self.points.copy()); self.points = self.undo_stack.pop(); self.selection.clear(); self.update(); self.parent_app.save_curve_data()
    def perform_redo(self):
        if not self.redo_stack: return
        self.undo_stack.append(self.points.copy()); self.points = self.redo_stack.pop(); self.selection.clear(); self.update(); self.parent_app.save_curve_data()
    def set_loop_window(self, start, length):
        self.loop_length = length; self.loop_start = max(0, min(start, 64 - length)); self.update()
        if hasattr(self.parent_app, 'loop_bar'): self.parent_app.loop_bar.update()
    def set_data(self, data): self.points = data.copy() if data else {}; self.selection.clear(); self.undo_stack.clear(); self.redo_stack.clear(); self.update()
    def get_data(self): return self.points
    def get_step_from_x(self, x): return max(0, min(int(x / (self.width()/self.steps)), self.steps - 1))
    def get_val_from_y(self, y): return max(0.0, min(1.0 - (y / self.height()), 1.0))
    def get_rect_for_note(self, step, val):
        step_w = self.width() / self.steps; block_h = 16; y = int(self.height() - (val * self.height())) - 8; y = max(0, min(y, self.height() - 16))
        return QRectF(int(step * step_w), y, step_w, block_h)
    
    def keyPressEvent(self, event):
        k = event.key(); keys = self.parent_app.key_bindings
        if k == Qt.Key.Key_Up or k == Qt.Key.Key_Down:
            if not self.selection: event.ignore(); super().keyPressEvent(event); return 
            self.push_to_undo(self.points.copy()); increment = 0.01 if k == Qt.Key.Key_Up else -0.01
            for step in self.selection:
                if step in self.points: self.points[step] = max(0.0, min(1.0, self.points[step] + increment))
            self.update(); self.parent_app.save_curve_data(); event.accept(); return
        if k == Qt.Key.Key_Left or k == Qt.Key.Key_Right:
            if not self.selection: event.ignore(); super().keyPressEvent(event); return 
            delta = -1 if k == Qt.Key.Key_Left else 1
            min_s = min(self.selection); max_s = max(self.selection)
            if (min_s + delta < 0) or (max_s + delta > 63): return 
            self.push_to_undo(self.points.copy()); new_points = self.points.copy(); [new_points.pop(s, None) for s in self.selection]; new_sel = set()
            for s in self.selection: new_points[s+delta] = self.points[s]; new_sel.add(s+delta)
            self.points = new_points; self.selection = new_sel; self.update(); self.parent_app.save_curve_data(); event.accept(); return
        if k == keys.get("QUANTIZE", Qt.Key.Key_Q): self.quantize_selection(); return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and k == Qt.Key.Key_Z:
            self.perform_redo() if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else self.perform_undo(); return
        if k in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
            self.push_to_undo(self.points.copy())
            for step in list(self.selection):
                if step in self.points: del self.points[step]
            self.selection.clear(); self.update(); self.parent_app.save_curve_data()
        else: super().keyPressEvent(event)

    def erase_at_pos(self, pos):
        step = self.get_step_from_x(pos.x())
        if step in self.points and self.get_rect_for_note(step, self.points[step]).adjusted(-5,-20,5,20).contains(pos):
            del self.points[step]; self.selection.discard(step); self.update()
    def interpolate_erase(self, p1, p2):
        steps = int(math.hypot(p2.x()-p1.x(), p2.y()-p1.y()) / 5) + 1 
        for i in range(steps + 1): t = i / steps; self.erase_at_pos(QPointF(p1.x() + (p2.x()-p1.x())*t, p1.y() + (p2.y()-p1.y())*t))
    def mousePressEvent(self, event):
        self.setFocus(); self.state_at_press = self.points.copy(); pos = event.position(); self.last_mouse_pos = pos; step = self.get_step_from_x(pos.x())
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or (event.button() == Qt.MouseButton.RightButton):
            self.mode = "ERASING"; self.setCursor(Qt.CursorShape.ForbiddenCursor); self.erase_at_pos(pos); return
        clicked = -1
        for s, v in self.points.items():
            if self.get_rect_for_note(s, v).adjusted(-2,-5,2,5).contains(pos): clicked = s; break
        if clicked != -1:
            if clicked not in self.selection:
                if not (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier): self.selection.clear()
                self.selection.add(clicked)
            self.mode = "MOVING"; self.drag_start_pos = pos; self.move_snapshot = {s: self.points[s] for s in self.selection}
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: self.clean_slate_points = self.points.copy(); self.setCursor(Qt.CursorShape.DragCopyCursor)
            else: self.clean_slate_points = self.points.copy(); [self.clean_slate_points.pop(s, None) for s in self.selection]; self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.points = self.clean_slate_points.copy()
        else:
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: self.mode = "SELECTING"; self.drag_start_pos = pos; self.marquee_rect = QRectF(pos, pos)
            else:
                if self.selection: self.selection.clear(); self.mode = "IDLE"
                else: self.selection.clear(); self.mode = "DRAWING"; self.points[step] = self.get_val_from_y(pos.y()); self.selection.add(step); self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.mode == "ERASING": self.interpolate_erase(self.last_mouse_pos, pos)
        elif self.mode == "SELECTING":
            self.marquee_rect = QRectF(self.dragged_rect(self.drag_start_pos, pos)); self.selection.clear()
            for s, v in self.points.items():
                if self.marquee_rect.intersects(self.get_rect_for_note(s, v)): self.selection.add(s)
            self.update()
        elif self.mode == "MOVING":
            d_s = int((pos.x()-self.drag_start_pos.x())/(self.width()/64)); d_v = -(pos.y()-self.drag_start_pos.y())/self.height()
            self.points = self.clean_slate_points.copy(); new_sel = set()
            for os, ov in self.move_snapshot.items():
                ns = os + d_s; nv = max(0.0, min(ov + d_v, 1.0))
                if 0 <= ns < 64: self.points[ns] = nv; new_sel.add(ns)
            self.selection = new_sel; self.update()
        elif self.mode == "DRAWING": self.points[self.get_step_from_x(pos.x())] = self.get_val_from_y(pos.y()); self.update()
        else:
            step = self.get_step_from_x(pos.x()); hover = False
            for s, v in self.points.items():
                if s == step and self.get_rect_for_note(s, v).contains(pos): hover = True; break
            self.setCursor(Qt.CursorShape.OpenHandCursor if hover else Qt.CursorShape.ArrowCursor)
        self.last_mouse_pos = pos
    def mouseReleaseEvent(self, event):
        if self.points != self.state_at_press: self.push_to_undo(self.state_at_press)
        self.mode = "IDLE"; self.marquee_rect = QRectF(); self.setCursor(Qt.CursorShape.ArrowCursor); self.parent_app.save_curve_data(); self.update()
    def dragged_rect(self, p1, p2): return QRectF(p1, p2).normalized()
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing, False); w = self.width(); h = self.height(); step_w = w / 64
        painter.fillRect(self.rect(), QColor("#080808")); lx = int(self.loop_start * step_w); lw = int(self.loop_length * step_w)
        painter.fillRect(0, 0, lx, h, QColor(0,0,0,180)); painter.fillRect(lx+lw, 0, w-(lx+lw), h, QColor(0,0,0,180))
        painter.setPen(QPen(QColor(40,40,40), 1)); [painter.drawLine(int(i*step_w),0,int(i*step_w),h) for i in range(0,64,4)]
        painter.setPen(QPen(QColor(30,30,30), 1)); [painter.drawLine(0,int(i*(h/5)),w,int(i*(h/5))) for i in range(1,5)]
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(255,255,255,30)); painter.drawRect(int(self.current_step*step_w), 0, int(step_w), h)
        for s, v in self.points.items():
            in_loop = self.loop_start <= s < (self.loop_start + self.loop_length)
            painter.setBrush(QColor("#FFFFFF") if s in self.selection else (QColor("#00CCFF") if in_loop else QColor("#004455")))
            rect = self.get_rect_for_note(s, v); painter.drawRect(rect)
            painter.setPen(QPen(QColor(0,204,255,60) if in_loop else QColor(0,50,60,40), 1))
            painter.drawLine(int(rect.center().x()), int(rect.bottom()), int(rect.center().x()), h); painter.setPen(Qt.PenStyle.NoPen)
        if self.mode == "SELECTING": painter.setPen(QPen(QColor(255,255,255),1,Qt.PenStyle.DashLine)); painter.setBrush(QColor(255,255,255,30)); painter.drawRect(self.marquee_rect)

# ==========================================
# 4. MAIN APPLICATION (4-TRACK MIXER)
# ==========================================

class LooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VJ Sequencer v24.0 (Video Link)")
        self.resize(800, 750)
        QApplication.instance().setStyleSheet(DARK_THEME)

        self.key_bindings = { 
            "PLAY_PAUSE": Qt.Key.Key_Space, "TOGGLE_SEQUENCER": Qt.Key.Key_P, "TAP_TEMPO": Qt.Key.Key_Return, 
            "BANK_1": Qt.Key.Key_5, "BANK_2": Qt.Key.Key_6, "BANK_3": Qt.Key.Key_7, "QUANTIZE": Qt.Key.Key_Q,
            "MUTE_A": Qt.Key.Key_1, "MUTE_S": Qt.Key.Key_2, "MUTE_D": Qt.Key.Key_3, "MUTE_F": Qt.Key.Key_4
        }
        self.midi_map = { "FADER_A": None, "FADER_S": None, "FADER_D": None, "FADER_F": None, "PLAY_PAUSE": None, "TOGGLE_SEQUENCER": None, "TAP_TEMPO": None, "TRIGGER_A": None, "TRIGGER_S": None, "TRIGGER_D": None, "TRIGGER_F": None }
        
        self.midi_worker = MidiWorker(); self.midi_worker.message_received.connect(self.handle_midi_message); self.midi_worker.start()

        # --- 2x2 VIDEO GRID SCENE ---
        self.projector = QWidget(); self.projector.resize(800,600); self.projector.setStyleSheet("background:black")
        self.proj_scene = QGraphicsScene(0,0,800,600); self.proj_view = QGraphicsView(self.projector); self.proj_view.setViewport(QOpenGLWidget()); self.proj_view.resize(800,600); self.proj_view.setScene(self.proj_scene)
        self.proj_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.proj_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.track_items = {}; positions = [(0,0), (400,0), (0,300), (400,300)]; keys = ['a', 's', 'd', 'f']
        for i, k in enumerate(keys):
            item = QGraphicsVideoItem(); item.setSize(QSizeF(400, 300)); item.setPos(positions[i][0], positions[i][1])
            self.proj_scene.addItem(item); self.track_items[k] = item
        self.projector.show()

        self.tracks = {}; 
        for k in keys: self.tracks[k] = VJDeck(f"Track {k.upper()}", self.track_items[k])

        self.buttons = {}; self.faders = {}; self.mute_buttons = {}
        self.bank_data = {0: {}, 1: {}, 2: {}}; self.clip_meta = {}; self.clip_curves = {}; self.clip_loops = {}
        self.active_edit_track = 'a'; self.current_bank = 0; self.current_generation = 0; self.active_workers = []; self.master_bpm = 120.0; self.tap_times = []; self.transport_start_time = time.time()
        self.seq_running = False; self.current_step = 0; self.seq_multiplier = 1.0; self.seq_timer = QTimer(); self.seq_timer.setTimerType(Qt.TimerType.PreciseTimer); self.seq_timer.timeout.connect(self.run_sequencer_step)
        self.mute_states = {'a': False, 's': False, 'd': False, 'f': False}

        scroll = QScrollArea(); scroll.setWidgetResizable(True); self.setCentralWidget(scroll); w = QWidget(); w.setObjectName("Container"); scroll.setWidget(w); l = QVBoxLayout(w); l.setSpacing(5)

        # 1. TOP BAR
        top = QHBoxLayout()
        for txt, fn in [("SAVE", self.save_set), ("LOAD", self.load_set), ("KEYS", self.open_hotkey_editor), ("MIDI", self.open_midi_editor)]:
            b = QPushButton(txt); b.clicked.connect(fn); top.addWidget(b)
        self.btn_vid_sync = QPushButton("SYNC: ON"); self.btn_vid_sync.setCheckable(True); self.btn_vid_sync.setChecked(True); self.btn_vid_sync.setProperty("sync","true"); self.btn_vid_sync.clicked.connect(self.toggle_vid_sync)
        self.btn_align = QPushButton("ALIGN"); self.btn_align.clicked.connect(self.auto_align_phase)
        top.addWidget(self.btn_vid_sync); top.addWidget(self.btn_align); l.addLayout(top)

        # 2. AUDIO & BANK
        io = QHBoxLayout(); devs = QMediaDevices.audioOutputs(); self.c_main = QComboBox(); self.c_cue = QComboBox()
        for d in devs: self.c_main.addItem(d.description()); self.c_cue.addItem(d.description())
        self.c_main.currentIndexChanged.connect(self.change_main_output); self.c_cue.currentIndexChanged.connect(self.change_cue_output)
        io.addWidget(QLabel("MAIN:")); io.addWidget(self.c_main); io.addWidget(QLabel("CUE:")); io.addWidget(self.c_cue)
        self.bank_btns = []
        for i in range(3):
            b = QPushButton(f"BANK {i+1}"); b.setCheckable(True); b.clicked.connect(lambda _, x=i: self.switch_bank(x)); io.addWidget(b); self.bank_btns.append(b)
        self.bank_btns[0].setChecked(True); l.addLayout(io)

        # 3. MIXER (STRIPS)
        mixer = QHBoxLayout()
        for k, (r,c,col) in KEY_MAP.items():
            mixer.addLayout(self.create_mixer_strip(k, col))
        l.addLayout(mixer)

        # 4. TRANSPORT
        trans = QHBoxLayout()
        self.bpm_lbl = QLabel("120.0 BPM")
        btn_d = QPushButton("-"); btn_d.setProperty("nudge", "true"); btn_d.clicked.connect(lambda: self.nudge_bpm(-0.1))
        btn_u = QPushButton("+"); btn_u.setProperty("nudge", "true"); btn_u.clicked.connect(lambda: self.nudge_bpm(0.1))
        self.c_speed = QComboBox(); self.c_speed.addItems(["1/2x", "1x", "2x"]); self.c_speed.setCurrentIndex(1); self.c_speed.currentIndexChanged.connect(self.change_seq_speed)
        trans.addWidget(btn_d); trans.addWidget(self.bpm_lbl); trans.addWidget(btn_u); trans.addWidget(QLabel("RATE:")); trans.addWidget(self.c_speed)
        btn_tap = QPushButton("TAP"); btn_tap.clicked.connect(self.handle_tap_tempo); trans.addWidget(btn_tap)
        trans.addStretch()
        trans.addWidget(QLabel("LOOP:")); self.loop_btns = QButtonGroup()
        for s in [4, 8, 16, 32, 64]:
            b = QPushButton(str(s)); b.setCheckable(True); b.setProperty("loop", "true"); 
            if s == 64: b.setChecked(True)
            b.clicked.connect(lambda _, size=s: self.set_loop_length(size)); self.loop_btns.addButton(b, s); trans.addWidget(b)
        l.addLayout(trans)

        # 5. SEQ TOOLS
        stools = QHBoxLayout()
        self.rad_group = QButtonGroup()
        for k in ['a', 's', 'd', 'f']:
            r = QRadioButton(f"EDIT {k.upper()}"); 
            if k == 'a': r.setChecked(True)
            r.toggled.connect(lambda checked, key=k: self.change_edit_track(key) if checked else None); self.rad_group.addButton(r); stools.addWidget(r)
        self.chk_loop_track = QCheckBox("LOOP CLIP"); self.chk_loop_track.toggled.connect(self.toggle_loop_current_track); stools.addWidget(self.chk_loop_track)
        stools.addStretch()
        btn_q = QPushButton("QUANTIZE (Q)"); btn_q.clicked.connect(lambda: self.piano_roll.quantize_selection()); stools.addWidget(btn_q)
        self.btn_run = QPushButton("RUN SEQ (P)"); self.btn_run.setCheckable(True); self.btn_run.clicked.connect(self.toggle_sequencer); stools.addWidget(self.btn_run)
        l.addLayout(stools)

        # 6. SEQUENCER
        self.piano_roll = PianoRollSequencer(self); self.loop_bar = LoopBar(self.piano_roll)
        l.addWidget(self.loop_bar); l.addWidget(self.piano_roll)

        self.reopen_btn = QPushButton("OPEN PROJECTOR WINDOW"); self.reopen_btn.clicked.connect(self.projector.show); l.addWidget(self.reopen_btn)
        QApplication.instance().installEventFilter(self); self.update_mixer()

    def create_mixer_strip(self, k, col):
        v = QVBoxLayout()
        pad = InteractiveWaveform(k, col, self); self.buttons[k] = pad; v.addWidget(pad)
        fader = QSlider(Qt.Orientation.Vertical); fader.setRange(0, 100); fader.setValue(100); fader.setMinimumHeight(120); fader.setMinimumWidth(40)
        fader.valueChanged.connect(lambda val, key=k: self.set_track_volume(key, val))
        self.faders[k] = fader; v.addWidget(fader, 0, Qt.AlignmentFlag.AlignCenter)
        mute = QPushButton("M"); mute.setCheckable(True); mute.setProperty("mute", "true")
        mute.toggled.connect(lambda checked, key=k: self.toggle_track_mute(key, checked))
        self.mute_buttons[k] = mute; v.addWidget(mute, 0, Qt.AlignmentFlag.AlignCenter)
        return v

    def select_track_for_edit(self, key):
        for btn in self.rad_group.buttons():
            if btn.text().lower().endswith(key): btn.setChecked(True)

    def change_edit_track(self, key):
        self.active_edit_track = key
        self.update_curve_ui()
        path = self.tracks[key].current_filepath
        if path:
            state = self.clip_loops.get(path, True)
            self.chk_loop_track.blockSignals(True); self.chk_loop_track.setChecked(state); self.chk_loop_track.blockSignals(False)

    def toggle_loop_current_track(self, state):
        t = self.tracks[self.active_edit_track]; t.set_loop_mode(state)
        if t.current_filepath: self.clip_loops[t.current_filepath] = state

    def load_track(self, key, path):
        self.bank_data[self.current_bank][key] = path
        self.buttons[key].set_loading()
        w = AudioAnalysisWorker(key, path, 200, 120, self.buttons[key].base_color.name(), self.current_generation)
        w.finished.connect(self.prep_done); self.active_workers.append(w); w.start()
        t = self.tracks[key]; t.load_video(path)
        loop_state = self.clip_loops.get(path, True); t.set_loop_mode(loop_state)
        if key == self.active_edit_track:
            self.chk_loop_track.blockSignals(True); self.chk_loop_track.setChecked(loop_state); self.chk_loop_track.blockSignals(False)
        t.play(); self.update_curve_ui()

    def set_track_volume(self, key, val):
        if not self.mute_states[key]: self.tracks[key].set_volume(val / 100.0)

    def toggle_track_mute(self, key, muted):
        self.mute_states[key] = muted
        self.mute_buttons[key].blockSignals(True); self.mute_buttons[key].setChecked(muted); self.mute_buttons[key].blockSignals(False)
        self.tracks[key].set_volume(0 if muted else self.faders[key].value() / 100.0)

    def prep_done(self, key, pix, bpm, dur, raw, rate, wav):
        path = self.bank_data[self.current_bank].get(key)
        if path:
            self.clip_meta[path] = bpm
            self.tracks[key].set_audio_data(raw, rate)
            self.tracks[key].load_base_audio(wav)
        self.buttons[key].set_data(pix, bpm, dur)

    def switch_bank(self, i):
        self.current_bank = i; self.current_generation += 1
        for b in self.bank_btns: b.setChecked(False)
        self.bank_btns[i].setChecked(True)
        for k in ['a','s','d','f']: 
            self.buttons[k].filename = "[Empty]"; self.buttons[k].update()
            p = self.bank_data[i].get(k)
            if p: self.load_track(k, p)

    def get_target_deck_info(self): t = self.tracks[self.active_edit_track]; return (t, t.current_filepath)

    def run_sequencer_step(self):
        ls = self.piano_roll.loop_start; ll = self.piano_roll.loop_length
        self.current_step = ls + ((self.current_step + 1 - ls) % ll)
        self.piano_roll.current_step = self.current_step; self.piano_roll.update()
        for k in ['a','s','d','f']:
            t = self.tracks[k]; path = t.current_filepath
            if path and path in self.clip_curves and self.current_step in self.clip_curves[path]:
                t.trigger(int(self.clip_curves[path][self.current_step] * t.duration()))

    def toggle_play_state(self):
        for t in self.tracks.values(): t.play() if t.has_media() and t.playbackState()!=QMediaPlayer.PlaybackState.PlayingState else t.pause()

    def update_mixer(self): pass
    def change_main_output(self, i): d = self.audio_devices[i]; [t.set_main_output(d) for t in self.tracks.values()]
    def change_cue_output(self, i): d = self.audio_devices[i]; [t.set_cue_output(d) for t in self.tracks.values()]

    def sync_deck(self, deck, key):
        path = deck.current_filepath; cb = self.clip_meta.get(path, 120.0) if path else 120.0
        rate = self.master_bpm / cb if cb > 0 else 1.0; deck.setPlaybackRate(rate)
        if deck.base_wav_path: w = RubberBandWorker(key, deck.base_wav_path, rate); w.finished.connect(lambda k,p,r: deck.swap_audio(p,False)); self.active_workers.append(w); w.start()

    def toggle_vid_sync(self):
        on = self.btn_vid_sync.isChecked(); self.btn_vid_sync.setText(f"SYNC: {'ON' if on else 'OFF'}")
        if on: [self.sync_deck(self.tracks[k], k) for k in self.tracks]
        else: [t.setPlaybackRate(1.0) for t in self.tracks.values()]

    def auto_align_phase(self):
        if self.master_bpm <= 0: return
        beat_ms = 60000.0/self.master_bpm; phase = (time.time() - self.transport_start_time)*1000 % beat_ms
        for t in self.tracks.values():
            if t.has_media(): diff = phase - (t.position() % beat_ms); t.seek(max(0, int(t.position() + (diff + beat_ms if diff < -beat_ms/2 else diff - beat_ms if diff > beat_ms/2 else diff))))

    def set_loop_length(self, length): self.piano_roll.set_loop_window(self.piano_roll.loop_start, length)
    def nudge_bpm(self, amount):
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: amount *= 10
        self.master_bpm = round(max(10.0, self.master_bpm + amount), 1); self.bpm_lbl.setText(f"{self.master_bpm} BPM")
        if self.btn_vid_sync.isChecked(): [self.sync_deck(self.tracks[k], k) for k in self.tracks]
        self.update_clock()
    def update_curve_ui(self): _, path = self.get_target_deck_info(); self.piano_roll.set_data(self.clip_curves.get(path, {}))
    def save_curve_data(self): _, path = self.get_target_deck_info(); self.clip_curves[path] = self.piano_roll.get_data() if path else None
    def toggle_sequencer(self): self.seq_running = not self.seq_running; self.btn_run.setChecked(self.seq_running); self.update_clock() if self.seq_running else self.seq_timer.stop()
    def update_clock(self):
        if self.master_bpm > 0: self.seq_timer.setInterval(int(((60000.0 / self.master_bpm) / 4) / self.seq_multiplier)); 
        if self.seq_running and not self.seq_timer.isActive(): self.seq_timer.start()
    def change_seq_speed(self, i): self.seq_multiplier = [0.5, 1.0, 2.0][i]; self.update_clock()
    def handle_tap_tempo(self):
        now = time.time(); self.tap_times.append(now)
        if len(self.tap_times)>4: self.tap_times.pop(0)
        if len(self.tap_times)>1:
            avg = sum([self.tap_times[i+1]-self.tap_times[i] for i in range(len(self.tap_times)-1)]) / (len(self.tap_times)-1)
            self.master_bpm = round(60.0/avg, 1); self.bpm_lbl.setText(f"{self.master_bpm} BPM")
            if self.btn_vid_sync.isChecked(): [self.sync_deck(self.tracks[k], k) for k in self.tracks]
            self.update_clock()
            
    def open_hotkey_editor(self):
        editor = HotkeyEditor(self.key_bindings, self)
        if editor.exec() == QDialog.DialogCode.Accepted: self.key_bindings = editor.get_bindings()
    def open_midi_editor(self): editor = MidiConfigDialog(self.midi_worker, self.midi_map, self); editor.exec()
    def handle_midi_message(self, msg):
        for action, binding in self.midi_map.items():
            if not binding: continue
            if (msg.type == binding['type'] and (msg.control if msg.type == 'control_change' else msg.note) == binding['val']):
                if action.startswith("FADER_"): self.faders[action.split("_")[1].lower()].setValue(int((msg.value / 127.0) * 100))
                elif action == "PLAY_PAUSE" and msg.type == 'note_on' and msg.velocity > 0: self.toggle_play_state()
                elif action == "TOGGLE_SEQUENCER" and msg.type == 'note_on' and msg.velocity > 0: self.toggle_sequencer()
                elif action == "TAP_TEMPO" and msg.type == 'note_on' and msg.velocity > 0: self.handle_tap_tempo()
                elif action.startswith("TRIGGER_") and msg.type == 'note_on' and msg.velocity > 0:
                    k = action.split("_")[1].lower(); self.select_track_for_edit(k); self.tracks[k].seek(0); self.tracks[k].play()

    def on_deck_a_pos(self, p): self.buttons['a'].update_playhead(p/self.tracks['a'].duration())
    def on_deck_b_pos(self, p): self.buttons['b'].update_playhead(p/self.tracks['b'].duration()) 
    
    def load_set(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if f: 
            d=json.load(open(f,'r')); self.bank_data=d['banks']
            self.key_bindings={k:int(v) for k,v in d.get('keys',self.key_bindings).items()}
            self.midi_map=d.get('midi',self.midi_map)
            self.clip_curves={path:{int(k):v for k,v in points.items()} for path,points in d.get('curves',{}).items()}
            self.clip_loops=d.get('loops', {})
            self.switch_bank(0)
    def save_set(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if f: json.dump({'banks':self.bank_data, 'curves':self.clip_curves, 'keys':self.key_bindings, 'midi':self.midi_map, 'loops':self.clip_loops}, open(f,'w'))

    def eventFilter(self, src, e):
        if e.type() == QEvent.Type.KeyPress and not e.isAutoRepeat():
            k = e.key(); keys = self.key_bindings
            if k == keys.get("MUTE_A"): self.toggle_track_mute('a', not self.mute_states['a']); return True
            if k == keys.get("MUTE_S"): self.toggle_track_mute('s', not self.mute_states['s']); return True
            if k == keys.get("MUTE_D"): self.toggle_track_mute('d', not self.mute_states['d']); return True
            if k == keys.get("MUTE_F"): self.toggle_track_mute('f', not self.mute_states['f']); return True
            if k == Qt.Key.Key_Left or k == Qt.Key.Key_Right:
                if self.piano_roll.hasFocus() and self.piano_roll.selection: return False 
            if k == keys["PLAY_PAUSE"]: self.toggle_play_state(); return True
            if k == keys["TOGGLE_SEQUENCER"]: self.toggle_sequencer(); return True
            if k == keys["TAP_TEMPO"]: self.handle_tap_tempo(); return True
            if k == keys["BANK_1"]: self.switch_bank(0); return True
            if k == keys["BANK_2"]: self.switch_bank(1); return True
            if k == keys["BANK_3"]: self.switch_bank(2); return True
        return super().eventFilter(src, e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LooperApp()
    window.show()
    sys.exit(app.exec())