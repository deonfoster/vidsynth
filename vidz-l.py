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
from pydub import AudioSegment

# --- IMPORTS ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QVBoxLayout, QPushButton, QSlider,
                             QFileDialog, QHBoxLayout, QComboBox, QScrollArea,
                             QSpinBox, QRadioButton, QButtonGroup, QFrame,
                             QGraphicsView, QGraphicsScene) 
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtMultimedia import (QMediaPlayer, QAudioOutput, QMediaDevices)
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import (QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, 
                          QRectF, QPointF, QSizeF, QRect)
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QPolygonF, QFont, QCursor, QAction

# --- STYLING ---
DARK_THEME = """
QWidget { background-color: #121212; color: #ffffff; font-family: 'Arial', sans-serif; font-size: 14px; }
QMainWindow, QScrollArea { background-color: #121212; border: none; }
QLabel { color: #e0e0e0; font-weight: 600; background-color: transparent; padding: 2px; }
QPushButton { background-color: #2a2a2a; color: #ffffff; border: 1px solid #444; padding: 8px; border-radius: 4px; font-weight: bold; min-width: 60px; }
QPushButton:hover { background-color: #3d3d3d; border: 1px solid #666; }
QPushButton:pressed { background-color: #00CCFF; color: #000000; }
QPushButton:checked { background-color: #00CCFF; color: #000000; border: 1px solid #ffffff; }
QPushButton[active="true"] { border: 2px solid #ffffff; background-color: #444; }
QPushButton[sync="true"]:checked { background-color: #FF0055; color: white; border: 1px solid white; }
QPushButton[nudge="true"] { min-width: 30px; max-width: 30px; background-color: #222; font-size: 16px; }
QPushButton[loop="true"] { min-width: 40px; font-size: 12px; } 
QRadioButton { color: #cccccc; font-weight: bold; background-color: transparent; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 1px solid #555; background-color: #222; }
QRadioButton::indicator:checked { background-color: #00CCFF; border: 2px solid white; }
QSpinBox, QComboBox { background-color: #222; color: #ffffff; border: 1px solid #444; padding: 6px; border-radius: 3px; font-weight: bold; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #222; color: white; selection-background-color: #00CCFF; selection-color: black; }
QSlider::groove:horizontal { border: 1px solid #333; height: 8px; background: #1a1a1a; margin: 2px 0; border-radius: 4px; }
QSlider::handle:horizontal { background: #00CCFF; border: 1px solid #00CCFF; width: 24px; height: 24px; margin: -9px 0; border-radius: 12px; }
QSlider::sub-page:horizontal { background: #444; border-radius: 4px; }
QScrollBar:vertical { border: none; background: #1a1a1a; width: 12px; margin: 0px; }
QScrollBar::handle:vertical { background: #444; min-height: 20px; border-radius: 6px; }
"""

KEY_MAP = {'a': (0, 0, "#FF0055"), 's': (0, 1, "#00CCFF"), 'd': (1, 0, "#00FF66"), 'f': (1, 1, "#FFAA00")}

# --- LOOP BAR WIDGET ---
class LoopBar(QWidget):
    def __init__(self, parent_sequencer):
        super().__init__()
        self.sequencer = parent_sequencer
        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        self.setMouseTracking(True)
        self.dragging = False
        self.drag_start_x = 0
        self.start_step_cache = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        step_w = w / 64.0

        painter.fillRect(self.rect(), QColor("#111"))

        start_x = self.sequencer.loop_start * step_w
        loop_w = self.sequencer.loop_length * step_w
        
        bar_rect = QRectF(start_x, 2, loop_w, h - 4)
        painter.setBrush(QColor("#00CCFF"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bar_rect, 4, 4)
        
        painter.setPen(QColor("black"))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        label = f"{self.sequencer.loop_length} STEPS"
        painter.drawText(bar_rect, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event):
        step_w = self.width() / 64.0
        start_x = self.sequencer.loop_start * step_w
        loop_w = self.sequencer.loop_length * step_w
        bar_rect = QRectF(start_x, 0, loop_w, self.height())

        if bar_rect.contains(event.position()):
            self.dragging = True
            self.drag_start_x = event.position().x()
            self.start_step_cache = self.sequencer.loop_start
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            click_step = int(event.position().x() / step_w)
            new_start = click_step - (self.sequencer.loop_length // 2)
            self.sequencer.set_loop_window(new_start, self.sequencer.loop_length)
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            step_w = self.width() / 64.0
            delta_pixels = event.position().x() - self.drag_start_x
            delta_steps = int(delta_pixels / step_w)
            
            new_start = self.start_step_cache + delta_steps
            self.sequencer.set_loop_window(new_start, self.sequencer.loop_length)
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

# --- PIANO ROLL SEQUENCER ---
class PianoRollSequencer(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #080808; border: 2px solid #333; margin-top: 0px; border-radius: 0px 0px 4px 4px;")
        
        self.points = {} 
        self.selection = set()
        
        self.current_step = 0
        self.steps = 64
        
        # Loop State
        self.loop_start = 0
        self.loop_length = 64
        
        self.mode = "IDLE" 
        self.drag_start_pos = QPointF()
        self.last_mouse_pos = QPointF() 
        self.marquee_rect = QRectF()
        self.move_snapshot = {} 
        self.clean_slate_points = {}
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus) 

    def set_loop_window(self, start, length):
        self.loop_length = length
        self.loop_start = max(0, min(start, 64 - length))
        self.update()
        if hasattr(self.parent_app, 'loop_bar'):
            self.parent_app.loop_bar.update()

    def set_data(self, data):
        self.points = data.copy() if data else {}
        self.selection.clear()
        self.update()

    def get_data(self): return self.points

    def get_step_from_x(self, x):
        step_w = self.width() / self.steps
        return max(0, min(int(x / step_w), self.steps - 1))

    def get_val_from_y(self, y):
        val = 1.0 - (y / self.height())
        return max(0.0, min(val, 1.0))

    def get_rect_for_note(self, step, val):
        step_w = self.width() / self.steps
        h = self.height()
        block_h = 20
        x = int(step * step_w)
        y = int(h - (val * h)) - (block_h // 2)
        y = max(0, min(y, h - block_h))
        return QRectF(x, y, step_w, block_h)

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
            for step in list(self.selection):
                if step in self.points: del self.points[step]
            self.selection.clear()
            self.update()
            self.parent_app.save_curve_data()
        else:
            super().keyPressEvent(event)

    def erase_at_pos(self, pos):
        step = self.get_step_from_x(pos.x())
        if step in self.points:
            val = self.points[step]
            rect = self.get_rect_for_note(step, val)
            if rect.adjusted(-5, -20, 5, 20).contains(pos):
                del self.points[step]
                if step in self.selection: self.selection.remove(step)
                self.update()

    def interpolate_erase(self, p1, p2):
        steps = int(math.hypot(p2.x()-p1.x(), p2.y()-p1.y()) / 5) + 1 
        for i in range(steps + 1):
            t = i / steps
            x = p1.x() + (p2.x() - p1.x()) * t
            y = p1.y() + (p2.y() - p1.y()) * t
            self.erase_at_pos(QPointF(x, y))

    def mousePressEvent(self, event):
        self.setFocus() 
        pos = event.position()
        self.last_mouse_pos = pos 
        step = self.get_step_from_x(pos.x())
        val = self.get_val_from_y(pos.y())
        
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) or (event.button() == Qt.MouseButton.RightButton):
            self.mode = "ERASING"
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            self.erase_at_pos(pos)
            return

        clicked_note_step = -1
        for s, v in self.points.items():
            rect = self.get_rect_for_note(s, v)
            if rect.adjusted(-2, -5, 2, 5).contains(pos):
                clicked_note_step = s
                break

        if clicked_note_step != -1:
            if clicked_note_step not in self.selection:
                if not (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
                    self.selection.clear()
                self.selection.add(clicked_note_step)
            
            self.mode = "MOVING"
            self.drag_start_pos = pos
            self.move_snapshot = {s: self.points[s] for s in self.selection}
            
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.clean_slate_points = self.points.copy() 
                self.setCursor(Qt.CursorShape.DragCopyCursor)
            else:
                self.clean_slate_points = self.points.copy()
                for s in self.selection:
                    if s in self.clean_slate_points: del self.clean_slate_points[s]
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
            self.points = self.clean_slate_points.copy()
            
        else:
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.mode = "SELECTING"
                self.drag_start_pos = pos
                self.marquee_rect = QRectF(pos, pos)
            else:
                if self.selection:
                    self.selection.clear()
                    self.mode = "IDLE"
                else:
                    self.selection.clear()
                    self.mode = "DRAWING"
                    self.points[step] = val
                    self.selection.add(step)
                    self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        
        if self.mode == "ERASING":
            self.interpolate_erase(self.last_mouse_pos, pos)

        elif self.mode == "SELECTING":
            self.marquee_rect = QRectF(self.dragged_rect(self.drag_start_pos, pos))
            self.selection.clear()
            for s, v in self.points.items():
                rect = self.get_rect_for_note(s, v)
                if self.marquee_rect.intersects(rect):
                    self.selection.add(s)
            self.update()

        elif self.mode == "MOVING":
            step_w = self.width() / self.steps
            delta_steps = int((pos.x() - self.drag_start_pos.x()) / step_w)
            delta_val = -(pos.y() - self.drag_start_pos.y()) / self.height()
            
            min_step = min(self.move_snapshot.keys())
            max_step = max(self.move_snapshot.keys())
            if min_step + delta_steps < 0: delta_steps = -min_step
            if max_step + delta_steps > 63: delta_steps = 63 - max_step
            
            self.points = self.clean_slate_points.copy()
            new_selection = set()
            for old_step, old_val in self.move_snapshot.items():
                new_step = old_step + delta_steps
                new_val = max(0.0, min(old_val + delta_val, 1.0))
                self.points[new_step] = new_val
                new_selection.add(new_step)
            
            self.selection = new_selection
            self.update()

        elif self.mode == "DRAWING":
            step = self.get_step_from_x(pos.x())
            val = self.get_val_from_y(pos.y())
            self.points[step] = val
            self.selection.add(step)
            self.update()
            
        else:
            step = self.get_step_from_x(pos.x())
            hover = False
            for s, v in self.points.items():
                if s == step: 
                    rect = self.get_rect_for_note(s, v)
                    if rect.contains(pos): hover = True; break
            if hover: self.setCursor(Qt.CursorShape.OpenHandCursor)
            else: self.setCursor(Qt.CursorShape.ArrowCursor)
            
        self.last_mouse_pos = pos

    def mouseReleaseEvent(self, event):
        self.mode = "IDLE"
        self.marquee_rect = QRectF()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.move_snapshot = {}
        self.clean_slate_points = {}
        self.parent_app.save_curve_data()
        self.update()

    def dragged_rect(self, p1, p2): return QRectF(p1, p2).normalized()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width(); h = self.height(); step_w = w / self.steps

        painter.fillRect(self.rect(), QColor("#080808"))

        loop_x = int(self.loop_start * step_w)
        loop_w_px = int(self.loop_length * step_w)
        
        dim_color = QColor(0, 0, 0, 180)
        painter.fillRect(0, 0, loop_x, h, dim_color)
        painter.fillRect(loop_x + loop_w_px, 0, w - (loop_x + loop_w_px), h, dim_color)

        painter.setPen(QPen(QColor(40, 40, 40), 1))
        for i in range(0, self.steps, 4):
            x = int(i * step_w)
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            painter.drawLine(x, 0, x, h)
        painter.setPen(QPen(QColor(30, 30, 30), 1))
        for i in range(1, 5):
            y = int(i * (h/5))
            painter.drawLine(0, y, w, y)

        ph_x = int(self.current_step * step_w)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.drawRect(ph_x, 0, int(step_w), h)

        for step, val in self.points.items():
            rect = self.get_rect_for_note(step, val)
            is_in_loop = self.loop_start <= step < (self.loop_start + self.loop_length)
            
            if step in self.selection:
                painter.setBrush(QColor("#FFFFFF"))
                painter.setPen(QPen(QColor("#00CCFF"), 2))
            else:
                base_col = QColor("#00CCFF")
                if not is_in_loop: base_col = QColor("#004455") 
                painter.setBrush(base_col)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
            
            painter.drawRect(rect)
            
            stem_x = int(rect.center().x())
            stem_y = int(rect.bottom())
            stem_col = QColor(0, 204, 255, 60) if is_in_loop else QColor(0, 50, 60, 40)
            painter.setPen(QPen(stem_col, 1))
            painter.drawLine(stem_x, stem_y, stem_x, h)

        if self.mode == "SELECTING":
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(self.marquee_rect)

# --- WORKERS (Unchanged) ---
class AudioAnalysisWorker(QThread):
    finished = pyqtSignal(str, QPixmap, float, int, object, int, str)
    def __init__(self, key, filepath, width, height, color_hex, gen_id):
        super().__init__()
        self.key, self.filepath, self.width, self.height = key, filepath, width, height
        self.bg_color, self.gen_id = QColor(color_hex), gen_id

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            audio_full = AudioSegment.from_file(self.filepath)
            
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            clean_name = os.path.basename(self.filepath).replace(" ", "_")
            wav_path = os.path.join(temp_dir, f"{clean_name}_base.wav")
            if not os.path.exists(wav_path):
                audio_full.export(wav_path, format="wav")

            duration_ms = len(audio_full)
            if duration_ms > 60000: audio_vis = audio_full[:60000]
            else: audio_vis = audio_full
            
            raw_samples = np.array(audio_full.get_array_of_samples())
            sample_rate = audio_full.frame_rate

            vis_samples = np.array(audio_vis.set_channels(1).set_frame_rate(11025).get_array_of_samples())
            tempo, _ = librosa.beat.beat_track(y=vis_samples.astype(np.float32)/32768.0, sr=11025)
            bpm = float(tempo.item()) if isinstance(tempo, np.ndarray) else float(round(tempo, 2))

            draw_samples = vis_samples[::150]
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(self.bg_color.darker(150), 1))
            center_y = self.height / 2
            step = len(draw_samples) / self.width
            for x in range(self.width):
                if self.isInterruptionRequested(): return
                idx = int(x * step)
                if idx < len(draw_samples):
                    h = abs(draw_samples[idx]) * (self.height * 0.9) / 32768.0
                    painter.drawLine(x, int(center_y - h/2), x, int(center_y + h/2))
            painter.end()
            
            if not self.isInterruptionRequested(): 
                self.finished.emit(self.key, pixmap, bpm, duration_ms, raw_samples, sample_rate, wav_path)
        except:
            if not self.isInterruptionRequested(): self.finished.emit(self.key, QPixmap(), 120.0, 0, None, 44100, "")

class RubberBandWorker(QThread):
    finished = pyqtSignal(str, str, float)
    def __init__(self, key, original_wav, tempo_ratio):
        super().__init__()
        self.key, self.original_wav, self.tempo_ratio = key, original_wav, tempo_ratio

    def run(self):
        try:
            if not os.path.exists(self.original_wav) or self.tempo_ratio <= 0: return
            unique_id = uuid.uuid4().hex[:8]
            base, ext = os.path.splitext(self.original_wav)
            out_path = f"{base}_st_{self.tempo_ratio:.2f}_{unique_id}{ext}"
            if shutil.which("rubberband") is None: return
            subprocess.run(["rubberband", "-q", "realtime", "-t", str(1.0/self.tempo_ratio), self.original_wav, out_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.finished.emit(self.key, out_path, self.tempo_ratio)
        except: pass

# --- DECK ---
class VJDeck:
    def __init__(self, name, video_item):
        self.name = name
        self.video_item = video_item
        self.current_filepath = None
        self.base_wav_path = None
        
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_item)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.video_audio = QAudioOutput(); self.player.setAudioOutput(self.video_audio); self.video_audio.setVolume(0) 

        self.audio_player = QMediaPlayer()
        self.main_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.main_output)
        self.audio_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.cue_player = QMediaPlayer()
        self.cue_output = QAudioOutput()
        self.cue_player.setAudioOutput(self.cue_output)
        self.cue_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.cue_active = False 
        self.raw_samples = None
        self.sample_rate = 44100
        self.target_volume = 1.0 
        self.playback_rate = 1.0
        
        self.fade_level = 1.0
        self.fade_timer = QTimer()
        self.fade_timer.setInterval(10)
        self.fade_timer.timeout.connect(self._process_fade)

    def load_video(self, filepath):
        self.current_filepath = filepath
        self.player.setSource(QUrl.fromLocalFile(filepath))

    def load_base_audio(self, wav_path):
        self.base_wav_path = wav_path
        self.swap_audio(wav_path, reset_rate_to_video=True)

    def swap_audio(self, path, reset_rate_to_video=False):
        pos = self.player.position()
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        url = QUrl.fromLocalFile(path)
        
        self.audio_player.setSource(url)
        self.cue_player.setSource(url)
        
        if reset_rate_to_video:
            self.audio_player.setPlaybackRate(self.playback_rate)
            self.cue_player.setPlaybackRate(self.playback_rate)
            self.audio_player.setPosition(pos)
            self.cue_player.setPosition(pos)
        else:
            self.audio_player.setPlaybackRate(1.0)
            self.cue_player.setPlaybackRate(1.0)
            mapped_pos = int(pos / self.playback_rate)
            self.audio_player.setPosition(mapped_pos)
            self.cue_player.setPosition(mapped_pos)

        if playing:
            self.audio_player.play()
            if self.cue_active: self.cue_player.play()

    def has_media(self): return self.player.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia
    def set_audio_data(self, samples, rate): self.raw_samples = samples; self.sample_rate = rate

    def find_zero_crossing(self, target_ms):
        if self.raw_samples is None: return target_ms
        idx = int((target_ms / 1000.0) * self.sample_rate); idx -= idx % 2 
        win = int(0.02 * self.sample_rate); s = max(0, idx - win); e = min(len(self.raw_samples), idx + win)
        if s >= e: return target_ms
        return int(((s + np.argmin(np.abs(self.raw_samples[s:e]))) / self.sample_rate) * 1000.0)

    def trigger(self, pos):
        self.main_output.setMuted(True)
        if self.cue_active: self.cue_output.setMuted(True)
        
        safe_pos = self.find_zero_crossing(pos)
        self.player.setPosition(safe_pos)
        
        audio_pos = safe_pos
        if self.audio_player.playbackRate() == 1.0 and self.playback_rate != 1.0:
            audio_pos = int(safe_pos / self.playback_rate)
            
        self.audio_player.setPosition(audio_pos)
        if self.cue_active: self.cue_player.setPosition(audio_pos)

        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.play(); self.audio_player.play()
            if self.cue_active: self.cue_player.play()

        self.fade_level = 0.0
        self.main_output.setVolume(0); self.main_output.setMuted(False)
        if self.cue_active: self.cue_output.setVolume(0); self.cue_output.setMuted(False)
        if not self.fade_timer.isActive(): self.fade_timer.start()

    def _process_fade(self):
        self.fade_level += 0.1
        if self.fade_level >= 1.0: self.fade_level = 1.0; self.fade_timer.stop()
        self.main_output.setVolume(self.target_volume * self.fade_level)
        if self.cue_active: self.cue_output.setVolume(1.0 * self.fade_level)

    def set_volume(self, vol):
        self.target_volume = vol
        if not self.fade_timer.isActive(): self.main_output.setVolume(vol)

    def set_cue_active(self, active):
        self.cue_active = active
        if active:
            self.cue_output.setVolume(1.0)
            self.cue_player.setPosition(self.audio_player.position())
            if self.audio_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.cue_player.play()
        else:
            self.cue_output.setVolume(0)

    def play(self): self.player.play(); self.audio_player.play(); self.cue_player.play() if self.cue_active else None
    def pause(self): self.player.pause(); self.audio_player.pause(); self.cue_player.pause()
    
    def seek(self, pos): 
        self.player.setPosition(pos)
        a_pos = int(pos / self.playback_rate) if (self.audio_player.playbackRate() == 1.0 and self.playback_rate != 1.0) else pos
        self.audio_player.setPosition(a_pos)
        if self.cue_active: self.cue_player.setPosition(a_pos)
    def position(self): return self.player.position()
    def duration(self): return self.player.duration()
    def playbackState(self): return self.player.playbackState()
    
    def setPlaybackRate(self, rate): 
        self.playback_rate = rate
        self.player.setPlaybackRate(rate)
        if self.base_wav_path and self.audio_player.playbackRate() == 1.0:
             self.swap_audio(self.base_wav_path, reset_rate_to_video=True)
        self.audio_player.setPlaybackRate(rate)
        self.cue_player.setPlaybackRate(rate)
        
    def set_main_output(self, device): self.main_output.setDevice(device)
    def set_cue_output(self, device): self.cue_output.setDevice(device)

class InteractiveWaveform(QLabel):
    def __init__(self, key_char, color, parent_app):
        super().__init__()
        self.key_char, self.parent_app = key_char, parent_app
        self.setAcceptDrops(True); self.setMouseTracking(True)
        self.base_color = QColor(color)
        self.setFixedSize(200, 120)
        self.setStyleSheet(f"border: 2px solid {color}; border-radius: 8px; background-color: #222;")
        self.filename = "[Empty]"; self.bpm_text = ""; self.waveform_pixmap = None
        self.playhead_x = 0; self.is_deck_a = False; self.is_deck_b = False
        self.loading = False; self.hotcues = {}; self.track_duration = 0

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.waveform_pixmap: painter.drawPixmap(0, 0, self.waveform_pixmap)
        if self.is_deck_a:
            painter.setPen(QPen(QColor("#FF0055"), 4)); painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(10, 20, "DECK A")
        elif self.is_deck_b:
            painter.setPen(QPen(QColor("#00CCFF"), 4)); painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(self.width()-60, 20, "DECK B")
        painter.setPen(QColor("white")); font = painter.font(); font.setBold(True); painter.setFont(font)
        status = " (...)" if self.loading else ""
        label = f"KEY: {self.key_char.upper()}\n{self.filename}{status}\n{self.bpm_text}"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, label); painter.end()

    def mousePressEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier: self.parent_app.assign_to_deck("B", self.key_char)
        elif event.button() == Qt.MouseButton.LeftButton: self.parent_app.assign_to_deck("A", self.key_char)
        elif event.button() == Qt.MouseButton.RightButton: self.parent_app.assign_to_deck("B", self.key_char)
    def dragEnterEvent(self, event): event.accept() if event.mimeData().hasUrls() else event.ignore()
    def dropEvent(self, event): self.parent_app.assign_clip_to_bank(self.key_char, [u.toLocalFile() for u in event.mimeData().urls()][0])
    def set_data(self, pixmap, bpm, duration): self.waveform_pixmap = pixmap; self.bpm_text = f"{bpm} BPM"; self.track_duration = duration; self.loading = False; self.update()
    def update_playhead(self, ratio): self.playhead_x = int(ratio * self.width()); self.update()
    def set_loading(self): self.loading = True; self.update()

# --- MAIN APP ---
class LooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VJ Piano Roll Sequencer v15.1 (Stable)")
        self.resize(800, 1000)
        QApplication.instance().setStyleSheet(DARK_THEME)

        self.projector = QWidget(); self.projector.resize(800,600); self.projector.setStyleSheet("background:black")
        self.proj_scene = QGraphicsScene(0, 0, 800, 600)
        self.proj_view = QGraphicsView(self.projector)
        self.proj_view.setViewport(QOpenGLWidget()); self.proj_view.resize(800,600)
        self.proj_view.setScene(self.proj_scene)
        self.proj_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.proj_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.deck_a_item = QGraphicsVideoItem(); self.deck_b_item = QGraphicsVideoItem()
        self.proj_scene.addItem(self.deck_a_item); self.proj_scene.addItem(self.deck_b_item)
        self.deck_b_item.setZValue(1)
        self.projector.show()

        self.deck_a = VJDeck("A", self.deck_a_item)
        self.deck_b = VJDeck("B", self.deck_b_item)
        self.deck_a.player.positionChanged.connect(self.on_deck_a_pos)
        self.deck_b.player.positionChanged.connect(self.on_deck_b_pos)

        self.buttons = {}; self.bank_data = {0: {}, 1: {}, 2: {}} 
        self.clip_meta = {}; self.hotcue_data = {}; self.clip_curves = {}
        self.active_clip_a = None; self.active_clip_b = None
        self.current_bank = 0; self.current_generation = 0; self.active_workers = []
        self.crossfader_value = 0.0; self.master_bpm = 120.0; self.tap_times = []
        # FIX: Init start time
        self.transport_start_time = time.time()
        
        self.seq_running = False; self.current_step = 0
        self.seq_multiplier = 1.0; self.seq_timer = QTimer(); self.seq_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.seq_timer.timeout.connect(self.run_sequencer_step)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); self.setCentralWidget(scroll)
        w = QWidget(); w.setObjectName("Container"); scroll.setWidget(w); l = QVBoxLayout(w); l.setSpacing(10)

        # Top Bar
        top = QHBoxLayout()
        btn_s = QPushButton("SAVE"); btn_s.clicked.connect(self.save_set)
        btn_l = QPushButton("LOAD"); btn_l.clicked.connect(self.load_set)
        
        self.btn_vid_sync = QPushButton("VID SYNC: ON"); self.btn_vid_sync.setCheckable(True); self.btn_vid_sync.setChecked(True); self.btn_vid_sync.setProperty("sync","true"); self.btn_vid_sync.clicked.connect(self.toggle_vid_sync)
        self.btn_align = QPushButton("AUTO-ALIGN"); self.btn_align.clicked.connect(self.auto_align_phase)
        top.addWidget(btn_s); top.addWidget(btn_l); top.addWidget(self.btn_vid_sync); top.addWidget(self.btn_align)
        l.addLayout(top)

        # Audio I/O
        io = QHBoxLayout()
        devs = QMediaDevices.audioOutputs()
        self.c_main = QComboBox(); self.c_cue = QComboBox()
        for d in devs: self.c_main.addItem(d.description()); self.c_cue.addItem(d.description())
        self.c_main.currentIndexChanged.connect(self.change_main_output)
        self.c_cue.currentIndexChanged.connect(self.change_cue_output)
        io.addWidget(QLabel("MAIN:")); io.addWidget(self.c_main); io.addWidget(QLabel("CUE:")); io.addWidget(self.c_cue)
        l.addLayout(io)

        # Bank
        bank_row = QHBoxLayout()
        self.bank_btns = []
        for i in range(3):
            b = QPushButton(f"BANK {i+1}"); b.setCheckable(True); b.clicked.connect(lambda _, x=i: self.switch_bank(x))
            bank_row.addWidget(b); self.bank_btns.append(b)
        self.bank_btns[0].setChecked(True); l.addLayout(bank_row)

        # Pads
        g = QGridLayout()
        for k, (r,c,col) in KEY_MAP.items():
            b = InteractiveWaveform(k, col, self); self.buttons[k] = b; g.addWidget(b, r, c)
        l.addLayout(g)

        # XFader
        l.addWidget(QLabel("CROSSFADER (Left/Right Arrows)"))
        self.fader = QSlider(Qt.Orientation.Horizontal); self.fader.setRange(0,100); self.fader.valueChanged.connect(self.on_fader_ui_changed); l.addWidget(self.fader)

        # BPM
        bpm_row = QHBoxLayout()
        self.bpm_lbl = QLabel("120.0 BPM")
        btn_nudge_down = QPushButton("-"); btn_nudge_down.setProperty("nudge", "true"); btn_nudge_down.clicked.connect(lambda: self.nudge_bpm(-0.1))
        btn_nudge_up = QPushButton("+"); btn_nudge_up.setProperty("nudge", "true"); btn_nudge_up.clicked.connect(lambda: self.nudge_bpm(0.1))
        self.c_speed = QComboBox(); self.c_speed.addItems(["1/2x", "1x", "2x"]); self.c_speed.setCurrentIndex(1)
        self.c_speed.currentIndexChanged.connect(self.change_seq_speed)
        btn_tap = QPushButton("TAP"); btn_tap.clicked.connect(self.handle_tap_tempo)
        bpm_row.addWidget(btn_nudge_down); bpm_row.addWidget(self.bpm_lbl); bpm_row.addWidget(btn_nudge_up)
        bpm_row.addWidget(QLabel("SEQ RATE:")); bpm_row.addWidget(self.c_speed); bpm_row.addWidget(btn_tap); l.addLayout(bpm_row)

        # LOOP BAR CONTROLS
        l.addWidget(QLabel("LOOP LENGTH:"))
        loop_size_row = QHBoxLayout()
        self.loop_btns = QButtonGroup()
        for s in [4, 8, 16, 32, 64]:
            b = QPushButton(str(s)); b.setCheckable(True); b.setProperty("loop", "true")
            if s == 64: b.setChecked(True)
            b.clicked.connect(lambda _, size=s: self.set_loop_length(size))
            self.loop_btns.addButton(b, s)
            loop_size_row.addWidget(b)
        loop_size_row.addStretch(); l.addLayout(loop_size_row)

        # PIANO ROLL
        cur_tools = QHBoxLayout()
        self.rad_a = QRadioButton("EDIT A"); self.rad_a.setChecked(True); self.rad_a.toggled.connect(self.update_curve_ui)
        self.rad_b = QRadioButton("EDIT B"); self.rad_b.toggled.connect(self.update_curve_ui)
        self.btn_run = QPushButton("RUN SEQ (P)"); self.btn_run.setCheckable(True); self.btn_run.clicked.connect(self.toggle_sequencer)
        cur_tools.addWidget(self.rad_a); cur_tools.addWidget(self.rad_b); cur_tools.addWidget(self.btn_run)
        l.addLayout(cur_tools)

        self.piano_roll = PianoRollSequencer(self)
        self.loop_bar = LoopBar(self.piano_roll)
        
        l.addWidget(self.loop_bar)
        l.addWidget(self.piano_roll)

        self.reopen_btn = QPushButton("OPEN PROJECTOR WINDOW"); self.reopen_btn.clicked.connect(self.projector.show); l.addWidget(self.reopen_btn)
        
        QApplication.instance().installEventFilter(self)
        self.update_mixer()

    def set_loop_length(self, length):
        self.piano_roll.set_loop_window(self.piano_roll.loop_start, length)

    def nudge_bpm(self, amount):
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: amount *= 10
        self.master_bpm = round(max(10.0, self.master_bpm + amount), 1)
        self.bpm_lbl.setText(f"{self.master_bpm} BPM")
        if self.btn_vid_sync.isChecked(): self.sync_deck(self.deck_a, self.active_clip_a); self.sync_deck(self.deck_b, self.active_clip_b)
        self.update_clock()

    def get_target_deck_info(self): return (self.deck_a, self.deck_a.current_filepath) if self.rad_a.isChecked() else (self.deck_b, self.deck_b.current_filepath)
    def update_curve_ui(self):
        _, path = self.get_target_deck_info()
        self.piano_roll.set_data(self.clip_curves.get(path, {}))
    def save_curve_data(self):
        _, path = self.get_target_deck_info()
        if path: self.clip_curves[path] = self.piano_roll.get_data()

    def run_sequencer_step(self):
        # Loop Logic
        ls = self.piano_roll.loop_start
        ll = self.piano_roll.loop_length
        self.current_step = ls + ((self.current_step + 1 - ls) % ll)
        
        self.piano_roll.current_step = self.current_step; self.piano_roll.update()
        
        path_a = self.deck_a.current_filepath
        if path_a and path_a in self.clip_curves and self.current_step in self.clip_curves[path_a]:
            val = self.clip_curves[path_a][self.current_step]; dur = self.deck_a.duration()
            if dur > 0: self.deck_a.trigger(int(val * dur))
        path_b = self.deck_b.current_filepath
        if path_b and path_b in self.clip_curves and self.current_step in self.clip_curves[path_b]:
            val = self.clip_curves[path_b][self.current_step]; dur = self.deck_b.duration()
            if dur > 0: self.deck_b.trigger(int(val * dur))

    def toggle_sequencer(self):
        self.seq_running = not self.seq_running
        self.btn_run.setChecked(self.seq_running)
        if self.seq_running: self.update_clock()
        else: self.seq_timer.stop()

    def update_clock(self):
        if self.master_bpm <= 0: return
        interval = ((60000.0 / self.master_bpm) / 4) / self.seq_multiplier
        self.seq_timer.setInterval(int(interval))
        if self.seq_running and not self.seq_timer.isActive(): self.seq_timer.start()

    def change_seq_speed(self, i):
        self.seq_multiplier = [0.5, 1.0, 2.0][i]
        self.update_clock()

    def handle_tap_tempo(self):
        now = time.time(); self.tap_times.append(now)
        if len(self.tap_times)>4: self.tap_times.pop(0)
        if len(self.tap_times)>1:
            avg = sum([self.tap_times[i+1]-self.tap_times[i] for i in range(len(self.tap_times)-1)]) / (len(self.tap_times)-1)
            self.master_bpm = round(60.0/avg, 1); self.bpm_lbl.setText(f"{self.master_bpm} BPM")
            if self.btn_vid_sync.isChecked(): self.sync_deck(self.deck_a, self.active_clip_a); self.sync_deck(self.deck_b, self.active_clip_b)
            self.update_clock()

    def sync_deck(self, deck, key):
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        cb = self.clip_meta.get(path, 120.0)
        rate = self.master_bpm / cb if cb > 0 else 1.0
        deck.setPlaybackRate(rate)
        if deck.base_wav_path:
            w = RubberBandWorker(key, deck.base_wav_path, rate)
            w.finished.connect(lambda k,p,r: deck.swap_audio(p,False)) 
            self.active_workers.append(w); w.start()

    def toggle_vid_sync(self):
        on = self.btn_vid_sync.isChecked()
        self.btn_vid_sync.setText(f"VID SYNC: {'ON' if on else 'OFF'}")
        if on: self.sync_deck(self.deck_a, self.active_clip_a); self.sync_deck(self.deck_b, self.active_clip_b)
        else: self.deck_a.setPlaybackRate(1.0); self.deck_b.setPlaybackRate(1.0)

    def assign_clip_to_bank(self, key, path):
        self.bank_data[self.current_bank][key] = path
        self.buttons[key].set_loading()
        w = AudioAnalysisWorker(key, path, 200, 120, self.buttons[key].base_color.name(), self.current_generation)
        w.finished.connect(self.prep_done); self.active_workers.append(w); w.start()

    def prep_done(self, key, pix, bpm, dur, raw, rate, wav):
        path = self.bank_data[self.current_bank].get(key)
        if path:
            self.clip_meta[path] = bpm
            if self.active_clip_a == key: self.deck_a.set_audio_data(raw, rate); self.deck_a.load_base_audio(wav)
            if self.active_clip_b == key: self.deck_b.set_audio_data(raw, rate); self.deck_b.load_base_audio(wav)
        self.buttons[key].set_data(pix, bpm, dur)

    def assign_to_deck(self, deck_name, key):
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        t = self.deck_a if deck_name == "A" else self.deck_b
        t.load_video(path)
        if deck_name == "A": self.active_clip_a = key
        else: self.active_clip_b = key
        
        self.buttons[key].set_loading()
        w = AudioAnalysisWorker(key, path, 200, 120, self.buttons[key].base_color.name(), self.current_generation)
        w.finished.connect(self.prep_done); self.active_workers.append(w); w.start()
        
        # --- FIXED SCENE INIT ---
        sz = self.proj_scene.sceneRect().size()
        if sz.width() == 0: sz = QSizeF(800, 600)
        t.video_item.setSize(sz)
        
        t.video_item.show()
        
        for k, b in self.buttons.items():
            if deck_name == "A": b.is_deck_a = (k==key)
            else: b.is_deck_b = (k==key); b.update()
        
        t.play(); self.update_mixer(); self.update_curve_ui()

    def auto_align_phase(self):
        # Crash Fix: Ensure media loaded
        if not self.deck_a.has_media() or not self.deck_b.has_media(): return
        
        # FIX: Check for 0 BPM
        if self.master_bpm <= 0: return

        beat_ms = 60000.0/self.master_bpm
        phase = (time.time() - self.transport_start_time)*1000 % beat_ms
        for d in [self.deck_a, self.deck_b]:
            diff = phase - (d.position() % beat_ms)
            if abs(diff) > beat_ms/2: diff += beat_ms if diff < 0 else -beat_ms
            d.seek(max(0, int(d.position() + diff)))

    def on_fader_ui_changed(self, v): self.crossfader_value = v/100.0; self.update_mixer()
    def update_mixer(self):
        v = self.crossfader_value
        self.deck_a.set_volume(1.0-v); self.deck_b.set_volume(v)
        self.deck_a.video_item.setOpacity(1.0-v); self.deck_b.video_item.setOpacity(v)

    def change_main_output(self, i): d = self.audio_devices[i]; self.deck_a.set_main_output(d); self.deck_b.set_main_output(d)
    def change_cue_output(self, i): d = self.audio_devices[i]; self.deck_a.set_cue_output(d); self.deck_b.set_cue_output(d)
    
    def on_deck_a_pos(self, p): 
        if self.active_clip_a: self.buttons[self.active_clip_a].update_playhead(p/self.deck_a.duration())
    def on_deck_b_pos(self, p):
        if self.active_clip_b: self.buttons[self.active_clip_b].update_playhead(p/self.deck_b.duration())

    def switch_bank(self, i):
        self.current_bank = i; self.current_generation += 1
        for b in self.bank_btns: b.setChecked(False)
        self.bank_btns[i].setChecked(True)
        for k in KEY_MAP: 
            self.buttons[k].filename = "[Empty]"; self.buttons[k].update()
            p = self.bank_data[i].get(k)
            if p: self.assign_clip_to_bank(k, p)

    def load_set(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if f: 
            d = json.load(open(f, 'r'))
            self.bank_data = d['banks']
            # SANITIZE: Convert string keys back to integers
            raw_curves = d.get('curves', {})
            self.clip_curves = {}
            for path, points in raw_curves.items():
                self.clip_curves[path] = {int(k): v for k, v in points.items()}
            self.switch_bank(0)

    def save_set(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if f: json.dump({'banks':self.bank_data, 'curves':self.clip_curves}, open(f,'w'))

    def eventFilter(self, src, e):
        if e.type() == QEvent.Type.KeyPress and not e.isAutoRepeat():
            k = e.key()
            if k == Qt.Key.Key_Space: 
                if self.deck_a.has_media(): self.deck_a.play() if self.deck_a.playbackState()!=QMediaPlayer.PlaybackState.PlayingState else self.deck_a.pause()
                if self.deck_b.has_media(): self.deck_b.play() if self.deck_b.playbackState()!=QMediaPlayer.PlaybackState.PlayingState else self.deck_b.pause()
                return True
            if e.text() == 'p': self.toggle_sequencer(); return True
            if k == Qt.Key.Key_Return: self.handle_tap_tempo(); return True
            if k == Qt.Key.Key_Left: self.fader.setValue(max(0, self.fader.value()-5)); return True
            if k == Qt.Key.Key_Right: self.fader.setValue(min(100, self.fader.value()+5)); return True
            if e.text() in ['5','6','7']: self.switch_bank(int(e.text())-5); return True
        return super().eventFilter(src, e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LooperApp()
    window.show()
    sys.exit(app.exec())