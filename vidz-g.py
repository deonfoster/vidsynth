import sys
import os
import time
import json
import numpy as np
import gc
import librosa
from pydub import AudioSegment

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QVBoxLayout, QPushButton, QSlider,
                             QFileDialog, QHBoxLayout, QProgressBar,
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem,
                             QFrame, QComboBox)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QCursor, QFont

# --- PRO STYLING ---
DARK_THEME = """
QMainWindow { background-color: #121212; }
QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QPushButton { 
    background-color: #2a2a2a; color: #ccc; border: 1px solid #444; 
    padding: 8px; border-radius: 4px; font-weight: bold; min-width: 60px;
}
QPushButton:hover { background-color: #3a3a3a; border: 1px solid #666; color: white; }
QPushButton:pressed { background-color: #00CCFF; color: black; }
QPushButton:checked { background-color: #00FF66; color: black; border: 1px solid #00FF66; }

QComboBox {
    background-color: #2a2a2a; color: #ccc; border: 1px solid #444;
    padding: 5px; border-radius: 4px; min-width: 150px;
}
QComboBox::drop-down { border: none; }

QSlider::groove:horizontal {
    border: 1px solid #333; height: 6px; background: #1a1a1a; margin: 2px 0; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00CCFF; border: 1px solid #00CCFF; width: 24px; height: 24px; margin: -9px 0; border-radius: 12px;
}
"""

KEY_MAP = {
    'a': (0, 0, "#FF0055"), 
    's': (0, 1, "#00CCFF"), 
    'd': (1, 0, "#00FF66"), 
    'f': (1, 1, "#FFAA00"), 
}

# --- WORKER ---
class AudioAnalysisWorker(QThread):
    finished = pyqtSignal(str, QPixmap, float, int, int) 
    
    def __init__(self, key, filepath, width, height, color_hex, gen_id):
        super().__init__()
        self.key, self.filepath = key, filepath
        self.width, self.height = width, height
        self.bg_color = QColor(color_hex)
        self.gen_id = gen_id

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            audio = AudioSegment.from_file(self.filepath)
            duration_ms = len(audio)
            if len(audio) > 60000: audio = audio[:60000]
            audio = audio.set_channels(1).set_frame_rate(22050)
            samples = np.array(audio.get_array_of_samples())
            samples_float = samples.astype(np.float32) / 32768.0
            
            tempo, _ = librosa.beat.beat_track(y=samples_float, sr=22050)
            if isinstance(tempo, np.ndarray): tempo = tempo.item()
            bpm = float(round(tempo, 2))

            vis_samples = samples[::100]
            max_val = np.max(np.abs(vis_samples)) or 1
            vis_samples = vis_samples / max_val
            
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.bg_color.darker(150), 1))
            
            center_y = self.height / 2
            step = len(vis_samples) / self.width
            for x in range(self.width):
                if self.isInterruptionRequested(): return
                idx = int(x * step)
                if idx < len(vis_samples):
                    h = abs(vis_samples[idx]) * (self.height * 0.9)
                    painter.drawLine(x, int(center_y - h/2), x, int(center_y + h/2))
            painter.end()

            if not self.isInterruptionRequested():
                self.finished.emit(self.key, pixmap, bpm, duration_ms, self.gen_id)
            del audio, samples, samples_float, vis_samples
            gc.collect()
        except:
            if not self.isInterruptionRequested():
                self.finished.emit(self.key, QPixmap(), 120.0, 0, self.gen_id)

# --- DECK ---
class VJDeck:
    def __init__(self, name, video_item):
        self.name = name
        self.video_item = video_item
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_item)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.base_volume = 1.0

    def load(self, filepath):
        self.player.setSource(QUrl.fromLocalFile(filepath))
        self.stop()

    def play(self): self.player.play()
    def stop(self): self.player.stop()
    def seek(self, pos): self.player.setPosition(pos)
    def set_volume(self, vol):
        self.base_volume = vol
        self.audio.setVolume(vol)
    def position(self): return self.player.position()
    def duration(self): return self.player.duration()
    def has_media(self): return self.player.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia
    def playbackState(self): return self.player.playbackState()
    def setPlaybackRate(self, rate): self.player.setPlaybackRate(rate)
    
    # NEW: Audio Device Setter
    def set_audio_device(self, device):
        self.audio.setDevice(device)

# --- BUTTON ---
class InteractiveWaveform(QLabel):
    def __init__(self, key_char, color, parent_app):
        super().__init__()
        self.key_char = key_char
        self.parent_app = parent_app
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.base_color = QColor(color)
        self.setFixedSize(200, 120)
        self.setStyleSheet(f"border: 2px solid {color}; border-radius: 8px; background-color: #222;")
        self.filename = "[Empty]"
        self.bpm_text = ""
        self.waveform_pixmap = None
        self.playhead_x = 0
        self.is_deck_a = False
        self.is_deck_b = False
        self.loading = False
        self.hotcues = {} 
        self.track_duration = 0
        
        self.is_selecting = False
        self.selection_start = 0
        self.selection_end = 0
        self.has_active_loop = False
        self.mode = "NONE"
        self.selected_edge = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.waveform_pixmap: painter.drawPixmap(0, 0, self.waveform_pixmap)
        
        if self.has_active_loop or self.mode == "DRAWING":
            x1 = min(self.selection_start, self.selection_end)
            x2 = max(self.selection_start, self.selection_end)
            w = x2 - x1
            painter.fillRect(QRectF(x1, 0, w, self.height()), QColor(0, 255, 255, 40)) 
            
            start_color = QColor(255, 255, 0) if self.selected_edge == 'start' else QColor(0, 255, 255)
            painter.setPen(QPen(start_color, 2))
            painter.drawLine(int(x1), 0, int(x1), self.height())
            
            end_color = QColor(255, 255, 0) if self.selected_edge == 'end' else QColor(0, 255, 255)
            painter.setPen(QPen(end_color, 2))
            painter.drawLine(int(x2), 0, int(x2), self.height())

        if self.is_deck_a:
            painter.setPen(QPen(QColor("#FF0055"), 4))
            painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(10, 20, "DECK A")
        elif self.is_deck_b:
            painter.setPen(QPen(QColor("#00CCFF"), 4))
            painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(self.width()-60, 20, "DECK B")

        if (self.is_deck_a or self.is_deck_b) and self.filename != "[Empty]":
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.drawLine(int(self.playhead_x), 0, int(self.playhead_x), self.height())
            
            cue_colors = {1: QColor("#FF0000"), 2: QColor("#00FF00"), 3: QColor("#0000FF")}
            if self.track_duration > 0:
                for num, pos_ms in self.hotcues.items():
                    cx = int((pos_ms / self.track_duration) * self.width())
                    col = cue_colors.get(num, QColor("white"))
                    painter.setPen(QPen(col, 2))
                    painter.drawLine(cx, 15, cx, self.height())
                    painter.setBrush(col)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(cx, 5, 12, 12)
                    painter.setPen(QColor("black"))
                    painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                    painter.drawText(cx+3, 15, str(num))

        painter.setPen(QColor("white"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        status = " (...)" if self.loading else ""
        label = f"KEY: {self.key_char.upper()}\n{self.filename}{status}"
        if self.bpm_text: label += f"\n{self.bpm_text}"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, label)
        painter.end()

    def mousePressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                x = event.pos().x()
                margin = 10
                if self.has_active_loop:
                    x1 = min(self.selection_start, self.selection_end)
                    x2 = max(self.selection_start, self.selection_end)
                    if abs(x - x1) < margin:
                        self.selected_edge = 'start'
                        self.parent_app.notify_selection(self.key_char, 'start')
                        self.update()
                        return
                    elif abs(x - x2) < margin:
                        self.selected_edge = 'end'
                        self.parent_app.notify_selection(self.key_char, 'end')
                        self.update()
                        return
                self.selected_edge = None
                self.is_selecting = True
                self.selection_start = x
                self.selection_end = x
                self.has_active_loop = False
                self.mode = "DRAWING"
                self.update()
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            self.parent_app.assign_to_deck("B", self.key_char)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.parent_app.assign_to_deck("A", self.key_char)
        elif event.button() == Qt.MouseButton.RightButton:
            self.parent_app.assign_to_deck("B", self.key_char)

    def mouseMoveEvent(self, event):
        if self.mode == "DRAWING":
            self.selection_end = max(0, min(event.pos().x(), self.width()))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == "DRAWING":
            self.mode = "NONE"
            start, end = sorted((self.selection_start, self.selection_end))
            if (end - start) < 5: self.clear_loop()
            else:
                self.has_active_loop = True
                self.parent_app.set_manual_loop(self.key_char, start/self.width(), end/self.width())
            self.update()

    def clear_loop(self):
        self.has_active_loop = False
        self.selected_edge = None
        self.parent_app.clear_manual_loop(self.key_char)
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files: self.parent_app.assign_clip_to_bank(self.key_char, files[0])

    def set_data(self, pixmap, bpm, duration):
        self.waveform_pixmap = pixmap
        self.bpm_text = f"{bpm} BPM"
        self.track_duration = duration
        self.loading = False
        self.update()

    def set_loading(self):
        self.loading = True
        self.update()

    def update_playhead(self, ratio):
        self.playhead_x = int(ratio * self.width())
        self.update()

class ProjectorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projector Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: black;")
        self.view = QGraphicsView(self)
        self.view.setStyleSheet("background: black; border: none;")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.scene.setBackgroundBrush(Qt.GlobalColor.black)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.overlay_item = QGraphicsRectItem()
        self.overlay_item.setBrush(QColor(0, 0, 0, 0))
        self.overlay_item.setPen(QPen(Qt.PenStyle.NoPen))
        self.overlay_item.setZValue(9999) 
        self.scene.addItem(self.overlay_item)

    def resizeEvent(self, event):
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        self.view.resize(self.width(), self.height())
        self.overlay_item.setRect(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def apply_effect(self, effect_type):
        if effect_type == "INVERT": self.overlay_item.setBrush(QColor(255, 255, 255, 220))
        elif effect_type == "RED": self.overlay_item.setBrush(QColor(255, 0, 0, 100))
        elif effect_type == "BLUR": self.overlay_item.setBrush(QColor(0, 0, 0, 180))

    def clear_effects(self):
        self.overlay_item.setBrush(QColor(0, 0, 0, 0))

class LooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VJ Looper v50 (Audio Output)")
        self.resize(600, 950)
        QApplication.instance().setStyleSheet(DARK_THEME)

        self.projector = ProjectorWindow()
        self.deck_a = VJDeck("A", QGraphicsVideoItem())
        self.deck_b = VJDeck("B", QGraphicsVideoItem())
        self.projector.scene.addItem(self.deck_a.video_item)
        self.projector.scene.addItem(self.deck_b.video_item)
        self.deck_a.video_item.setZValue(0)
        self.deck_b.video_item.setZValue(1)
        self.projector.show()

        self.deck_a.player.positionChanged.connect(self.on_deck_a_pos)
        self.deck_b.player.positionChanged.connect(self.on_deck_b_pos)

        self.buttons = {} 
        self.bank_data = {0: {}, 1: {}, 2: {}} 
        self.clip_meta = {} 
        self.hotcue_data = {} 
        self.manual_loops = {}
        
        self.active_clip_a = None
        self.active_clip_b = None
        self.current_bank = 0
        self.current_generation = 0 
        self.active_workers = []
        
        self.crossfader_value = 0.0 
        self.active_effect = None
        self.current_loop_speed = 500
        self.is_stuttering = False
        self.stutter_cue = 0
        self.master_bpm = 120.0
        self.tap_times = []
        self.quantize_active = True
        
        self.active_selection_key = None
        self.active_selection_edge = None
        
        self.stutter_timer = QTimer()
        self.stutter_timer.timeout.connect(self.perform_stutter_loop)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("VJ MIXER DELUXE")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        top_row = QHBoxLayout()
        btn_save = QPushButton("SAVE")
        btn_save.clicked.connect(self.save_set)
        btn_load = QPushButton("LOAD")
        btn_load.clicked.connect(self.load_set)
        
        # Audio Selector
        self.audio_combo = QComboBox()
        self.audio_devices = QMediaDevices.audioOutputs()
        for device in self.audio_devices:
            self.audio_combo.addItem(device.description())
        self.audio_combo.currentIndexChanged.connect(self.change_audio_output)
        
        top_row.addWidget(btn_save)
        top_row.addWidget(btn_load)
        top_row.addWidget(self.audio_combo)
        main_layout.addLayout(top_row)
        
        bank_row = QHBoxLayout()
        self.bank_btns = []
        for i in range(3):
            b = QPushButton(f"BANK {i+1}")
            b.setCheckable(True)
            b.clicked.connect(lambda _, x=i: self.switch_bank(x))
            bank_row.addWidget(b)
            self.bank_btns.append(b)
        self.bank_btns[0].setChecked(True)
        main_layout.addLayout(bank_row)

        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)
        for key, (row, col, color) in KEY_MAP.items():
            btn = InteractiveWaveform(key, color, self)
            self.buttons[key] = btn
            grid_layout.addWidget(btn, row, col)

        lbl_loop = QLabel("STUTTER SIZE (Q-R)")
        lbl_loop.setStyleSheet("color: #888; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(lbl_loop)
        loop_row = QHBoxLayout()
        self.loop_btns = {}
        sizes = [("1/1 (Q)", 1000), ("1/2 (W)", 500), ("1/4 (E)", 250), ("1/8 (R)", 125)]
        for name, ms in sizes:
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(lambda _, m=ms, n=name: self.set_loop_speed(m, n))
            self.loop_btns[ms] = b
            loop_row.addWidget(b)
        self.loop_btns[500].setChecked(True) 
        main_layout.addLayout(loop_row)

        lbl_fx = QLabel("VISUAL FX (Z-X-C)")
        lbl_fx.setStyleSheet("color: #888; font-weight: bold; margin-top: 5px;")
        main_layout.addWidget(lbl_fx)
        fx_row = QHBoxLayout()
        self.fx_btns = {}
        effects = [("STROBE (Z)", "INVERT"), ("RED (X)", "RED"), ("BLUR (C)", "BLUR")]
        for label, code in effects:
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _, c=code: self.toggle_effect(c))
            self.fx_btns[code] = b
            fx_row.addWidget(b)
        main_layout.addLayout(fx_row)

        xfader_lbl = QLabel("CROSSFADER (Shift+Arrow = Beatjump)")
        xfader_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(xfader_lbl)
        self.fader_slider = QSlider(Qt.Orientation.Horizontal)
        self.fader_slider.setRange(0, 100)
        self.fader_slider.setValue(0)
        self.fader_slider.valueChanged.connect(self.on_fader_ui_changed)
        main_layout.addWidget(self.fader_slider)

        bpm_row = QHBoxLayout()
        self.bpm_label = QLabel("MASTER BPM: 120.0")
        self.bpm_label.setStyleSheet("color: #00FF66; font-size: 14px; font-weight: bold;")
        btn_tap = QPushButton("TAP (Ret)")
        btn_tap.setFixedSize(80, 30)
        btn_tap.setStyleSheet("background: #444; border: 1px solid #666;")
        btn_tap.clicked.connect(self.handle_tap_tempo)
        bpm_row.addStretch()
        bpm_row.addWidget(self.bpm_label)
        bpm_row.addWidget(btn_tap)
        bpm_row.addStretch()
        main_layout.addLayout(bpm_row)
        
        self.status_label = QLabel("Hotcues: 1/2/3 | Shift+Click Edge + '-/=' to Nudge Loop")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.reopen_btn = QPushButton("OPEN PROJECTOR WINDOW")
        self.reopen_btn.clicked.connect(self.projector.show)
        main_layout.addWidget(self.reopen_btn)

        QApplication.instance().installEventFilter(self)
        self.update_mixer()

    def change_audio_output(self, index):
        if 0 <= index < len(self.audio_devices):
            device = self.audio_devices[index]
            self.deck_a.set_audio_device(device)
            self.deck_b.set_audio_device(device)
            self.status_label.setText(f"Audio Output: {device.description()}")

    def notify_selection(self, key, edge):
        for k, b in self.buttons.items():
            if k != key:
                b.selected_edge = None
                b.update()
        self.active_selection_key = key
        self.active_selection_edge = edge
        self.status_label.setText(f"Selected {edge.upper()} edge of Clip {key.upper()}")

    def nudge_loop_selection(self, amount_ms):
        key = self.active_selection_key
        edge = self.active_selection_edge
        if key and edge:
            path = self.bank_data[self.current_bank].get(key)
            if path and path in self.manual_loops:
                loop = self.manual_loops[path]
                if loop['active']:
                    if edge == 'start': loop['start'] = max(0, loop['start'] + amount_ms)
                    elif edge == 'end': loop['end'] = max(0, loop['end'] + amount_ms)
                    self._update_loop_visuals(key, loop)
                    self.status_label.setText(f"Nudged {edge} {amount_ms:+d}ms")

    def halve_loop(self):
        deck, key = self.get_dominant_deck()
        if key: self._modify_loop_len(key, 0.5)

    def double_loop(self):
        deck, key = self.get_dominant_deck()
        if key: self._modify_loop_len(key, 2.0)

    def move_loop(self, direction):
        deck, key = self.get_dominant_deck()
        if key:
            path = self.bank_data[self.current_bank].get(key)
            if path and path in self.manual_loops:
                loop = self.manual_loops[path]
                if loop['active']:
                    bpm = 120.0 # Simplification
                    beat_ms = 60000 / bpm
                    move_ms = int(beat_ms * direction)
                    loop['start'] = max(0, loop['start'] + move_ms)
                    loop['end'] = max(0, loop['end'] + move_ms)
                    self._update_loop_visuals(key, loop)
                    self.status_label.setText(f"Moved Loop {'Right' if direction>0 else 'Left'}")

    def snap_loop_to_grid(self):
        deck, key = self.get_dominant_deck()
        if key:
            path = self.bank_data[self.current_bank].get(key)
            if path and path in self.manual_loops:
                loop = self.manual_loops[path]
                if loop['active']:
                    bpm = 120.0
                    beat_ms = 60000 / bpm
                    loop['start'] = int(round(loop['start'] / beat_ms) * beat_ms)
                    loop['end'] = int(round(loop['end'] / beat_ms) * beat_ms)
                    self._update_loop_visuals(key, loop)
                    self.status_label.setText("Snapped Loop to Beat Grid")

    def _modify_loop_len(self, key, factor):
        path = self.bank_data[self.current_bank].get(key)
        if path and path in self.manual_loops:
            loop = self.manual_loops[path]
            if loop['active']:
                length = loop['end'] - loop['start']
                new_len = length * factor
                loop['end'] = int(loop['start'] + new_len)
                self._update_loop_visuals(key, loop)
                self.status_label.setText(f"Loop x{factor}")

    def _update_loop_visuals(self, key, loop):
        btn = self.buttons[key]
        if btn.track_duration > 0:
            btn.selection_start = (loop['start'] / btn.track_duration) * btn.width()
            btn.selection_end = (loop['end'] / btn.track_duration) * btn.width()
            btn.update()

    def handle_tap_tempo(self):
        now = time.time()
        if self.tap_times and (now - self.tap_times[-1] > 2.0): self.tap_times = []
        self.tap_times.append(now)
        if len(self.tap_times) > 4: self.tap_times.pop(0)
        if len(self.tap_times) > 1:
            intervals = [self.tap_times[i+1] - self.tap_times[i] for i in range(len(self.tap_times)-1)]
            avg = sum(intervals) / len(intervals)
            if avg > 0:
                self.master_bpm = round(60.0 / avg, 1)
                self.bpm_label.setText(f"MASTER BPM: {self.master_bpm} (TAP)")
                self.sync_deck_speed(self.deck_a, self.active_clip_a)
                self.sync_deck_speed(self.deck_b, self.active_clip_b)

    def sync_deck_speed(self, deck, key):
        if not key: return
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        clip_bpm = self.clip_meta.get(path, 120.0)
        sync_rate = 1.0
        if clip_bpm > 0 and self.master_bpm > 0:
            sync_rate = self.master_bpm / clip_bpm
        deck.setPlaybackRate(sync_rate)

    def nudge_deck(self, amount_ms):
        deck, _ = self.get_dominant_deck()
        if deck and deck.has_media():
            new_pos = max(0, deck.position() + amount_ms)
            deck.seek(new_pos)
            self.status_label.setText(f"Nudged Playback {amount_ms}ms")

    def assign_clip_to_bank(self, key, filepath):
        self.bank_data[self.current_bank][key] = filepath
        self.start_processing(key, filepath)

    def start_processing(self, key, filepath):
        self.buttons[key].set_loading()
        color = self.buttons[key].base_color.name()
        worker = AudioAnalysisWorker(key, filepath, 200, 120, color, self.current_generation)
        worker.finished.connect(self.on_prep_done)
        self.active_workers.append(worker)
        worker.start()

    def on_prep_done(self, key, pixmap, bpm, duration, gen_id):
        path = self.bank_data[self.current_bank].get(key)
        if path: self.clip_meta[path] = bpm
        if key in self.buttons: self.buttons[key].set_data(pixmap, bpm, duration)

    def assign_to_deck(self, deck, key):
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        if path not in self.hotcue_data: self.hotcue_data[path] = {}
        self.buttons[key].hotcues = self.hotcue_data[path]
        target_deck = self.deck_a if deck == "A" else self.deck_b
        target_deck.load(path)
        if deck == "A": self.active_clip_a = key
        else: self.active_clip_b = key
        self.sync_deck_speed(target_deck, key)
        start_pos = 0
        if self.quantize_active and self.master_bpm > 0:
            other_deck = self.deck_b if deck == "A" else self.deck_a
            if other_deck and other_deck.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                beat_ms = 60000 / self.master_bpm
                pos_ms = other_deck.position()
                offset_ms = pos_ms % beat_ms
                start_pos = int(offset_ms)
        target_deck.seek(start_pos)
        target_deck.play()
        target_deck.video_item.show()
        target_deck.video_item.setSize(self.projector.scene.sceneRect().size())
        for k, b in self.buttons.items():
            if deck == "A": b.is_deck_a = (k == key)
            else: b.is_deck_b = (k == key)
        for b in self.buttons.values(): b.update()
        self.update_mixer()

    def switch_bank(self, index):
        self.current_bank = index
        self.current_generation += 1
        for i, btn in enumerate(self.bank_btns): btn.setChecked(i == index)
        current_data = self.bank_data[self.current_bank]
        for key in KEY_MAP.keys():
            self.buttons[key].is_deck_a = (key == self.active_clip_a)
            self.buttons[key].is_deck_b = (key == self.active_clip_b)
            self.buttons[key].has_active_loop = False
            if key in current_data:
                path = current_data[key]
                self.start_processing(key, path)
            else:
                self.buttons[key].filename = "[Empty]"
                self.buttons[key].bpm_text = ""
                self.buttons[key].waveform_pixmap = None
                self.buttons[key].update()

    def on_fader_ui_changed(self, value):
        self.crossfader_value = value / 100.0
        self.update_mixer()

    def update_mixer(self):
        val = self.crossfader_value
        self.deck_a.set_volume(1.0 - val)
        self.deck_b.set_volume(val)
        self.deck_a.video_item.setOpacity(1.0 - val)
        self.deck_b.video_item.setOpacity(val)

    def on_deck_a_pos(self, pos):
        if self.active_clip_a and self.active_clip_a in self.buttons:
            dur = self.deck_a.duration()
            if dur > 0: self.buttons[self.active_clip_a].update_playhead(pos/dur)
        path = self.bank_data[self.current_bank].get(self.active_clip_a)
        if path and path in self.manual_loops and not self.is_stuttering:
            loop = self.manual_loops[path]
            if loop['active'] and pos >= loop['end']: self.deck_a.seek(loop['start'])

    def on_deck_b_pos(self, pos):
        if self.active_clip_b and self.active_clip_b in self.buttons:
            dur = self.deck_b.duration()
            if dur > 0: self.buttons[self.active_clip_b].update_playhead(pos/dur)
        path = self.bank_data[self.current_bank].get(self.active_clip_b)
        if path and path in self.manual_loops and not self.is_stuttering:
            loop = self.manual_loops[path]
            if loop['active'] and pos >= loop['end']: self.deck_b.seek(loop['start'])

    def set_manual_loop(self, key, start_ratio, end_ratio):
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        dur = 0
        if key == self.active_clip_a: dur = self.deck_a.duration()
        elif key == self.active_clip_b: dur = self.deck_b.duration()
        if dur > 0:
            start_ms = int(start_ratio * dur)
            end_ms = int(end_ratio * dur)
            self.manual_loops[path] = {'active': True, 'start': start_ms, 'end': end_ms}
            if key == self.active_clip_a: self.deck_a.seek(start_ms)
            if key == self.active_clip_b: self.deck_b.seek(start_ms)

    def clear_manual_loop(self, key):
        path = self.bank_data[self.current_bank].get(key)
        if path and path in self.manual_loops:
            self.manual_loops[path]['active'] = False

    def handle_hotcue(self, num, is_delete):
        deck, key = self.get_dominant_deck()
        path = self.bank_data[self.current_bank].get(key)
        if deck and path:
            if is_delete:
                if num in self.hotcue_data[path]: 
                    del self.hotcue_data[path][num]
                    self.status_label.setText(f"Deleted Hotcue {num}")
            else:
                if num in self.hotcue_data[path]: 
                    deck.seek(self.hotcue_data[path][num])
                else: 
                    self.hotcue_data[path][num] = deck.position()
                    self.status_label.setText(f"Set Hotcue {num}")
            self.buttons[key].update()

    def get_dominant_deck(self):
        if self.crossfader_value > 0.5: return self.deck_b, self.active_clip_b
        return self.deck_a, self.active_clip_a

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            key = event.key()
            mods = event.modifiers()
            
            # --- NEW LOOP PERFORMANCE CONTROLS ---
            if event.text() == ';': self.halve_loop(); return True
            if event.text() == "'": self.double_loop(); return True
            if event.text() == ',': self.move_loop(-1); return True
            if event.text() == '.': self.move_loop(1); return True
            if event.text() == 'm': self.snap_loop_to_grid(); return True

            # Loop Nudging
            if event.text() == '-': self.nudge_loop_selection(-10); return True
            if event.text() == '=': self.nudge_loop_selection(10); return True

            # Tap Tempo
            if key in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
                self.handle_tap_tempo(); return True

            # Hotcues
            if key == Qt.Key.Key_1: self.handle_hotcue(1, mods == Qt.KeyboardModifier.ShiftModifier); return True
            if key == Qt.Key.Key_Exclam: self.handle_hotcue(1, True); return True 
            if key == Qt.Key.Key_2: self.handle_hotcue(2, mods == Qt.KeyboardModifier.ShiftModifier); return True
            if key == Qt.Key.Key_At: self.handle_hotcue(2, True); return True
            if key == Qt.Key.Key_3: self.handle_hotcue(3, mods == Qt.KeyboardModifier.ShiftModifier); return True
            if key == Qt.Key.Key_NumberSign: self.handle_hotcue(3, True); return True

            # Playback Nudge
            if event.text() == '[': self.nudge_deck(-20); return True
            if event.text() == ']': self.nudge_deck(20); return True

            # Stutter Sizes
            if event.text() == 'q': self.set_loop_speed(1000, "1/1 (Q)")
            elif event.text() == 'w': self.set_loop_speed(500, "1/2 (W)")
            elif event.text() == 'e': self.set_loop_speed(250, "1/4 (E)")
            elif event.text() == 'r': self.set_loop_speed(125, "1/8 (R)")
            
            elif key == Qt.Key.Key_Space:
                deck, _ = self.get_dominant_deck()
                if deck and deck.has_media():
                    self.is_stuttering = True
                    self.stutter_cue = deck.position()
                    self.stutter_timer.start(self.current_loop_speed)
                    return True

            elif key == Qt.Key.Key_Left:
                if mods == Qt.KeyboardModifier.ShiftModifier: self.handle_beatjump(-4)
                else: self.fader_slider.setValue(max(0, self.fader_slider.value()-5))
            elif key == Qt.Key.Key_Right:
                if mods == Qt.KeyboardModifier.ShiftModifier: self.handle_beatjump(4)
                else: self.fader_slider.setValue(min(100, self.fader_slider.value()+5))

            elif event.text() == 'z': self.toggle_effect("INVERT")
            elif event.text() == 'x': self.toggle_effect("RED")
            elif event.text() == 'c': self.toggle_effect("BLUR")
            elif event.text() == '5': self.switch_bank(0)
            elif event.text() == '6': self.switch_bank(1)
            elif event.text() == '7': self.switch_bank(2)

            return True
            
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                self.is_stuttering = False
                self.stutter_timer.stop()
                deck, _ = self.get_dominant_deck()
                if deck and deck.has_media(): deck.seek(self.stutter_cue) 
                if deck: deck.play()
                return True

        return super().eventFilter(source, event)

    def set_loop_speed(self, ms, name):
        self.current_loop_speed = ms
        if self.stutter_timer.isActive(): self.stutter_timer.setInterval(ms)
        for m, btn in self.loop_btns.items(): btn.setChecked(m == ms)

    def perform_stutter_loop(self):
        deck, _ = self.get_dominant_deck()
        if deck and deck.has_media():
            deck.seek(self.stutter_cue)
            if deck.playbackState() != QMediaPlayer.PlaybackState.PlayingState: deck.play()

    def handle_beatjump(self, beats):
        deck, _ = self.get_dominant_deck()
        if deck and deck.has_media():
            bpm = self.master_bpm if self.master_bpm > 0 else 120.0
            ms = (60000 / bpm) * beats
            deck.seek(deck.position() + ms)

    def toggle_effect(self, effect_name):
        if self.active_effect == effect_name:
            self.active_effect = None
            self.projector.clear_effects()
            if effect_name in self.fx_btns: self.fx_btns[effect_name].setChecked(False)
        else:
            for k, b in self.fx_btns.items(): b.setChecked(False)
            self.active_effect = effect_name
            self.projector.apply_effect(effect_name)
            if effect_name in self.fx_btns: self.fx_btns[effect_name].setChecked(True)

    def save_set(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Set", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f: json.dump(self.bank_data, f)

    def load_set(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Set", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                raw_data = json.load(f)
                self.bank_data = {int(k): v for k, v in raw_data.items()}
            self.switch_bank(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LooperApp()
    window.show()
    sys.exit(app.exec())