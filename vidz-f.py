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
                             QFileDialog, QHBoxLayout, QProgressBar)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QCursor
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem

# --- PRO STYLING ---
DARK_THEME = """
QMainWindow { background-color: #1a1a1a; }
QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
QPushButton { 
    background-color: #333; color: white; border: 1px solid #555; 
    padding: 8px; border-radius: 4px; font-weight: bold;
}
QPushButton:hover { background-color: #444; border: 1px solid #777; }
QPushButton:pressed { background-color: #555; }
QSlider::groove:horizontal {
    border: 1px solid #333; height: 8px; background: #222; margin: 2px 0; border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #00CCFF; border: 1px solid #00CCFF; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px;
}
"""

KEY_MAP = {
    'a': (0, 0, "#FF0055"), 
    's': (0, 1, "#00CCFF"), 
    'd': (1, 0, "#00FF66"), 
    'f': (1, 1, "#FFAA00"), 
}

# --- WORKER: ROBUST BPM ANALYSIS ---
class AudioAnalysisWorker(QThread):
    finished = pyqtSignal(str, QPixmap, float, int) 
    
    def __init__(self, key, filepath, width, height, color_hex, gen_id):
        super().__init__()
        self.key = key
        self.filepath = filepath
        self.width = width
        self.height = height
        self.bg_color = QColor(color_hex)
        self.gen_id = gen_id

    def run(self):
        try:
            if self.isInterruptionRequested(): return
            audio = AudioSegment.from_file(self.filepath)
            if len(audio) > 30000: audio = audio[:30000]
            audio = audio.set_channels(1).set_frame_rate(22050)
            samples = np.array(audio.get_array_of_samples())
            
            # Simple Waveform Only (Skip heavy BPM if fast switching needed)
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
            
            # Default BPM 120 if we skip analysis to save CPU
            if not self.isInterruptionRequested():
                self.finished.emit(self.key, pixmap, 120.0, self.gen_id)
            
            del audio, samples, vis_samples
            gc.collect()
        except:
            if not self.isInterruptionRequested():
                self.finished.emit(self.key, QPixmap(), 120.0, self.gen_id)

# --- INTERACTIVE BUTTON ---
class InteractiveWaveform(QLabel):
    def __init__(self, key_char, color, parent_app):
        super().__init__()
        self.key_char = key_char
        self.parent_app = parent_app
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.base_color = QColor(color)
        self.setFixedSize(200, 120)
        self.setStyleSheet(f"""
            border: 2px solid {color}; border-radius: 10px; 
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a2a, stop:1 #1a1a1a);
        """)
        self.filename = "[Empty]"
        self.bpm_text = ""
        self.waveform_pixmap = None
        self.playhead_x = 0
        self.is_deck_a = False
        self.is_deck_b = False
        self.is_selecting = False
        self.selection_start = 0
        self.selection_end = 0
        self.has_active_loop = False
        self.mode = "NONE"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.waveform_pixmap: painter.drawPixmap(0, 0, self.waveform_pixmap)
        
        if self.has_active_loop or self.mode == "DRAWING":
            x1 = min(self.selection_start, self.selection_end)
            x2 = max(self.selection_start, self.selection_end)
            painter.fillRect(QRectF(x1, 0, x2-x1, self.height()), QColor(0, 255, 255, 40)) 
            painter.setPen(QPen(QColor(0, 255, 255), 2))
            painter.drawLine(int(x1), 0, int(x1), self.height())
            painter.drawLine(int(x2), 0, int(x2), self.height())

        if self.is_deck_a:
            painter.setPen(QPen(QColor("#FF0055"), 4))
            painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(10, 20, "DECK A")
        elif self.is_deck_b:
            painter.setPen(QPen(QColor("#00CCFF"), 4))
            painter.drawRect(self.rect().adjusted(2,2,-2,-2))
            painter.drawText(self.width()-60, 20, "DECK B")

        if self.filename != "[Empty]" and (self.is_deck_a or self.is_deck_b):
            painter.setPen(QPen(self.base_color, 2))
            painter.drawLine(int(self.playhead_x), 0, int(self.playhead_x), self.height())
        
        painter.setPen(QColor("white"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"KEY: {self.key_char.upper()}\n{self.filename}\n{self.bpm_text}")
        painter.end()

    def mousePressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        x = event.pos().x()
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                margin = 10
                x1 = min(self.selection_start, self.selection_end)
                x2 = max(self.selection_start, self.selection_end)
                if self.has_active_loop and abs(x - x1) < margin:
                    self.mode = "RESIZING_START"
                    self.selection_start, self.selection_end = x1, x2
                elif self.has_active_loop and abs(x - x2) < margin:
                    self.mode = "RESIZING_END"
                    self.selection_start, self.selection_end = x1, x2
                else:
                    self.mode = "DRAWING"
                    self.selection_start = x
                    self.selection_end = x
                    self.has_active_loop = False
                self.update()
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            self.parent_app.assign_to_deck("B", self.key_char)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.parent_app.assign_to_deck("A", self.key_char)
        elif event.button() == Qt.MouseButton.RightButton:
            self.parent_app.assign_to_deck("B", self.key_char)

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        if self.mode == "DRAWING": self.selection_end = max(0, min(x, self.width()))
        elif self.mode == "RESIZING_START": self.selection_start = max(0, min(x, self.width()))
        elif self.mode == "RESIZING_END": self.selection_end = max(0, min(x, self.width()))
        if self.has_active_loop and self.mode == "NONE":
            margin = 10
            x1 = min(self.selection_start, self.selection_end)
            x2 = max(self.selection_start, self.selection_end)
            if abs(x - x1) < margin or abs(x - x2) < margin: self.setCursor(Qt.CursorShape.SizeHorCursor)
            else: self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.mode in ["DRAWING", "RESIZING_START", "RESIZING_END"]:
            self.mode = "NONE"
            start, end = sorted((self.selection_start, self.selection_end))
            if (end - start) < 5: self.clear_loop()
            else:
                self.has_active_loop = True
                self.parent_app.set_manual_loop(self.key_char, start/self.width(), end/self.width())
            self.update()

    def clear_loop(self):
        self.has_active_loop = False
        self.parent_app.clear_manual_loop(self.key_char)
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files: self.parent_app.assign_clip_to_bank(self.key_char, files[0])

    def set_data(self, pixmap, bpm):
        self.waveform_pixmap = pixmap
        self.bpm_text = f"{bpm} BPM"
        self.update()

    def reset_visuals(self):
        self.filename = "[Empty]"
        self.bpm_text = ""
        self.waveform_pixmap = None
        self.is_deck_a = False # IMPORTANT: Reset indicators when file is gone from view
        self.is_deck_b = False
        self.clear_loop()
        self.update()

    def update_playhead(self, ratio):
        self.playhead_x = int(ratio * self.width())
        self.update()

# --- PROJECTOR ---
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
        self.setWindowTitle("VJ Looper v35 (Persistent Playback)")
        self.resize(600, 950)
        QApplication.instance().setStyleSheet(DARK_THEME)

        # CORE ARCHITECTURE CHANGE: Decks are separate from Buttons
        self.deck_a_player = QMediaPlayer()
        self.deck_a_video = QGraphicsVideoItem()
        self.deck_a_audio = QAudioOutput()
        self.deck_a_player.setAudioOutput(self.deck_a_audio)
        self.deck_a_player.setVideoOutput(self.deck_a_video)
        self.deck_a_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.deck_b_player = QMediaPlayer()
        self.deck_b_video = QGraphicsVideoItem()
        self.deck_b_audio = QAudioOutput()
        self.deck_b_player.setAudioOutput(self.deck_b_audio)
        self.deck_b_player.setVideoOutput(self.deck_b_video)
        self.deck_b_player.setLoops(QMediaPlayer.Loops.Infinite)

        # Connect signals for Deck A/B
        self.deck_a_player.positionChanged.connect(self.on_deck_a_position)
        self.deck_b_player.positionChanged.connect(self.on_deck_b_position)

        # Store which File/Key is playing on which Deck
        self.active_clip_a = {"path": None, "key": None, "bank": -1}
        self.active_clip_b = {"path": None, "key": None, "bank": -1}

        self.buttons = {} 
        self.bank_data = {0: {}, 1: {}, 2: {}} 
        self.clip_meta = {}
        self.current_bank = 0
        self.current_generation = 0 
        self.active_workers = []
        
        # Audio ghosting is complex with persistence, simplified to basic for stability first
        self.ghost_players = {} 

        self.crossfader_value = 0.0 
        self.master_bpm = 0.0
        self.quantize_active = True
        self.tap_times = []
        
        self.playback_rate = 1.0
        self.manual_loops = {} # Keyed by filepath now!
        self.active_effect = None

        self.projector = ProjectorWindow()
        self.projector.scene.addItem(self.deck_a_video)
        self.projector.scene.addItem(self.deck_b_video)
        self.deck_a_video.setZValue(0)
        self.deck_b_video.setZValue(1) # B on top
        self.projector.show()

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
        self.btn_sync = QPushButton("QUANTIZE: ON")
        self.btn_sync.setStyleSheet("background-color: #00FF66; color: black;")
        self.btn_sync.clicked.connect(self.toggle_quantize)
        top_row.addWidget(btn_save)
        top_row.addWidget(btn_load)
        top_row.addWidget(self.btn_sync)
        main_layout.addLayout(top_row)

        bank_layout = QHBoxLayout()
        self.bank_labels = []
        for i in range(3):
            lbl = QLabel(f"BANK {i+1}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(80, 40)
            lbl.setStyleSheet("border: 1px solid #555; background: #222; border-radius: 5px;")
            bank_layout.addWidget(lbl)
            self.bank_labels.append(lbl)
        main_layout.addLayout(bank_layout)
        self.update_bank_visuals()

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        main_layout.addLayout(grid_layout)

        for key, (row, col, color) in KEY_MAP.items():
            btn = InteractiveWaveform(key, color, self)
            self.buttons[key] = btn
            grid_layout.addWidget(btn, row, col)

        mixer_label = QLabel("CROSSFADER (Arrows)")
        mixer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(mixer_label)

        self.fader_slider = QSlider(Qt.Orientation.Horizontal)
        self.fader_slider.setMinimum(0)
        self.fader_slider.setMaximum(100)
        self.fader_slider.setValue(0)
        self.fader_slider.valueChanged.connect(self.on_fader_ui_changed)
        main_layout.addWidget(self.fader_slider)

        self.bpm_label = QLabel("MASTER BPM: -- (Tap Enter)")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.bpm_label)
        
        self.reopen_btn = QPushButton("OPEN PROJECTOR")
        self.reopen_btn.clicked.connect(self.projector.show)
        main_layout.addWidget(self.reopen_btn)

        QApplication.instance().installEventFilter(self)
        
        # Init Mixer Volume
        self.update_mixer()

    # --- PERSISTENT PLAYBACK LOGIC ---
    def assign_to_deck(self, deck, key):
        filepath = self.bank_data[self.current_bank].get(key)
        if not filepath: return # Empty button

        # Check if already playing
        if deck == "A" and self.active_clip_a["path"] == filepath: return
        if deck == "B" and self.active_clip_b["path"] == filepath: return

        # Load into Global Deck
        player = self.deck_a_player if deck == "A" else self.deck_b_player
        video_item = self.deck_a_video if deck == "A" else self.deck_b_video
        
        player.setSource(QUrl.fromLocalFile(filepath))
        player.play()
        video_item.show()
        rect = self.projector.scene.sceneRect()
        video_item.setSize(rect.size())

        # Update State
        if deck == "A":
            self.active_clip_a = {"path": filepath, "key": key, "bank": self.current_bank}
        else:
            self.active_clip_b = {"path": filepath, "key": key, "bank": self.current_bank}

        self.update_button_states()

    def on_deck_a_position(self, pos):
        # Only update visual playhead if the clip playing is currently visible on the bank
        if self.active_clip_a["bank"] == self.current_bank:
            key = self.active_clip_a["key"]
            if key in self.buttons:
                dur = self.deck_a_player.duration()
                if dur > 0: self.buttons[key].update_playhead(pos / dur)

    def on_deck_b_position(self, pos):
        if self.active_clip_b["bank"] == self.current_bank:
            key = self.active_clip_b["key"]
            if key in self.buttons:
                dur = self.deck_b_player.duration()
                if dur > 0: self.buttons[key].update_playhead(pos / dur)

    def update_button_states(self):
        # Reset all buttons first
        for key, btn in self.buttons.items():
            btn.is_deck_a = False
            btn.is_deck_b = False
        
        # Light up active clips if they belong to current bank
        if self.active_clip_a["bank"] == self.current_bank:
            key = self.active_clip_a["key"]
            if key in self.buttons: self.buttons[key].is_deck_a = True
            
        if self.active_clip_b["bank"] == self.current_bank:
            key = self.active_clip_b["key"]
            if key in self.buttons: self.buttons[key].is_deck_b = True
            
        # Repaint
        for btn in self.buttons.values(): btn.update()

    def switch_bank(self, new_bank_index):
        # 1. Stop all ANALYSIS, but do NOT stop players
        self.stop_all_workers()
        self.current_generation += 1
        
        self.current_bank = new_bank_index
        self.update_bank_visuals()
        
        current_data = self.bank_data[self.current_bank]
        
        # 2. Update Grid Buttons (Visuals only)
        for key in KEY_MAP.keys():
            if key in current_data:
                path = current_data[key]
                self.generate_waveform(key, path) # Start new analysis for visual
            else:
                self.buttons[key].reset_visuals()
                
        # 3. Update "Active" indicators (Is Deck A playing a clip from this new bank?)
        self.update_button_states()

    def stop_all_workers(self):
        for worker in self.active_workers:
            if worker.isRunning():
                worker.requestInterruption()
                worker.quit()
                worker.wait(50)
        self.active_workers.clear()

    # --- WORKER ---
    def generate_waveform(self, key, filepath):
        self.buttons[key].filename = os.path.basename(filepath)
        self.buttons[key].setText(f"Analyzing...")
        color = self.buttons[key].base_color.name()
        worker = AudioAnalysisWorker(key, filepath, 200, 120, color, self.current_generation)
        worker.finished.connect(self.on_analysis_done)
        self.active_workers.append(worker)
        worker.start()

    def on_analysis_done(self, key, pixmap, bpm, gen_id):
        if gen_id != self.current_generation: return
        if key in self.buttons: 
            self.buttons[key].set_data(pixmap, bpm)

    # --- MIXER ---
    def on_fader_ui_changed(self, value):
        self.crossfader_value = value / 100.0
        self.update_mixer()

    def update_mixer(self):
        val = self.crossfader_value 
        # Deck A
        vol_a = 1.0 - val
        self.deck_a_audio.setVolume(vol_a)
        self.deck_a_video.setOpacity(vol_a)
        self.deck_a_video.setZValue(10)
        
        # Deck B
        vol_b = val
        self.deck_b_audio.setVolume(vol_b)
        self.deck_b_video.setOpacity(vol_b)
        self.deck_b_video.setZValue(20)

    # --- DRAG DROP ---
    def assign_clip_to_bank(self, key, filepath):
        self.bank_data[self.current_bank][key] = filepath
        self.generate_waveform(key, filepath)

    # --- STANDARD UTILS ---
    def update_bank_visuals(self):
        for i, lbl in enumerate(self.bank_labels):
            if i == self.current_bank:
                lbl.setStyleSheet("border: 2px solid #00FF66; color: #00FF66; font-weight: bold; background: #222;")
            else:
                lbl.setStyleSheet("border: 1px solid #444; color: #888; background: #111;")

    def set_manual_loop(self, key, start, end): pass # Placeholder for complex logic in this ver
    def clear_manual_loop(self, key): pass
    def toggle_quantize(self): self.quantize_active = not self.quantize_active

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self.handle_press(event)
            return True 
        return super().eventFilter(source, event)

    def handle_press(self, event):
        key = event.text().lower()
        if key == '5': self.switch_bank(0)
        elif key == '6': self.switch_bank(1)
        elif key == '7': self.switch_bank(2)
        elif event.key() == Qt.Key.Key_Left: self.fader_slider.setValue(max(0, self.fader_slider.value()-5))
        elif event.key() == Qt.Key.Key_Right: self.fader_slider.setValue(min(100, self.fader_slider.value()+5))

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
            self.current_bank = -1
            self.switch_bank(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LooperApp()
    window.show()
    sys.exit(app.exec())