import sys
import os
import time
import json
import numpy as np
import gc # Garbage Collector
from pydub import AudioSegment

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QVBoxLayout, QPushButton,
                             QFileDialog, QHBoxLayout, QProgressBar)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap
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
QProgressBar { 
    border: 1px solid #444; background: #222; height: 8px; border-radius: 2px;
}
"""

KEY_MAP = {
    'a': (0, 0, "#FF0055"), # Neon Pink
    's': (0, 1, "#00CCFF"), # Neon Cyan
    'd': (1, 0, "#00FF66"), # Neon Green
    'f': (1, 1, "#FFAA00"), # Neon Orange
}

DEFAULT_LOOP_SPEED = 500

# --- BACKGROUND WORKER (Waveforms) ---
class WaveformWorker(QThread):
    finished = pyqtSignal(str, QPixmap) 
    def __init__(self, key, filepath, width, height, color_hex):
        super().__init__()
        self.key, self.filepath, self.width, self.height = key, filepath, width, height
        self.bg_color = QColor(color_hex)

    def run(self):
        try:
            # SAFETY: Only read the first 60 seconds to prevent RAM explosion on long files
            audio = AudioSegment.from_file(self.filepath)
            if len(audio) > 60000: 
                audio = audio[:60000] 

            audio = audio.set_channels(1).set_frame_rate(50)
            samples = np.array(audio.get_array_of_samples())
            
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen_color = self.bg_color.darker(150)
            painter.setPen(QPen(pen_color, 1))
            
            if len(samples) > 0:
                step = len(samples) / self.width
                center_y = self.height / 2
                max_val = np.max(np.abs(samples)) or 1
                for x in range(self.width):
                    chunk = samples[int(x*step):int((x+1)*step)]
                    if len(chunk) > 0:
                        h = (np.max(np.abs(chunk)) / max_val) * (self.height * 0.9)
                        painter.drawLine(x, int(center_y - h/2), x, int(center_y + h/2))
            painter.end()
            self.finished.emit(self.key, pixmap)
            
            # Help Python clear memory
            del audio
            del samples
            gc.collect()
            
        except Exception as e:
            print(f"Waveform Error: {e}")

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
            border: 2px solid {color}; 
            border-radius: 10px; 
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2a2a2a, stop:1 #1a1a1a);
        """)
        self.filename = "[Empty]"
        self.waveform_pixmap = None
        self.playhead_x = 0
        self.is_selecting = False
        self.selection_start = 0
        self.selection_end = 0
        self.has_active_loop = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.waveform_pixmap: painter.drawPixmap(0, 0, self.waveform_pixmap)
        if self.has_active_loop or self.is_selecting:
            x = min(self.selection_start, self.selection_end)
            w = abs(self.selection_end - self.selection_start)
            painter.fillRect(QRectF(x, 0, w, self.height()), QColor(0, 255, 255, 40)) 
            painter.setPen(QPen(QColor(0, 255, 255), 1))
            painter.drawRect(QRectF(x, 0, w, self.height()))
        if self.filename != "[Empty]":
            painter.setPen(QPen(self.base_color, 2))
            painter.drawLine(int(self.playhead_x), 0, int(self.playhead_x), self.height())
        painter.setPen(QColor("black"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        rect = self.rect()
        painter.drawText(rect.adjusted(1,1,1,1), Qt.AlignmentFlag.AlignCenter, f"KEY: {self.key_char.upper()}\n{self.filename}")
        painter.setPen(QColor("white"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"KEY: {self.key_char.upper()}\n{self.filename}")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = True
            self.selection_start = event.pos().x()
            self.selection_end = event.pos().x()
            self.has_active_loop = False
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_loop()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.selection_end = max(0, min(event.pos().x(), self.width()))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            start, end = sorted((self.selection_start, self.selection_end))
            if (end - start) > 5:
                self.has_active_loop = True
                self.parent_app.set_manual_loop(self.key_char, start/self.width(), end/self.width())
            else: self.clear_loop()
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

    def set_waveform(self, pixmap):
        self.waveform_pixmap = pixmap
        self.update()

    def update_playhead(self, ratio):
        self.playhead_x = int(ratio * self.width())
        self.update()


# --- GRAPHICS VIEW PROJECTOR ---
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
        self.overlay_item.setZValue(100) 
        self.scene.addItem(self.overlay_item)
        
    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.scene.setSceneRect(0, 0, w, h)
        self.view.resize(w, h)
        self.overlay_item.setRect(0, 0, w, h)
        super().resizeEvent(event)

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def apply_effect(self, effect_type):
        if effect_type == "INVERT": 
            self.overlay_item.setBrush(QColor(255, 255, 255, 220))
        elif effect_type == "RED": 
            self.overlay_item.setBrush(QColor(255, 0, 0, 100))
        elif effect_type == "BLUR": 
            self.overlay_item.setBrush(QColor(0, 0, 0, 180))

    def clear_effects(self):
        self.overlay_item.setBrush(QColor(0, 0, 0, 0)) 


class LooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VJ Looper v21 (Stability Fixed)")
        self.resize(600, 850)
        QApplication.instance().setStyleSheet(DARK_THEME)

        self.players = {}
        self.audio_outputs = {}
        self.video_items = {} 
        self.buttons = {} 
        self.progress_bars = {}
        self.bank_data = {0: {}, 1: {}, 2: {}} 
        self.current_bank = 0
        self.current_key = None
        self.drum_key_active = None
        self.cue_point = 0 
        self.slip_start_time = 0 
        self.is_stuttering = False
        self.current_loop_speed = DEFAULT_LOOP_SPEED
        self.playback_rate = 1.0
        self.active_workers = [] # Track threads
        self.manual_loops = {} 
        self.active_effect = None 

        self.stutter_timer = QTimer()
        self.stutter_timer.timeout.connect(self.perform_stutter_loop)

        self.projector = ProjectorWindow()
        self.projector.show()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("VJ LOOPER DELUXE")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff; letter-spacing: 2px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Menu
        top_row = QHBoxLayout()
        btn_save = QPushButton("SAVE SET")
        btn_save.clicked.connect(self.save_set)
        btn_load = QPushButton("LOAD SET")
        btn_load.clicked.connect(self.load_set)
        top_row.addWidget(btn_save)
        top_row.addWidget(btn_load)
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

        # Grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        main_layout.addLayout(grid_layout)

        for key, (row, col, color) in KEY_MAP.items():
            btn = InteractiveWaveform(key, color, self)
            self.buttons[key] = btn
            grid_layout.addWidget(btn, row, col)

            vid_item = QGraphicsVideoItem()
            self.projector.scene.addItem(vid_item)
            vid_item.hide()
            vid_item.setZValue(0) 
            self.video_items[key] = vid_item

            audio_out = QAudioOutput()
            audio_out.setVolume(1.0)
            self.audio_outputs[key] = audio_out

            player = QMediaPlayer()
            player.setAudioOutput(audio_out)
            player.setVideoOutput(vid_item)
            player.setLoops(QMediaPlayer.Loops.Infinite)
            
            player.positionChanged.connect(lambda pos, k=key: self.on_position_changed(k, pos))
            player.durationChanged.connect(lambda dur, k=key: self.on_duration_changed(k, dur))

            self.players[key] = player
            self.manual_loops[key] = {'active': False, 'start': 0, 'end': 0}

            pbar = QProgressBar()
            pbar.setTextVisible(False)
            self.progress_bars[key] = pbar

        instr = QLabel(
            "<b>CONTROLS:</b><br>"
            "Keys A/S/D/F = Play | Shift=Drum | Space=Loop | Tab=Slip<br>"
            "<b>EFFECTS:</b> Z=Whiteout | X=Red Tint | C=Dim"
        )
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr.setStyleSheet("color: #888;")
        main_layout.addWidget(instr)

        self.reopen_btn = QPushButton("OPEN PROJECTOR")
        self.reopen_btn.setStyleSheet("background-color: #444; font-size: 14px; padding: 15px;")
        self.reopen_btn.clicked.connect(self.projector.show)
        main_layout.addWidget(self.reopen_btn)

        QApplication.instance().installEventFilter(self)

    # --- LOGIC ---
    def toggle_effect(self, effect_name):
        if self.active_effect == effect_name:
            self.active_effect = None
            self.projector.clear_effects()
            self.update_status("Effects Off")
        else:
            self.active_effect = effect_name
            self.projector.apply_effect(effect_name)
            self.update_status(f"EFFECT: {effect_name}")

    def set_manual_loop(self, key, start_ratio, end_ratio):
        duration = self.players[key].duration()
        if duration > 0:
            self.manual_loops[key] = {'active': True, 'start': int(start_ratio*duration), 'end': int(end_ratio*duration)}
            if self.current_key == key: self.players[key].setPosition(self.manual_loops[key]['start'])

    def clear_manual_loop(self, key):
        self.manual_loops[key]['active'] = False

    def on_position_changed(self, key, position):
        duration = self.players[key].duration()
        if duration == 0: return
        self.buttons[key].update_playhead(position / duration)
        
        loop_data = self.manual_loops[key]
        if loop_data['active'] and not self.is_stuttering:
            if position >= loop_data['end']: self.players[key].setPosition(loop_data['start'])

    def on_duration_changed(self, key, duration):
        pass

    def generate_waveform(self, key, filepath):
        self.buttons[key].filename = os.path.basename(filepath)
        self.buttons[key].update()
        color = self.buttons[key].base_color.name()
        
        worker = WaveformWorker(key, filepath, 200, 120, color)
        
        # --- THE STABILITY FIX: CONNECT FINISHED TO CLEANUP ---
        worker.finished.connect(self.on_waveform_ready)
        worker.finished.connect(lambda: self.cleanup_worker(worker))
        
        self.active_workers.append(worker)
        worker.start()

    def cleanup_worker(self, worker):
        """ Removes thread from memory once done. """
        if worker in self.active_workers:
            self.active_workers.remove(worker)
        worker.deleteLater()

    def on_waveform_ready(self, key, pixmap):
        if key in self.buttons: self.buttons[key].set_waveform(pixmap)

    def assign_clip_to_bank(self, key, filepath):
        self.bank_data[self.current_bank][key] = filepath
        self.load_player(key, filepath)
        self.generate_waveform(key, filepath)

    def switch_bank(self, new_bank_index):
        if self.current_bank == new_bank_index: return
        if self.current_key:
            self.players[self.current_key].stop()
            self.video_items[self.current_key].hide()
            self.current_key = None
        self.current_bank = new_bank_index
        self.update_bank_visuals()
        current_data = self.bank_data[self.current_bank]
        for key in KEY_MAP.keys():
            self.manual_loops[key]['active'] = False
            self.buttons[key].clear_loop()
            if key in current_data:
                path = current_data[key]
                self.load_player(key, path)
                self.generate_waveform(key, path)
            else:
                self.players[key].setSource(QUrl())
                self.buttons[key].filename = "[Empty]"
                self.buttons[key].set_waveform(None)

    def load_player(self, key, filepath):
        self.players[key].setSource(QUrl.fromLocalFile(filepath))
        self.audio_outputs[key].setVolume(1.0)
        self.players[key].pause()
        self.players[key].setPosition(0)
        self.players[key].setPlaybackRate(self.playback_rate)

    def update_bank_visuals(self):
        for i, lbl in enumerate(self.bank_labels):
            if i == self.current_bank:
                lbl.setStyleSheet("border: 2px solid #00FF66; color: #00FF66; font-weight: bold; background: #222;")
            else:
                lbl.setStyleSheet("border: 1px solid #444; color: #888; background: #111;")

    def save_set(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Set", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f: json.dump(self.bank_data, f)
                self.update_status("Full Set Saved!")
            except Exception as e: self.update_status(f"Error: {e}")

    def load_set(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Set", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    raw_data = json.load(f)
                    self.bank_data = {int(k): v for k, v in raw_data.items()}
                saved_bank = self.current_bank
                self.current_bank = -1 
                self.switch_bank(saved_bank)
                self.update_status("Set Loaded!")
            except Exception as e: self.update_status(f"Error: {e}")

    def perform_stutter_loop(self):
        if self.current_key:
            self.players[self.current_key].setPosition(self.cue_point)
            if self.players[self.current_key].playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self.players[self.current_key].play()

    def set_loop_speed(self, ms, name):
        self.current_loop_speed = ms
        if self.stutter_timer.isActive(): self.stutter_timer.setInterval(ms)
        self.update_status(f"Loop Size: {name}")

    def change_playback_rate(self, delta):
        new_rate = max(0.1, min(4.0, round(self.playback_rate + delta, 2)))
        self.playback_rate = new_rate
        if self.current_key: self.players[self.current_key].setPlaybackRate(self.playback_rate)
        self.update_status(f"Speed: {self.playback_rate}x")

    def reset_playback_rate(self):
        self.playback_rate = 1.0
        if self.current_key: self.players[self.current_key].setPlaybackRate(1.0)
        self.update_status("Speed: Normal")

    def update_status(self, msg):
        self.setWindowTitle(f"VJ Looper v21 - {msg}")

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self.handle_press(event)
            return True 
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self.handle_release(event)
            return False 
        return super().eventFilter(source, event)

    def handle_press(self, event):
        key = event.text().lower()
        key_code = event.key()
        modifiers = QApplication.keyboardModifiers()

        if key == '5': self.switch_bank(0)
        elif key == '6': self.switch_bank(1)
        elif key == '7': self.switch_bank(2)

        elif key_code == Qt.Key.Key_Up: self.change_playback_rate(0.1) 
        elif key_code == Qt.Key.Key_Down: self.change_playback_rate(-0.1) 
        elif key_code == Qt.Key.Key_Right: self.reset_playback_rate() 

        elif key == '1': self.set_loop_speed(1000, "1.0s")
        elif key == '2': self.set_loop_speed(500, "0.5s")
        elif key == '3': self.set_loop_speed(250, "0.25s")
        elif key == '4': self.set_loop_speed(100, "0.1s")

        elif key == 'z': self.toggle_effect("INVERT")
        elif key == 'x': self.toggle_effect("RED")
        elif key == 'c': self.toggle_effect("BLUR")

        elif key in self.players:
            is_shift = modifiers == Qt.KeyboardModifier.ShiftModifier
            if is_shift:
                self.switch_to_key(key)
                self.drum_key_active = key
                self.players[key].setPosition(0)
                self.players[key].play()
                self.update_status(f"DRUM: {key.upper()}")
            elif not modifiers:
                self.switch_to_key(key)
                self.drum_key_active = None
                if self.manual_loops[key]['active']:
                    self.players[key].setPosition(self.manual_loops[key]['start'])
                else:
                    self.players[key].setPosition(0)
                self.players[key].play()
                self.update_status(f"Play: {key.upper()}")
            return

        if self.current_key:
            if key_code == Qt.Key.Key_Space:
                self.start_stutter("MACHINE GUN")
            elif key_code == Qt.Key.Key_Tab:
                self.start_stutter("SLIP ROLL")
                self.slip_start_time = time.time()

    def handle_release(self, event):
        key = event.text().lower()
        key_code = event.key()
        if key == self.drum_key_active:
            self.players[key].pause()
            self.players[key].setPosition(0)
            self.drum_key_active = None
            self.update_status("Stopped")
        if key_code == Qt.Key.Key_Space: self.stop_stutter(slip_roll=False)
        elif key_code == Qt.Key.Key_Tab: self.stop_stutter(slip_roll=True)

    def start_stutter(self, mode_name):
        if not self.is_stuttering:
            self.is_stuttering = True
            self.cue_point = self.players[self.current_key].position()
            self.stutter_timer.start(self.current_loop_speed)
            self.update_status(f"{mode_name}")
            self.centralWidget().setStyleSheet("QWidget { background-color: #550000; }") 

    def stop_stutter(self, slip_roll=False):
        if self.is_stuttering:
            self.is_stuttering = False
            self.stutter_timer.stop()
            self.update_status("Playing")
            self.centralWidget().setStyleSheet("")
            if self.current_key:
                if slip_roll:
                    elapsed_real_time_ms = (time.time() - self.slip_start_time) * 1000
                    elapsed_video_time_ms = elapsed_real_time_ms * self.playback_rate
                    target_pos = self.cue_point + elapsed_video_time_ms
                    self.players[self.current_key].setPosition(int(target_pos))
                self.players[self.current_key].play()

    def switch_to_key(self, key):
        if self.current_key and self.current_key != key:
            self.players[self.current_key].stop()
            self.video_items[self.current_key].hide()
        
        self.players[key].setPlaybackRate(self.playback_rate)
        self.video_items[key].show()
        rect = self.projector.scene.sceneRect()
        self.video_items[key].setSize(rect.size())
        self.current_key = key

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LooperApp()
    window.show()
    sys.exit(app.exec())