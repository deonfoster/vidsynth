import sys
import os
import time
import math
import json
import numpy as np
import librosa
import subprocess
import shutil
import uuid # For unique filenames to prevent locking
from pydub import AudioSegment

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QLabel, QVBoxLayout, QPushButton, QSlider,
                             QFileDialog, QHBoxLayout, QComboBox, QScrollArea,
                             QSpinBox, QLineEdit, QAbstractSpinBox, 
                             QGraphicsView, QGraphicsScene, QGraphicsRectItem,
                             QButtonGroup, QRadioButton)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtMultimedia import (QMediaPlayer, QAudioOutput, QMediaDevices)
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import (QUrl, Qt, QTimer, QEvent, QThread, pyqtSignal, 
                          QRectF, QObject)
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QFont

# --- STYLING ---
DARK_THEME = """
QMainWindow { background-color: #121212; }
QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QPushButton { 
    background-color: #2a2a2a; color: #ccc; border: 1px solid #444; 
    padding: 8px; border-radius: 4px; font-weight: bold; min-width: 60px;
}
QPushButton:hover { background-color: #3a3a3a; border: 1px solid #666; color: white; }
QPushButton:pressed { background-color: #00CCFF; color: black; }

QPushButton[step="true"] { 
    background-color: #1a1a1a; border: 1px solid #333; 
    min-width: 25px; min-height: 25px; max-width: 40px; max-height: 40px;
}
QPushButton[step="true"]:disabled { background-color: #0a0a0a; border: 1px solid #222; color: #333; }
QPushButton[active="true"] { border: 2px solid white; background-color: #444; }
QPushButton[recording="true"] { background-color: #FF0000; color: white; border: 1px solid #FF5555; }
QPushButton[paint="true"]:checked { background-color: #00CCFF; color: black; border: 1px solid white; }
QPushButton[len="true"]:checked { background-color: #00FF66; color: black; border: 1px solid white; }
QPushButton[sync="true"] { background-color: #444; color: #888; }
QPushButton[sync="true"]:checked { background-color: #FF0055; color: white; border: 1px solid white; }

QRadioButton { color: #ccc; font-weight: bold; }
QRadioButton::indicator:checked { background-color: #00CCFF; border: 1px solid white; border-radius: 6px; width: 12px; height: 12px;}
QSpinBox { background-color: #2a2a2a; color: #00CCFF; border: 1px solid #444; padding: 5px; font-weight: bold; }
QSpinBox::up-button, QSpinBox::down-button { width: 20px; background-color: #333; border-left: 1px solid #444; }
QComboBox { background-color: #2a2a2a; color: #ccc; border: 1px solid #444; padding: 5px; min-width: 80px; }
QSlider::groove:horizontal { border: 1px solid #333; height: 6px; background: #1a1a1a; margin: 2px 0; border-radius: 3px; }
QSlider::handle:horizontal { background: #00CCFF; border: 1px solid #00CCFF; width: 24px; height: 24px; margin: -9px 0; border-radius: 12px; }
"""

KEY_MAP = {'a': (0, 0, "#FF0055"), 's': (0, 1, "#00CCFF"), 'd': (1, 0, "#00FF66"), 'f': (1, 1, "#FFAA00")}

# --- HELPERS ---
class ShiftSpinBox(QSpinBox):
    def stepBy(self, steps):
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier: steps *= 10
        super().stepBy(steps)

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
            
            # Create a Base WAV for Rubber Band to use later
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            clean_name = os.path.basename(self.filepath).replace(" ", "_")
            wav_path = os.path.join(temp_dir, f"{clean_name}_base.wav")
            
            # Export raw WAV
            audio_full.export(wav_path, format="wav")

            # Visualization Data
            duration_ms = len(audio_full)
            if duration_ms > 60000: audio_vis = audio_full[:60000]
            else: audio_vis = audio_full
            
            raw_samples = np.array(audio_full.get_array_of_samples())
            sample_rate = audio_full.frame_rate

            audio_vis = audio_vis.set_channels(1).set_frame_rate(11025) 
            vis_samples = np.array(audio_vis.get_array_of_samples())
            samples_float = vis_samples.astype(np.float32) / 32768.0
            tempo, _ = librosa.beat.beat_track(y=samples_float, sr=11025)
            if isinstance(tempo, np.ndarray): tempo = tempo.item()
            bpm = float(round(tempo, 2))

            draw_samples = vis_samples[::150]
            max_val = np.max(np.abs(draw_samples)) or 1
            draw_samples = draw_samples / max_val
            pixmap = QPixmap(self.width, self.height)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.bg_color.darker(150), 1))
            center_y = self.height / 2
            step = len(draw_samples) / self.width
            for x in range(self.width):
                if self.isInterruptionRequested(): return
                idx = int(x * step)
                if idx < len(draw_samples):
                    h = abs(draw_samples[idx]) * (self.height * 0.9)
                    painter.drawLine(x, int(center_y - h/2), x, int(center_y + h/2))
            painter.end()
            
            if not self.isInterruptionRequested(): 
                self.finished.emit(self.key, pixmap, bpm, duration_ms, raw_samples, sample_rate, wav_path)
        except Exception as e:
            print(f"Analysis Error: {e}")
            if not self.isInterruptionRequested(): self.finished.emit(self.key, QPixmap(), 120.0, 0, None, 44100, "")

class RubberBandWorker(QThread):
    finished = pyqtSignal(str, str, float) # key, new_path, target_rate
    
    def __init__(self, key, original_wav, tempo_ratio):
        super().__init__()
        self.key = key
        self.original_wav = original_wav
        self.tempo_ratio = tempo_ratio

    def run(self):
        try:
            if not os.path.exists(self.original_wav): return
            if self.tempo_ratio <= 0: return
            
            # --- CRITICAL: Use UUID to prevent file locking issues ---
            unique_id = uuid.uuid4().hex[:8]
            base, ext = os.path.splitext(self.original_wav)
            out_path = f"{base}_stretch_{self.tempo_ratio:.2f}_{unique_id}{ext}"
            
            # Rubber Band CLI Logic
            # If video speeds up (2.0x), duration is halved.
            # Rubberband -t modifies duration. -t 0.5 = Half duration.
            # So rubberband ratio = 1.0 / video_rate
            rb_ratio = 1.0 / self.tempo_ratio
            
            if shutil.which("rubberband") is None:
                print("ERROR: 'rubberband' command not found in PATH.")
                return

            # Execute Rubber Band
            # -q = quiet, realtime = optimize for streaming/speed
            cmd = ["rubberband", "-q", "realtime", "-t", str(rb_ratio), self.original_wav, out_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.finished.emit(self.key, out_path, self.tempo_ratio)
            
        except Exception as e:
            print(f"Rubber Band Process Error: {e}")

# --- DECK ---
class VJDeck:
    def __init__(self, name, video_item):
        self.name = name
        self.video_item = video_item
        self.current_filepath = None
        self.base_wav_path = None
        
        # 1. Video Player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_item)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        # Dummy audio for video clock synchronization
        self.video_audio = QAudioOutput()
        self.player.setAudioOutput(self.video_audio)
        self.video_audio.setVolume(0) 

        # 2. Main Audio Player (Plays WAVs)
        self.audio_player = QMediaPlayer()
        self.main_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.main_output)
        self.audio_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        # 3. Cue Player
        self.cue_player = QMediaPlayer()
        self.cue_output = QAudioOutput()
        self.cue_player.setAudioOutput(self.cue_output)
        self.cue_player.setLoops(QMediaPlayer.Loops.Infinite)
        
        self.cue_active = False 
        self.raw_samples = None
        self.sample_rate = 44100
        self.target_volume = 1.0 
        self.playback_rate = 1.0
        
        # Anti-Pop Fader
        self.fade_level = 1.0
        self.fade_timer = QTimer()
        self.fade_timer.setInterval(10)
        self.fade_timer.timeout.connect(self._process_fade)

    def load_video(self, filepath):
        self.current_filepath = filepath
        self.player.setSource(QUrl.fromLocalFile(filepath))

    def load_base_audio(self, wav_path):
        self.base_wav_path = wav_path
        # Initially load the base audio
        self.switch_audio_source(wav_path, reset_rate_to_video=True)

    def switch_audio_source(self, path, reset_rate_to_video=False):
        """ Swaps the audio file seamlessly """
        pos = self.player.position()
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        
        url = QUrl.fromLocalFile(path)
        
        # We must memorize state before swap
        was_muted = self.main_output.isMuted()
        
        self.audio_player.setSource(url)
        self.cue_player.setSource(url)
        
        # RATE LOGIC
        if reset_rate_to_video:
            # Chipmunk mode (standard pitch shift)
            self.audio_player.setPlaybackRate(self.playback_rate)
            self.cue_player.setPlaybackRate(self.playback_rate)
            # Position matches exactly
            self.audio_player.setPosition(pos)
            self.cue_player.setPosition(pos)
        else:
            # Stretched mode (pitch corrected file)
            # The file itself is stretched, so play at 1.0x
            self.audio_player.setPlaybackRate(1.0)
            self.cue_player.setPlaybackRate(1.0)
            # Map Position: if video is 2.0x, 10s video is at 5000ms real-time.
            # Stretched audio file length is 5000ms.
            # So Audio Pos = Video Pos / Rate
            mapped_pos = int(pos / self.playback_rate)
            self.audio_player.setPosition(mapped_pos)
            self.cue_player.setPosition(mapped_pos)

        if playing:
            self.audio_player.play()
            if self.cue_active: self.cue_player.play()

    def has_media(self): 
        return self.player.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia

    def set_audio_data(self, samples, rate):
        self.raw_samples = samples; self.sample_rate = rate

    def find_zero_crossing(self, target_ms):
        if self.raw_samples is None: return target_ms
        target_idx = int((target_ms / 1000.0) * self.sample_rate)
        target_idx -= target_idx % 2 
        window = int(0.02 * self.sample_rate) 
        start = max(0, target_idx - window)
        end = min(len(self.raw_samples), target_idx + window)
        if start >= end: return target_ms
        segment = self.raw_samples[start:end]
        min_local_idx = np.argmin(np.abs(segment))
        return int(((start + min_local_idx) / self.sample_rate) * 1000.0)

    def trigger(self, pos):
        self.main_output.setMuted(True)
        if self.cue_active: self.cue_output.setMuted(True)
        
        safe_pos = self.find_zero_crossing(pos)
        
        self.player.setPosition(safe_pos)
        
        # Audio Trigger Mapping
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

    def play(self): 
        self.player.play(); self.audio_player.play()
        if self.cue_active: self.cue_player.play()
    def pause(self): 
        self.player.pause(); self.audio_player.pause(); self.cue_player.pause()
    def seek(self, pos):
        self.player.setPosition(pos)
        # Handle Rate Mapping
        a_pos = pos
        if self.audio_player.playbackRate() == 1.0 and self.playback_rate != 1.0:
            a_pos = int(pos / self.playback_rate)
        self.audio_player.setPosition(a_pos)
        if self.cue_active: self.cue_player.setPosition(a_pos)
        
    def position(self): return self.player.position()
    def duration(self): return self.player.duration()
    def playbackState(self): return self.player.playbackState()
    
    def setPlaybackRate(self, rate): 
        # Sets Video Rate + Instant Audio Pitch Shift (Chipmunk)
        self.playback_rate = rate
        self.player.setPlaybackRate(rate)
        
        # If we currently have a stretched file loaded, we need to revert to base audio
        # so we can pitch shift it instantly to match the new video speed
        if self.base_wav_path:
             # Only revert if currently playing a "fixed" file (rate 1.0)
             if self.audio_player.playbackRate() == 1.0:
                 self.switch_audio_source(self.base_wav_path, reset_rate_to_video=True)
             
             # Apply chipmunk rate
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
        if self.filename != "[Empty]" and self.track_duration > 0:
            if self.is_deck_a or self.is_deck_b:
                painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
                painter.drawLine(int(self.playhead_x), 0, int(self.playhead_x), self.height())
            cue_colors = {1: QColor("#FF0000"), 2: QColor("#00FF00"), 3: QColor("#00CCFF")}
            for num, pos_ms in self.hotcues.items():
                if pos_ms <= self.track_duration:
                    cx = int((pos_ms / self.track_duration) * self.width())
                    col = cue_colors.get(num, QColor("white"))
                    painter.setPen(QPen(col, 2)); painter.drawLine(cx, 0, cx, self.height())
                    painter.setBrush(col); painter.setPen(Qt.PenStyle.NoPen); painter.drawRect(cx, 0, 14, 14)
                    painter.setPen(QColor("black")); painter.setFont(QFont("Arial", 9, QFont.Weight.Bold)); painter.drawText(cx+3, 11, str(num))
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

class ProjectorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projector Output"); self.resize(800, 600); self.setStyleSheet("background-color: black;")
        self.view = QGraphicsView(self)
        self.view.setViewport(QOpenGLWidget()); self.view.viewport().setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.view.setStyleSheet("background: black; border: none;"); self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.scene = QGraphicsScene(self); self.view.setScene(self.scene); self.scene.setBackgroundBrush(Qt.GlobalColor.black)
        layout = QVBoxLayout(); layout.addWidget(self.view); layout.setContentsMargins(0, 0, 0, 0); self.setLayout(layout)
    def resizeEvent(self, event): self.scene.setSceneRect(0, 0, self.width(), self.height()); self.view.resize(self.width(), self.height()); super().resizeEvent(event)

class SeqStepButton(QPushButton):
    rightClicked = pyqtSignal(int)
    def __init__(self, index, parent_app):
        super().__init__("")
        self.index = index; self.parent_app = parent_app; self.setProperty("step", "true")
        self.data = None; self.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.clicked.connect(self.on_click)
    def on_click(self):
        mods = QApplication.keyboardModifiers()
        if mods == Qt.KeyboardModifier.ShiftModifier: self.parent_app.save_step_data(self.index)
        else: self.parent_app.handle_step_click(self.index)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton: self.rightClicked.emit(self.index)
        else: super().mousePressEvent(event)
    def update_style(self, is_active_step):
        style = "border-radius: 4px; min-width: 25px; min-height: 25px;"
        if is_active_step: style += "border: 2px solid white;"
        else: style += "border: 1px solid #333;"
        if self.data:
            c = self.data.get('cue_num', 0)
            if c == 1: style += "background-color: #FF0000;" 
            elif c == 2: style += "background-color: #00FF00; color: black;" 
            elif c == 3: style += "background-color: #00CCFF; color: black;" 
            else: style += "background-color: #555;"
        else: style += "background-color: #1a1a1a;"
        self.setStyleSheet(style)

# --- MAIN APP ---
class LooperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VJ Sequencer v10.0 (DSP Engine)")
        self.resize(800, 1000)
        QApplication.instance().setStyleSheet(DARK_THEME)

        self.projector = ProjectorWindow()
        self.deck_a = VJDeck("A", QGraphicsVideoItem())
        self.deck_b = VJDeck("B", QGraphicsVideoItem())
        self.projector.scene.addItem(self.deck_a.video_item)
        self.projector.scene.addItem(self.deck_b.video_item)
        self.deck_a.video_item.setZValue(0); self.deck_b.video_item.setZValue(1)
        self.projector.show()

        self.deck_a.player.positionChanged.connect(self.on_deck_a_pos)
        self.deck_b.player.positionChanged.connect(self.on_deck_b_pos)

        self.buttons = {}; self.bank_data = {0: {}, 1: {}, 2: {}} 
        self.clip_meta = {}; self.hotcue_data = {}; self.audio_buffer = {}
        self.clip_patterns = {} 
        self.active_clip_a = None; self.active_clip_b = None
        self.current_bank = 0; self.current_generation = 0; self.active_workers = []
        self.crossfader_value = 0.0; self.master_bpm = 120.0; self.tap_times = []
        self.transport_start_time = time.time(); self.quantize_active = True 

        # SEQUENCER
        self.seq_running = False; self.seq_recording = False
        self.current_step = 0; self.active_paint_cue = 1 
        self.seq_length = 64 
        self.seq_multiplier = 1.0
        self.seq_timer = QTimer(); self.seq_timer.setTimerType(Qt.TimerType.PreciseTimer); self.seq_timer.timeout.connect(self.run_sequencer_step)
        self.sequencer_buttons = [] 
        self.seq_edit_target = "A"

        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); self.setCentralWidget(scroll_area)
        central_widget = QWidget(); scroll_area.setWidget(central_widget)
        main_layout = QVBoxLayout(central_widget); main_layout.setSpacing(15); main_layout.setContentsMargins(20, 20, 20, 20)

        # Controls
        top_row = QHBoxLayout()
        btn_save = QPushButton("SAVE"); btn_save.clicked.connect(self.save_set)
        btn_load = QPushButton("LOAD"); btn_load.clicked.connect(self.load_set)
        lbl_main = QLabel("MAIN:"); self.main_out_combo = QComboBox()
        lbl_cue = QLabel("HEADPHONES:"); self.cue_out_combo = QComboBox()
        self.audio_devices = QMediaDevices.audioOutputs()
        for device in self.audio_devices:
            self.main_out_combo.addItem(device.description())
            self.cue_out_combo.addItem(device.description())
        self.main_out_combo.currentIndexChanged.connect(self.change_main_output)
        self.cue_out_combo.currentIndexChanged.connect(self.change_cue_output)
        
        self.btn_vid_sync = QPushButton("VID SYNC: ON"); self.btn_vid_sync.setCheckable(True); self.btn_vid_sync.setChecked(True)
        self.btn_vid_sync.setProperty("sync", "true"); self.btn_vid_sync.clicked.connect(self.toggle_vid_sync)

        self.btn_quant = QPushButton("QUANT: ON"); self.btn_quant.setCheckable(True); self.btn_quant.setChecked(True)
        self.btn_quant.setStyleSheet("background-color: #00FF66; color: black;")
        self.btn_quant.clicked.connect(self.toggle_quantize)
        self.btn_align = QPushButton("AUTO-ALIGN"); self.btn_align.clicked.connect(self.auto_align_phase)
        
        top_row.addWidget(btn_save); top_row.addWidget(btn_load); top_row.addWidget(self.btn_vid_sync); top_row.addWidget(self.btn_quant); top_row.addWidget(self.btn_align)
        top_row.addWidget(lbl_main); top_row.addWidget(self.main_out_combo); top_row.addWidget(lbl_cue); top_row.addWidget(self.cue_out_combo)
        main_layout.addLayout(top_row)
        
        bank_row = QHBoxLayout()
        self.bank_btns = []
        for i in range(3):
            b = QPushButton(f"BANK {i+1}"); b.setCheckable(True); b.clicked.connect(lambda _, x=i: self.switch_bank(x))
            bank_row.addWidget(b); self.bank_btns.append(b)
        self.bank_btns[0].setChecked(True); main_layout.addLayout(bank_row)

        grid_layout = QGridLayout(); main_layout.addLayout(grid_layout)
        for key, (row, col, color) in KEY_MAP.items():
            btn = InteractiveWaveform(key, color, self); self.buttons[key] = btn; grid_layout.addWidget(btn, row, col)

        cue_row = QHBoxLayout()
        self.btn_cue_a = QPushButton("CUE A"); self.btn_cue_a.setCheckable(True); self.btn_cue_a.setProperty("cue", "true")
        self.btn_cue_a.clicked.connect(self.toggle_cue_a)
        self.btn_cue_b = QPushButton("CUE B"); self.btn_cue_b.setCheckable(True); self.btn_cue_b.setProperty("cue", "true")
        self.btn_cue_b.clicked.connect(self.toggle_cue_b)
        cue_row.addWidget(self.btn_cue_a); cue_row.addStretch(); cue_row.addWidget(self.btn_cue_b)
        main_layout.addLayout(cue_row)

        xfader_lbl = QLabel("CROSSFADER"); xfader_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); main_layout.addWidget(xfader_lbl)
        self.fader_slider = QSlider(Qt.Orientation.Horizontal); self.fader_slider.setRange(0, 100); self.fader_slider.setValue(0)
        self.fader_slider.valueChanged.connect(self.on_fader_ui_changed); main_layout.addWidget(self.fader_slider)

        cue_edit_row = QHBoxLayout()
        self.cue_spinboxes = {}
        for i in range(1, 4):
            lbl = QLabel(f"CUE {i}:"); sb = ShiftSpinBox(); sb.setRange(0, 9999999); sb.setFixedWidth(100)
            sb.valueChanged.connect(lambda val, num=i: self.on_cue_edited(num, val))
            cue_edit_row.addWidget(lbl); cue_edit_row.addWidget(sb); self.cue_spinboxes[i] = sb
        cue_edit_row.addStretch(); main_layout.addLayout(cue_edit_row)

        bpm_row = QHBoxLayout()
        self.bpm_label = QLabel("MASTER BPM: 120.0"); self.bpm_label.setStyleSheet("color: #00FF66; font-size: 14px; font-weight: bold;")
        
        lbl_speed = QLabel("SEQ SPEED:")
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["1/2x", "1x", "2x"])
        self.combo_speed.setCurrentIndex(1)
        self.combo_speed.currentIndexChanged.connect(self.change_seq_speed)
        
        btn_tap = QPushButton("TAP (Ret)"); btn_tap.setFixedSize(80, 30); btn_tap.setStyleSheet("background: #444; border: 1px solid #666;")
        btn_tap.clicked.connect(self.handle_tap_tempo)
        
        bpm_row.addStretch(); bpm_row.addWidget(self.bpm_label); bpm_row.addWidget(lbl_speed); bpm_row.addWidget(self.combo_speed); bpm_row.addWidget(btn_tap); bpm_row.addStretch(); main_layout.addLayout(bpm_row)

        seq_header = QHBoxLayout()
        seq_lbl = QLabel("64-STEP GRID")
        self.len_group = QButtonGroup(self)
        len_vals = [16, 32, 64]
        for val in len_vals:
            btn = QPushButton(str(val)); btn.setCheckable(True); btn.setFixedWidth(40); btn.setProperty("len", "true")
            if val == 64: btn.setChecked(True)
            self.len_group.addButton(btn, val); seq_header.addWidget(btn)
        self.len_group.idClicked.connect(self.set_sequence_length)

        self.rad_edit_a = QRadioButton("EDIT DECK A"); self.rad_edit_a.setChecked(True)
        self.rad_edit_a.toggled.connect(self.update_sequencer_ui)
        self.rad_edit_b = QRadioButton("EDIT DECK B")
        self.rad_edit_b.toggled.connect(self.update_sequencer_ui)
        seq_header.addStretch(); seq_header.addWidget(seq_lbl); seq_header.addSpacing(20); seq_header.addWidget(self.rad_edit_a); seq_header.addWidget(self.rad_edit_b); seq_header.addStretch()
        main_layout.addLayout(seq_header)
        
        paint_row = QHBoxLayout()
        paint_lbl = QLabel("PAINT BRUSH:"); paint_row.addStretch(); paint_row.addWidget(paint_lbl)
        self.paint_group = QButtonGroup(self)
        for i in range(1, 4):
            btn = QPushButton(str(i)); btn.setCheckable(True); btn.setFixedWidth(40); btn.setProperty("paint", "true")
            if i == 1: btn.setChecked(True)
            self.paint_group.addButton(btn, i); paint_row.addWidget(btn)
        self.paint_group.idClicked.connect(self.set_paint_cue); paint_row.addStretch(); main_layout.addLayout(paint_row)

        seq_grid = QGridLayout(); seq_grid.setSpacing(4)
        for i in range(64):
            btn = SeqStepButton(i, self)
            btn.rightClicked.connect(self.handle_step_right_click)
            self.sequencer_buttons.append(btn)
            row = i // 8; col = i % 8
            seq_grid.addWidget(btn, row, col)
        main_layout.addLayout(seq_grid)
        
        seq_ctrl = QHBoxLayout()
        self.btn_play_seq = QPushButton("START SEQ (P)"); self.btn_play_seq.setCheckable(True); self.btn_play_seq.clicked.connect(self.toggle_sequencer)
        self.btn_rec_seq = QPushButton("REC"); self.btn_rec_seq.setCheckable(True); self.btn_rec_seq.clicked.connect(self.toggle_record)
        self.btn_clear_seq = QPushButton("CLEAR SEQ"); self.btn_clear_seq.setStyleSheet("background-color: #AA0000;"); self.btn_clear_seq.clicked.connect(self.clear_sequence)
        seq_ctrl.addWidget(self.btn_play_seq); seq_ctrl.addWidget(self.btn_rec_seq); seq_ctrl.addWidget(self.btn_clear_seq)
        main_layout.addLayout(seq_ctrl)

        self.status_label = QLabel("Ready (Ensure rubberband CLI installed)"); self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter); main_layout.addWidget(self.status_label)
        self.reopen_btn = QPushButton("OPEN PROJECTOR WINDOW"); self.reopen_btn.clicked.connect(self.projector.show); main_layout.addWidget(self.reopen_btn)

        for btn in self.findChildren(QPushButton): btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for slider in self.findChildren(QSlider): slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for sb in self.findChildren(QSpinBox): sb.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        QApplication.instance().installEventFilter(self)
        self.update_mixer()

    def change_seq_speed(self, index):
        if index == 0: self.seq_multiplier = 0.5 
        elif index == 1: self.seq_multiplier = 1.0 
        elif index == 2: self.seq_multiplier = 2.0 
        self.update_clock()

    def toggle_vid_sync(self):
        on = self.btn_vid_sync.isChecked()
        self.btn_vid_sync.setText(f"VID SYNC: {'ON' if on else 'OFF'}")
        if on:
            self.sync_deck_speed(self.deck_a, self.active_clip_a)
            self.sync_deck_speed(self.deck_b, self.active_clip_b)
        else:
            self.deck_a.setPlaybackRate(1.0)
            self.deck_b.setPlaybackRate(1.0)

    def set_sequence_length(self, length):
        self.seq_length = length
        self.update_grid_visibility()
        self.status_label.setText(f"Sequence Length: {length} Steps")

    def update_grid_visibility(self):
        for i, btn in enumerate(self.sequencer_buttons):
            if i < self.seq_length:
                btn.setEnabled(True)
                btn.setProperty("step", "true") 
            else:
                btn.setEnabled(False)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            if i < self.seq_length and btn.data: btn.update_style(False)

    def change_main_output(self, index):
        if 0 <= index < len(self.audio_devices): d = self.audio_devices[index]; self.deck_a.set_main_output(d); self.deck_b.set_main_output(d)
    def change_cue_output(self, index):
        if 0 <= index < len(self.audio_devices): d = self.audio_devices[index]; self.deck_a.set_cue_output(d); self.deck_b.set_cue_output(d)
    def toggle_cue_a(self): self.deck_a.set_cue_active(self.btn_cue_a.isChecked())
    def toggle_cue_b(self): self.deck_b.set_cue_active(self.btn_cue_b.isChecked())

    def get_target_deck_info(self):
        if self.rad_edit_a.isChecked(): return self.deck_a, self.deck_a.current_filepath
        else: return self.deck_b, self.deck_b.current_filepath

    def update_sequencer_ui(self):
        _, filepath = self.get_target_deck_info()
        pattern = {}
        if filepath and filepath in self.clip_patterns: pattern = self.clip_patterns[filepath]
        for i, btn in enumerate(self.sequencer_buttons):
            if i in pattern: btn.data = pattern[i]
            else: btn.data = None
            btn.update_style(False)
        self.update_grid_visibility()

    def clear_sequence(self):
        _, filepath = self.get_target_deck_info()
        if filepath:
            self.clip_patterns[filepath] = {}
            self.update_sequencer_ui()
            self.status_label.setText("Cleared Sequence for Current Clip")

    def handle_step_click(self, index):
        if index >= self.seq_length: return 
        deck, filepath = self.get_target_deck_info()
        if not filepath: return
        if filepath not in self.clip_patterns: self.clip_patterns[filepath] = {}
        if index in self.clip_patterns[filepath]: del self.clip_patterns[filepath][index]
        else:
            cue_num = self.active_paint_cue
            if filepath in self.hotcue_data and cue_num in self.hotcue_data[filepath]:
                pos = self.hotcue_data[filepath][cue_num]
                self.clip_patterns[filepath][index] = {'pos': pos, 'cue_num': cue_num}
            else:
                self.status_label.setText(f"Cue {cue_num} not set!")
                return
        self.update_sequencer_ui()

    def handle_step_right_click(self, index):
        deck, filepath = self.get_target_deck_info()
        if not filepath: return
        if filepath in self.clip_patterns and index in self.clip_patterns[filepath]:
            del self.clip_patterns[filepath][index]
            self.update_sequencer_ui()

    def run_sequencer_step(self):
        prev_step = (self.current_step - 1) % self.seq_length
        self.sequencer_buttons[prev_step].update_style(False)
        if self.current_step >= self.seq_length: self.current_step = 0
        self.sequencer_buttons[self.current_step].update_style(True)

        path_a = self.deck_a.current_filepath
        if path_a and path_a in self.clip_patterns:
            pat_a = self.clip_patterns[path_a]
            if self.current_step in pat_a: self.deck_a.trigger(pat_a[self.current_step]['pos'])

        path_b = self.deck_b.current_filepath
        if path_b and path_b in self.clip_patterns:
            pat_b = self.clip_patterns[path_b]
            if self.current_step in pat_b: self.deck_b.trigger(pat_b[self.current_step]['pos'])

        self.current_step = (self.current_step + 1) % self.seq_length

    def handle_hotcue(self, num, is_delete):
        deck, key = self.get_dominant_deck()
        path = self.bank_data[self.current_bank].get(key)
        if deck and path:
            if is_delete:
                if num in self.hotcue_data[path]: del self.hotcue_data[path][num]
                self.status_label.setText(f"Deleted Cue {num}")
            else:
                if num in self.hotcue_data[path]: 
                    pos = self.hotcue_data[path][num]
                    deck.trigger(pos)
                    self.status_label.setText(f"Triggered Cue {num}")
                    if self.seq_running and self.seq_recording:
                        if path not in self.clip_patterns: self.clip_patterns[path] = {}
                        self.clip_patterns[path][self.current_step] = {'pos': pos, 'cue_num': num}
                        target_deck, target_path = self.get_target_deck_info()
                        if target_path == path: self.update_sequencer_ui()
                        self.status_label.setText(f"RECORDED Cue {num}")
                else: 
                    self.hotcue_data[path][num] = deck.position()
                    self.status_label.setText(f"Set Hotcue {num}")
            self.buttons[key].update()
            self.update_cue_display()

    def set_paint_cue(self, id): self.active_paint_cue = id
    def update_cue_display(self):
        deck, key = self.get_dominant_deck()
        path = self.bank_data[self.current_bank].get(key)
        if path and path in self.hotcue_data: self.buttons[key].hotcues = self.hotcue_data[path]; self.buttons[key].update()
        for sb in self.cue_spinboxes.values(): sb.blockSignals(True)
        if path and path in self.hotcue_data:
            cues = self.hotcue_data[path]
            for i in range(1, 4): self.cue_spinboxes[i].setValue(cues.get(i, 0))
        else:
            for i in range(1, 4): self.cue_spinboxes[i].setValue(0)
        for sb in self.cue_spinboxes.values(): sb.blockSignals(False)

    def on_cue_edited(self, num, value):
        deck, key = self.get_dominant_deck()
        path = self.bank_data[self.current_bank].get(key)
        if path:
            if path not in self.hotcue_data: self.hotcue_data[path] = {}
            self.hotcue_data[path][num] = value
            self.buttons[key].hotcues = self.hotcue_data[path]; self.buttons[key].update()

    def toggle_sequencer(self):
        self.seq_running = not self.seq_running
        self.btn_play_seq.setChecked(self.seq_running)
        self.btn_play_seq.setText("STOP SEQ (P)" if self.seq_running else "START SEQ (P)")
        if self.seq_running: self.update_clock()
        else:
            self.seq_timer.stop()
            for b in self.sequencer_buttons: b.update_style(False)

    def update_clock(self):
        if self.master_bpm <= 0: return
        base_interval = (60000.0 / self.master_bpm) / 4 
        final_interval = base_interval / self.seq_multiplier
        self.seq_timer.setInterval(int(final_interval))
        if self.seq_running and not self.seq_timer.isActive(): self.seq_timer.start()

    def toggle_record(self):
        self.seq_recording = self.btn_rec_seq.isChecked()
        self.btn_rec_seq.setProperty("recording", str(self.seq_recording).lower())
        self.btn_rec_seq.style().unpolish(self.btn_rec_seq); self.btn_rec_seq.style().polish(self.btn_rec_seq)

    def save_step_data(self, index): pass

    def assign_clip_to_bank(self, key, filepath):
        self.bank_data[self.current_bank][key] = filepath
        self.start_processing(key, filepath)

    def start_processing(self, key, filepath):
        self.buttons[key].set_loading()
        worker = AudioAnalysisWorker(key, filepath, 200, 120, self.buttons[key].base_color.name(), self.current_generation)
        worker.finished.connect(self.on_prep_done)
        self.active_workers.append(worker)
        worker.start()

    def on_prep_done(self, key, pixmap, bpm, duration, raw_samples, rate, wav_path):
        path = self.bank_data[self.current_bank].get(key)
        if path: 
            self.clip_meta[path] = bpm
            # self.audio_buffer[path] = (raw_samples, rate) # Old buffer, not used for rubberband
            if self.active_clip_a == key: self.deck_a.load_base_audio(wav_path)
            if self.active_clip_b == key: self.deck_b.load_base_audio(wav_path)
        if key in self.buttons: self.buttons[key].set_data(pixmap, bpm, duration)
        if path and path in self.hotcue_data: self.buttons[key].hotcues = self.hotcue_data[path]

    def assign_to_deck(self, deck_name, key):
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        if path not in self.hotcue_data: self.hotcue_data[path] = {}
        self.buttons[key].hotcues = self.hotcue_data[path]
        
        target_deck = self.deck_a if deck_name == "A" else self.deck_b
        target_deck.load_video(path) 
        
        if deck_name == "A": self.active_clip_a = key
        else: self.active_clip_b = key
        
        # Re-trigger analysis to ensure WAV path is found
        self.start_processing(key, path)
        
        target_deck.video_item.show(); target_deck.video_item.setSize(self.projector.scene.sceneRect().size())
        for k, b in self.buttons.items():
            if deck_name == "A": b.is_deck_a = (k == key)
            else: b.is_deck_b = (k == key)
        for b in self.buttons.values(): b.update()
        
        if self.quantize_active and self.master_bpm > 0:
            wait_ms = self.get_ms_until_next_bar()
            self.status_label.setText(f"Queued: Starts in {wait_ms}ms...")
            self.buttons[key].set_loading()
            target_deck.pause()
            QTimer.singleShot(wait_ms, lambda: self._execute_play_synced(target_deck))
            QTimer.singleShot(wait_ms, lambda: self._clear_loading_state(key)) 
        else:
            target_deck.play()
        self.update_mixer(); self.update_sequencer_ui()

    def _execute_play_synced(self, deck): deck.seek(0); deck.play(); self.status_label.setText("Playing (Quantized)")
    def _clear_loading_state(self, key): 
        if key in self.buttons: self.buttons[key].loading = False; self.buttons[key].update()

    def switch_bank(self, index):
        self.current_bank = index; self.current_generation += 1
        for i, btn in enumerate(self.bank_btns): btn.setChecked(i == index)
        current_data = self.bank_data[self.current_bank]
        for key in KEY_MAP.keys():
            self.buttons[key].is_deck_a = (key == self.active_clip_a)
            self.buttons[key].is_deck_b = (key == self.active_clip_b)
            if key in current_data: self.start_processing(key, current_data[key])
            else: self.buttons[key].filename = "[Empty]"; self.buttons[key].update()

    def toggle_all_playback(self):
        state_a = self.deck_a.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        state_b = self.deck_b.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if state_a or state_b:
            if state_a: self.deck_a.pause()
            if state_b: self.deck_b.pause()
            if self.seq_running: self.toggle_sequencer()
            self.status_label.setText("Paused All")
        else:
            if self.deck_a.has_media(): self.deck_a.play()
            if self.deck_b.has_media(): self.deck_b.play()
            self.status_label.setText("Resumed All")

    def auto_align_phase(self):
        if self.master_bpm <= 0: return
        beat_ms = 60000.0 / self.master_bpm
        now = time.time(); elapsed_ms = (now - self.transport_start_time) * 1000
        master_phase_offset = elapsed_ms % beat_ms
        for deck in [self.deck_a, self.deck_b]:
            if deck.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                current_pos = deck.position()
                diff = master_phase_offset - (current_pos % beat_ms)
                if abs(diff) > (beat_ms / 2): diff += beat_ms if diff < 0 else -beat_ms
                deck.seek(max(0, int(current_pos + diff)))
        self.status_label.setText("System Auto-Aligned")

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
                if self.btn_vid_sync.isChecked():
                    self.sync_deck_speed(self.deck_a, self.active_clip_a)
                    self.sync_deck_speed(self.deck_b, self.active_clip_b)
                self.update_clock()

    def sync_deck_speed(self, deck, key):
        if not key: return
        path = self.bank_data[self.current_bank].get(key)
        if not path: return
        clip_bpm = self.clip_meta.get(path, 120.0)
        sync_rate = 1.0
        if clip_bpm > 0 and self.master_bpm > 0: sync_rate = self.master_bpm / clip_bpm
        
        # Apply Instant Rate to Video + Audio (Chipmunk mode first)
        deck.setPlaybackRate(sync_rate)
        
        # Start Rubber Band Background Job
        if deck.base_wav_path:
            worker = RubberBandWorker(key, deck.base_wav_path, sync_rate)
            worker.finished.connect(lambda k, new_p, r: self.on_rubberband_done(deck, new_p, r))
            self.active_workers.append(worker) 
            worker.start()

    def on_rubberband_done(self, deck, new_wav_path, target_rate):
        # Swap seamlessly to the high-quality file
        deck.switch_audio_source(new_wav_path, reset_rate_to_video=False)
        self.status_label.setText(f"DSP Active: {target_rate:.2f}x")

    def toggle_quantize(self):
        self.quantize_active = self.btn_quant.isChecked()
        self.btn_quant.setText(f"QUANT: {'ON' if self.quantize_active else 'OFF'}")
        self.btn_quant.setStyleSheet(f"background-color: {'#00FF66' if self.quantize_active else '#444'}; color: black;")

    def get_ms_until_next_bar(self):
        if self.master_bpm <= 0: return 0
        beat_sec = 60.0 / self.master_bpm
        bar_sec = beat_sec * 4
        now = time.time(); elapsed = now - self.transport_start_time
        next_bar_time = math.ceil(elapsed / bar_sec) * bar_sec
        return int((next_bar_time - elapsed) * 1000)

    def on_fader_ui_changed(self, value): self.crossfader_value = value / 100.0; self.update_mixer()
    def update_mixer(self):
        val = self.crossfader_value
        self.deck_a.set_volume(1.0 - val); self.deck_b.set_volume(val)
        self.deck_a.video_item.setOpacity(1.0 - val); self.deck_b.video_item.setOpacity(val)
        self.update_cue_display()

    def on_deck_a_pos(self, pos):
        if self.active_clip_a and self.active_clip_a in self.buttons: self.buttons[self.active_clip_a].update_playhead(pos/self.deck_a.duration())
    def on_deck_b_pos(self, pos):
        if self.active_clip_b and self.active_clip_b in self.buttons: self.buttons[self.active_clip_b].update_playhead(pos/self.deck_b.duration())
    def change_audio_output(self, index):
        if 0 <= index < len(self.audio_devices): d = self.audio_devices[index]; self.deck_a.set_audio_device(d); self.deck_b.set_audio_device(d)
    
    def save_set(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if fn: 
            data = {
                'banks': self.bank_data,
                'hotcues': self.hotcue_data,
                'patterns': {k: {int(s): v for s, v in p.items()} for k, p in self.clip_patterns.items()}
            }
            json.dump(data, open(fn, 'w'))

    def load_set(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if fn: 
            data = json.load(open(fn, 'r'))
            if 'banks' in data:
                self.bank_data = {int(k): v for k, v in data['banks'].items()}
                self.hotcue_data = data.get('hotcues', {})
                self.clip_patterns = {k: {int(s): v for s, v in p.items()} for k, p in data.get('patterns', {}).items()}
            else:
                self.bank_data = {int(k): v for k, v in data.items()}
            self.switch_bank(0)

    def get_dominant_deck(self): return (self.deck_b, self.active_clip_b) if self.crossfader_value > 0.5 else (self.deck_a, self.active_clip_a)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            if focus and (isinstance(focus, (QSpinBox, QLineEdit, QAbstractSpinBox)) or focus.inherits("QAbstractSpinBox") or focus.inherits("QLineEdit")): return super().eventFilter(source, event)
            key = event.key(); text = event.text().lower(); mods = event.modifiers()
            if key == Qt.Key.Key_Space: self.toggle_all_playback(); return True
            if text == 'p': self.toggle_sequencer(); return True
            if mods & Qt.KeyboardModifier.ShiftModifier:
                if text == '!': self.handle_hotcue(1, True); return True
                if text == '@': self.handle_hotcue(2, True); return True
                if text == '#': self.handle_hotcue(3, True); return True
            else:
                if text == '1': self.handle_hotcue(1, False); return True
                if text == '2': self.handle_hotcue(2, False); return True
                if text == '3': self.handle_hotcue(3, False); return True
            if key == Qt.Key.Key_Return: self.handle_tap_tempo(); return True
            if key == Qt.Key.Key_Left: self.fader_slider.setValue(max(0, self.fader_slider.value()-5)); return True
            if key == Qt.Key.Key_Right: self.fader_slider.setValue(min(100, self.fader_slider.value()+5)); return True
            if text in ['5','6','7']: self.switch_bank(int(text)-5); return True
        return super().eventFilter(source, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LooperApp()
    window.show()
    sys.exit(app.exec())