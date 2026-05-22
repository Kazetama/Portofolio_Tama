import sys
import cv2
import torch
import time
import os
from ultralytics import YOLO
from collections import defaultdict, deque

from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QStackedWidget, QSpacerItem,
    QComboBox
)
import random
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QLinearGradient, QFont, QBrush, QPen
from PySide6.QtCore import QTimer, Qt, QUrl, QPointF
from PySide6.QtMultimedia import QSoundEffect, QMediaDevices
from PySide6.QtWidgets import QGraphicsDropShadowEffect

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-pose.pt")
AUDIO_DIR = os.path.join(BASE_DIR, "assets", "audio")
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "images")

HISTORY_LEN = 2
FINISH_SCORE = 20

MIN_RISE = 10
MIN_JUMP = 25
MIN_FALL = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)


class CaptureWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Capture Window")

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:#f0f0f0;")

        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(self.capture_image)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(10)
        layout.addWidget(self.label)
        layout.addWidget(self.btn_capture)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(50)

    def update_preview(self):
        if self.parent.current_frame is None:
            return

        frame = cv2.flip(self.parent.current_frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def capture_image(self):
        if self.parent.current_frame is None:
            return

        filename = f"capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, self.parent.current_frame)
        print("Saved:", filename)


class JumpApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Jump Battle Race | Premium Edition")
        self.apply_styles()

        self.game_mode = "PVP" # "PVP" or "AI"
        
        # Game State Variables
        self.is_playing = False
        self.countdown = None
        self.left_start_time = None
        self.right_start_time = None
        self.left_time = 0
        self.right_time = 0
        self.left_score = 0
        self.right_score = 0
        self.winner = None
        self.data_store = {}
        self.history_y = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.current_frame = None

        # AI Variables
        self.ai_last_jump_time = 0
        self.ai_target_cooldown = 1.0
        self.ai_jump_anim_start = 0
        self.ai_jump_duration = 0.6 # Durasi animasi lompat (detik)

        # Assets
        self.snd_go = QSoundEffect()
        self.snd_jump = QSoundEffect()
        self.snd_finish = QSoundEffect()
        self.snd_go.setSource(QUrl.fromLocalFile(os.path.join(AUDIO_DIR, "go.wav")))
        self.snd_jump.setSource(QUrl.fromLocalFile(os.path.join(AUDIO_DIR, "jump.wav")))
        self.snd_finish.setSource(QUrl.fromLocalFile(os.path.join(AUDIO_DIR, "finish.wav")))
        self.snd_go.setVolume(0.5)
        self.snd_jump.setVolume(0.5)
        self.snd_finish.setVolume(0.7)

        self.icon_size = 140
        
        # Player Assets
        self.p1_data = {"gender": "cowok", "neutral": None, "running": [], "happy": None, "sad": None}
        self.p2_data = {"gender": "cowok", "neutral": None, "running": [], "happy": None, "sad": None}
        
        # Load defaults to prevent crash before selection
        self.load_player_assets(1, "cowok")
        self.load_player_assets(2, "cowok")
        
        self.current_frame_idx = 0
        self.selecting_player = 1 # 1 or 2

        # Camera
        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)

        # Build UI
        self.stacked_widget = QStackedWidget()
        
        self.menu_page = self.create_menu_page()
        self.selection_page = self.create_selection_page()
        self.game_page = self.create_game_page()
        self.settings_page = self.create_settings_page()

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.selection_page)
        self.stacked_widget.addWidget(self.game_page)
        self.stacked_widget.addWidget(self.settings_page)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

    def load_player_assets(self, player_num, gender):
        base_path = os.path.join(IMAGE_DIR, gender)
        ext = "png" if gender == "cowok" else "jpeg"
        
        data = self.p1_data if player_num == 1 else self.p2_data
        data["gender"] = gender
        data["neutral_path"] = os.path.join(base_path, f"neutral.{ext}")
        data["neutral"] = QPixmap(data["neutral_path"])
        data["happy"] = os.path.join(base_path, f"happy.{ext}")
        data["sad"] = os.path.join(base_path, f"sad.{ext}")
        
        data["running"] = []
        webm_path = os.path.join(base_path, "running.webm")
        if os.path.exists(webm_path):
            cap_webm = cv2.VideoCapture(webm_path)
            while True:
                ret, wframe = cap_webm.read()
                if not ret: break
                rgba = cv2.cvtColor(wframe, cv2.COLOR_BGR2RGBA)
                black_pixels = (rgba[:, :, 0] < 30) & (rgba[:, :, 1] < 30) & (rgba[:, :, 2] < 30)
                rgba[black_pixels, 3] = 0
                rgba_copy = rgba.copy()
                wh, ww, wch = rgba_copy.shape
                wimg = QImage(rgba_copy.data, ww, wh, wch*ww, QImage.Format_RGBA8888)
                data["running"].append(QPixmap.fromImage(wimg))
            cap_webm.release()

    def create_menu_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("JUMP BATTLE RACE")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("🌟 Fun & Cute Edition 🌟")
        subtitle.setObjectName("subTitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        btn_pvp = QPushButton("👥 Player vs Player")
        btn_pvp.setFixedSize(320, 70)
        btn_pvp.setObjectName("modeBtn")
        btn_pvp.clicked.connect(lambda: self.start_selection("PVP"))
        
        btn_ai = QPushButton("🤖 Player vs AI")
        btn_ai.setFixedSize(320, 70)
        btn_ai.setObjectName("modeBtn")
        btn_ai.clicked.connect(lambda: self.start_selection("AI"))

        btn_settings = QPushButton("⚙️ Camera Settings")
        btn_settings.setFixedSize(320, 70)
        btn_settings.setObjectName("modeBtn")
        btn_settings.setStyleSheet("""
            QPushButton { background-color: #a8e6cf; border: 3px solid #8be3c4; }
            QPushButton:hover { background-color: #8be3c4; border: 3px solid #75cfb0; }
        """)
        btn_settings.clicked.connect(self.open_settings)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(btn_pvp, 0, Qt.AlignHCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_ai, 0, Qt.AlignHCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_settings, 0, Qt.AlignHCenter)
        
        return page

    def open_settings(self):
        self.refresh_cameras()
        self.stacked_widget.setCurrentWidget(self.settings_page)

    def create_selection_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)

        self.selection_title = QLabel("✨ CHOOSE YOUR CHARACTER ✨")
        self.selection_title.setStyleSheet("font-size: 42px; font-weight: 900; color: #ff6b81; margin-bottom: 40px;")
        self.selection_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.selection_title)
        
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(60)
        panels_layout.setAlignment(Qt.AlignCenter)
        
        # Player 1 Panel
        self.p1_panel = self.create_player_panel(1)
        panels_layout.addWidget(self.p1_panel)
        
        # VS Text
        self.vs_label = QLabel("VS")
        self.vs_label.setStyleSheet("font-size: 64px; font-weight: 900; color: #ff9eb0;")
        panels_layout.addWidget(self.vs_label)
        
        # Player 2 Panel
        self.p2_panel = self.create_player_panel(2)
        panels_layout.addWidget(self.p2_panel)
        
        layout.addLayout(panels_layout)
        
        # Action Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter)

        # Back Button
        self.btn_back_menu = QPushButton("🚪 BACK")
        self.btn_back_menu.setObjectName("modeBtn") 
        self.btn_back_menu.setFixedSize(200, 70)
        self.btn_back_menu.setStyleSheet("""
            QPushButton { background-color: #a4b0be; border: 3px solid #747d8c; }
            QPushButton:hover { background-color: #747d8c; border: 3px solid #57606f; }
        """)
        self.btn_back_menu.clicked.connect(self.back_to_menu)

        # Start Button
        self.btn_start_game = QPushButton("START RACE")
        self.btn_start_game.setObjectName("modeBtn") 
        self.btn_start_game.setFixedSize(320, 70)
        self.btn_start_game.clicked.connect(lambda: self.enter_game(self.game_mode))
        
        btn_layout.addWidget(self.btn_back_menu)
        btn_layout.addWidget(self.btn_start_game)
        
        layout.addSpacing(50)
        layout.addLayout(btn_layout)
        
        return page

    def back_to_menu(self):
        self.stacked_widget.setCurrentWidget(self.menu_page)

    def create_player_panel(self, player_num):
        panel = QWidget()
        panel.setObjectName("selectionPanel")
        panel.setFixedSize(450, 420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel(f"PLAYER {player_num}" if player_num == 1 else "PLAYER 2")
        title.setStyleSheet("font-size: 28px; font-weight: 900; color: #ff6b81; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        char_layout = QHBoxLayout()
        char_layout.setSpacing(30)
        
        # Male
        btn_male = QPushButton()
        btn_male.setFixedSize(170, 230)
        btn_male.setObjectName("charBtn")
        m_vbox = QVBoxLayout(btn_male)
        m_img = QLabel()
        m_img.setPixmap(QPixmap(os.path.join(IMAGE_DIR, "cowok", "neutral.png")).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        m_img.setAlignment(Qt.AlignCenter)
        m_text = QLabel("COWOK")
        m_text.setStyleSheet("font-weight: 900; font-size: 16px; background: transparent; color: #4a4a4a;")
        m_text.setAlignment(Qt.AlignCenter)
        m_vbox.addWidget(m_img)
        m_vbox.addWidget(m_text)
        btn_male.clicked.connect(lambda: self.select_gender(player_num, "cowok"))
        
        # Female
        btn_female = QPushButton()
        btn_female.setFixedSize(170, 230)
        btn_female.setObjectName("charBtn")
        f_vbox = QVBoxLayout(btn_female)
        f_img = QLabel()
        f_img.setPixmap(QPixmap(os.path.join(IMAGE_DIR, "cewek", "neutral.jpeg")).scaled(110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        f_img.setAlignment(Qt.AlignCenter)
        f_text = QLabel("CEWEK")
        f_text.setStyleSheet("font-weight: 900; font-size: 16px; background: transparent; color: #4a4a4a;")
        f_text.setAlignment(Qt.AlignCenter)
        f_vbox.addWidget(f_img)
        f_vbox.addWidget(f_text)
        btn_female.clicked.connect(lambda: self.select_gender(player_num, "cewek"))
        
        char_layout.addWidget(btn_male)
        char_layout.addWidget(btn_female)
        layout.addLayout(char_layout)
        
        for btn in [btn_male, btn_female]:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(255, 182, 193, 100))
            shadow.setOffset(0, 5)
            btn.setGraphicsEffect(shadow)

        if player_num == 1:
            self.p1_btns = {"cowok": btn_male, "cewek": btn_female}
        else:
            self.p2_btns = {"cowok": btn_male, "cewek": btn_female}
            self.p2_title = title
            
        return panel

    def select_gender(self, player_num, gender):
        self.load_player_assets(player_num, gender)
        
        btns = self.p1_btns if player_num == 1 else self.p2_btns
        for g, btn in btns.items():
            btn.setObjectName("charBtnActive" if g == gender else "charBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            
        if player_num == 1 and getattr(self, 'game_mode', 'PVP') == "AI":
            ai_gender = "cewek" if gender == "cowok" else "cowok"
            self.load_player_assets(2, ai_gender)
            for g, btn in self.p2_btns.items():
                btn.setObjectName("charBtnActive" if g == ai_gender else "charBtn")
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    def start_selection(self, mode):
        self.game_mode = mode
        
        self.vs_label.setVisible(True)
        self.p2_panel.setVisible(True)
        
        if mode == "PVP":
            self.p2_title.setText("PLAYER 2")
            for btn in self.p2_btns.values(): btn.setEnabled(True)
            self.select_gender(1, "cowok")
            self.select_gender(2, "cewek")
        else:
            self.p2_title.setText("AI (AUTO)")
            for btn in self.p2_btns.values(): btn.setEnabled(False)
            self.select_gender(1, "cowok")
            
        self.stacked_widget.setCurrentWidget(self.selection_page)

    def enter_game(self, mode):
        self.game_mode = mode
        self.reset()
        self.stacked_widget.setCurrentWidget(self.game_page)

    def create_game_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.race_label = QLabel()
        self.race_label.setFixedHeight(120)
        self.race_label.setObjectName("raceLabel")
        self.race_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        btn_play = QPushButton("▶ Play")
        btn_play.setObjectName("actionBtn")
        btn_pause = QPushButton("⏸ Pause")
        btn_pause.setObjectName("actionBtn")
        btn_reset = QPushButton("🔄 Reset")
        btn_reset.setObjectName("actionBtn")
        btn_capture = QPushButton("📷 Capture")
        btn_capture.setObjectName("actionBtn")
        btn_back = QPushButton("🚪 Back to Menu")
        btn_back.setObjectName("actionBtn")

        btn_play.clicked.connect(self.start)
        btn_pause.clicked.connect(self.stop)
        btn_reset.clicked.connect(self.reset)
        btn_capture.clicked.connect(self.open_capture_window)
        btn_back.clicked.connect(self.exit_game)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        btn_layout.addWidget(btn_play)
        btn_layout.addWidget(btn_pause)
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(btn_capture)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_back)

        main_container = QWidget()
        main_container.setObjectName("mainContainer")
        container_layout = QVBoxLayout(main_container)
        container_layout.setContentsMargins(70, 25, 70, 25)
        container_layout.addWidget(self.video_label)
        container_layout.addLayout(btn_layout)

        layout.addWidget(self.race_label)
        layout.addWidget(main_container)

        for widget in [self.race_label, main_container]:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(255, 182, 193, 100))
            shadow.setOffset(0, 5)
            widget.setGraphicsEffect(shadow)

        self.update_race()
        return page

    def exit_game(self):
        self.stop()
        self.stacked_widget.setCurrentWidget(self.menu_page)

    def create_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 40, 50, 40)
        layout.setSpacing(20)

        title = QLabel("📷 CAMERA SETTINGS")
        title.setStyleSheet("font-size: 42px; font-weight: 900; color: #ff6b81; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Main Panel
        panel = QWidget()
        panel.setObjectName("selectionPanel")
        panel.setMinimumWidth(800)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(30, 30, 30, 30)
        panel_layout.setSpacing(25)

        # Camera selection row
        select_layout = QHBoxLayout()
        select_label = QLabel("Select Camera:")
        select_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4a4a4a;")
        
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(45)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                padding: 5px 15px;
                border: 3px solid #ffb6c1;
                border-radius: 12px;
                background-color: white;
                color: #4a4a4a;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                border: 3px solid #ffb6c1;
                selection-background-color: #ffb6c1;
                selection-color: white;
                background-color: white;
                color: #4a4a4a;
            }
        """)
        
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        select_layout.addWidget(select_label)
        select_layout.addWidget(self.camera_combo, 1)
        panel_layout.addLayout(select_layout)

        # Settings Live Preview Label
        self.settings_preview_label = QLabel()
        self.settings_preview_label.setFixedSize(640, 360)
        self.settings_preview_label.setObjectName("videoLabel")
        self.settings_preview_label.setAlignment(Qt.AlignCenter)
        self.settings_preview_label.setStyleSheet("background-color: #f0f0f0; border-radius: 15px; border: 4px dashed #87cefa;")
        panel_layout.addWidget(self.settings_preview_label, 0, Qt.AlignCenter)

        # Guide Text
        guide_text = QLabel("💡 Tip: Make sure your full body is visible in the frame, especially your feet!")
        guide_text.setStyleSheet("font-size: 16px; color: #747d8c; font-style: italic;")
        guide_text.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(guide_text)

        layout.addWidget(panel)

        # Back Button
        btn_back = QPushButton("💾 SAVE & BACK")
        btn_back.setObjectName("modeBtn")
        btn_back.setFixedSize(240, 60)
        btn_back.setStyleSheet("""
            QPushButton { background-color: #a8e6cf; border: 3px solid #8be3c4; }
            QPushButton:hover { background-color: #8be3c4; border: 3px solid #75cfb0; }
        """)
        btn_back.clicked.connect(self.back_to_menu)
        layout.addWidget(btn_back, 0, Qt.AlignHCenter)

        # Shadow for settings panel
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(255, 182, 193, 100))
        shadow.setOffset(0, 5)
        panel.setGraphicsEffect(shadow)

        return page

    def refresh_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        
        devices = QMediaDevices.videoInputs()
        if not devices:
            self.camera_combo.addItem("No Camera Detected", 0)
        else:
            for i, dev in enumerate(devices):
                description = dev.description()
                if not description:
                    description = f"Camera {i}"
                self.camera_combo.addItem(description, i)
                
        # Set selection to current index
        idx = self.camera_combo.findData(getattr(self, "camera_index", 0))
        if idx != -1:
            self.camera_combo.setCurrentIndex(idx)
        else:
            self.camera_combo.setCurrentIndex(0)
            
        self.camera_combo.blockSignals(False)

    def change_camera(self, index):
        cam_idx = self.camera_combo.itemData(index)
        if cam_idx is None:
            return
        
        # Save previous camera index in case the new one fails to open
        prev_idx = getattr(self, "camera_index", 0)
        if hasattr(self, "camera_index") and self.camera_index == cam_idx and self.cap is not None and self.cap.isOpened():
            return
            
        self.camera_index = cam_idx
        
        # Re-initialize camera
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Failed to open camera index {self.camera_index}, falling back to {prev_idx}")
            self.camera_index = prev_idx
            self.cap = cv2.VideoCapture(self.camera_index)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #fdfbf7;
                font-family: 'Comic Sans MS', 'Fredoka One', 'Varela Round', 'Segoe UI', sans-serif;
                color: #4a4a4a;
            }
            #raceLabel, #mainContainer {
                background-color: #ffffff;
                border: 3px solid #ffb6c1;
                border-radius: 20px;
            }
            #videoLabel {
                background-color: #f0f0f0;
                border-radius: 15px;
                border: 4px dashed #87cefa;
            }
            QPushButton {
                color: #ffffff;
                font-weight: bold;
                border-radius: 15px;
            }
            QPushButton#modeBtn {
                background-color: #ffb6c1;
                border: 3px solid #ff9eb0;
                padding: 15px 30px;
                font-size: 20px;
                color: white;
            }
            QPushButton#modeBtn:hover {
                background-color: #ff9eb0;
                border: 3px solid #ff7a93;
            }
            QPushButton#actionBtn {
                background-color: #87cefa;
                border: 3px solid #63b8ff;
                padding: 8px 15px;
                font-size: 16px;
                border-radius: 12px;
            }
            QPushButton#actionBtn:hover {
                background-color: #63b8ff;
            }
            QPushButton#charBtn {
                background-color: #ffffff;
                border: 4px solid #e0e0e0;
                border-radius: 25px;
                color: #4a4a4a;
            }
            QPushButton#charBtn:hover {
                background-color: #f0f8ff;
                border: 4px solid #87cefa;
            }
            QPushButton#charBtnActive {
                background-color: #e6f2ff;
                border: 4px solid #00a8ff;
                border-radius: 25px;
                color: #4a4a4a;
            }
            #selectionPanel {
                background-color: #ffffff;
                border: 4px solid #ffd1dc;
                border-radius: 30px;
            }
            QLabel#mainTitle {
                color: #ff6b81;
                font-size: 72px;
                font-weight: 900;
            }
            QLabel#subTitle {
                color: #a4b0be;
                font-size: 24px;
                font-weight: bold;
            }
        """)

    def overlay_icon(self, frame, img_path, x, y, size=220):
        icon = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            return frame

        icon = cv2.resize(icon, (size, size))
        h, w = icon.shape[:2]

        x = int(x - w // 2)
        y = int(y - h // 2)

        if icon.shape[2] == 4:
            alpha = icon[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * icon[:, :, c] +
                    (1 - alpha) * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = icon

        return frame

    def open_capture_window(self):
        self.capture_window = CaptureWindow(self)
        self.capture_window.show()

    def start(self):
        if self.winner:
            return

        self.reset()
        self.countdown = 3
        self.countdown_timer.start(1000)

    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
        else:
            self.countdown_timer.stop()
            self.countdown = None

            now = time.time()
            self.left_start_time = now
            self.right_start_time = now

            self.is_playing = True

    def stop(self):
        self.is_playing = False

    def reset(self):
        self.left_score = 0
        self.right_score = 0
        self.winner = None
        self.data_store.clear()
        self.history_y.clear()

        self.left_start_time = None
        self.right_start_time = None
        self.left_time = 0
        self.right_time = 0
        self.is_playing = False

        self.update_race()

    def update_race(self):
        w = self.race_label.width()
        h = self.race_label.height()
        if w <= 0 or h <= 0: return

        pix = QPixmap(w, h)
        pix.fill(Qt.transparent)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        p1 = min(1.0, self.left_score / FINISH_SCORE)
        p2 = min(1.0, self.right_score / FINISH_SCORE)

        y1, y2 = 40, 80
        bar_h = 14

        # Background Bars
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(230, 230, 230))
        painter.drawRoundedRect(20, y1 - bar_h // 2, w - 40, bar_h, 7, 7)
        painter.drawRoundedRect(20, y2 - bar_h // 2, w - 40, bar_h, 7, 7)

        # Player 1 Bar (Cute Blue)
        grad1 = QLinearGradient(20, 0, w - 20, 0)
        grad1.setColorAt(0, QColor(135, 206, 250))
        grad1.setColorAt(p1 if p1 > 0 else 0.01, QColor(0, 168, 255))
        grad1.setColorAt(1, QColor(135, 206, 250, 100))
        
        painter.setBrush(grad1)
        painter.drawRoundedRect(20, y1 - bar_h // 2, int((w - 40) * p1), bar_h, 7, 7)

        # Player 2 Bar (Cute Pink)
        grad2 = QLinearGradient(20, 0, w - 20, 0)
        grad2.setColorAt(0, QColor(255, 182, 193))
        grad2.setColorAt(p2 if p2 > 0 else 0.01, QColor(255, 107, 129))
        grad2.setColorAt(1, QColor(255, 182, 193, 100))

        painter.setBrush(grad2)
        painter.drawRoundedRect(20, y2 - bar_h // 2, int((w - 40) * p2), bar_h, 7, 7)

        # Icons
        if self.p1_data["running"]:
            icon1 = self.p1_data["running"][self.current_frame_idx]
        else:
            icon1 = self.p1_data["neutral"]

        if self.p2_data["running"]:
            icon2 = self.p2_data["running"][self.current_frame_idx]
        else:
            icon2 = self.p2_data["neutral"]

        icon1_px = icon1.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon2_px = icon2.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter.drawPixmap(int(20 + (w - 40) * p1) - icon1_px.width() // 2, y1 - icon1_px.height() // 2, icon1_px)
        painter.drawPixmap(int(20 + (w - 40) * p2) - icon2_px.width() // 2, y2 - icon2_px.height() // 2, icon2_px)

        # Labels
        painter.setPen(QColor(74, 74, 74))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(20, y1 - 15, f"PLAYER 1: {self.left_score}")
        right_name = "AI OPPONENT" if getattr(self, 'game_mode', 'PVP') == "AI" else "PLAYER 2"
        painter.drawText(20, y2 - 15, f"{right_name}: {self.right_score}")

        painter.end()
        self.race_label.setPixmap(pix)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        # If settings page is active, show raw preview with guidelines and return
        if self.stacked_widget.currentWidget() == self.settings_page:
            frame_preview = cv2.flip(frame, 1)
            h_p, w_p, _ = frame_preview.shape
            
            # Guide lines (vertical pink lines, horizontal blue line)
            cv2.line(frame_preview, (int(w_p * 0.25), 0), (int(w_p * 0.25), h_p), (255, 182, 193), 2, cv2.LINE_AA)
            cv2.line(frame_preview, (int(w_p * 0.75), 0), (int(w_p * 0.75), h_p), (255, 182, 193), 2, cv2.LINE_AA)
            cv2.line(frame_preview, (0, int(h_p * 0.7)), (w_p, int(h_p * 0.7)), (135, 206, 250), 2, cv2.LINE_AA)
            
            # Text on preview
            cv2.putText(frame_preview, "ALIGN FEET BELOW THIS LINE", (20, int(h_p * 0.65)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (135, 206, 250), 2, cv2.LINE_AA)
            cv2.putText(frame_preview, "STAND IN THE MIDDLE", (int(w_p * 0.32), 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 182, 193), 2, cv2.LINE_AA)

            rgb = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, w_p, h_p, 3*w_p, QImage.Format_RGB888)
            self.settings_preview_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.settings_preview_label.width(), 
                self.settings_preview_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            return

        # Update animation indices for both players
        self.current_frame_idx += 1
        if self.p1_data["running"] and self.current_frame_idx >= len(self.p1_data["running"]):
            self.current_frame_idx = 0
        if self.p2_data["running"] and self.current_frame_idx >= len(self.p2_data["running"]):
            self.current_frame_idx = 0
            
        self.update_race()

        # AI LOGIC
        if getattr(self, 'game_mode', 'PVP') == "AI" and self.is_playing:
            now = time.time()
            if now - self.ai_last_jump_time > self.ai_target_cooldown:
                self.right_score += 1
                self.snd_jump.play()
                self.update_race()
                self.ai_last_jump_time = now
                self.ai_jump_anim_start = now # Trigger animasi visual
                
                # Dynamic Difficulty (Rubber Banding)
                diff = self.left_score - self.right_score
                base = random.uniform(0.7, 1.2) # AI naturally jumps every 0.7-1.2s
                adjustment = diff * 0.1 # if player leads by 5, adjustment = 0.5s faster
                self.ai_target_cooldown = max(0.4, base - adjustment)
                
                if self.right_score >= FINISH_SCORE:
                    self.winner = "AI OPPONENT WIN"
                    self.snd_finish.play()
                    self.stop()

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mid = w // 2

        # update waktu
        if self.left_start_time is not None and self.winner is None:
            self.left_time = time.time() - self.left_start_time
        if self.right_start_time is not None and self.winner is None:
            self.right_time = time.time() - self.right_start_time

        results = model.predict(frame, conf=0.5, device=device)

        # Adaptive brightness calculation (OpenCV mean is very fast)
        avg_brightness = cv2.mean(frame)[0]
        text_color = (0, 0, 0) if avg_brightness > 127 else (255, 255, 255)

        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].keypoints.conf.cpu().numpy()

            players = []

            # Extract boxes for depth/size estimation
            boxes = None
            if results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()

            players = []

            # ambil semua kandidat player (berdasarkan kaki agar lebih fokus sesuai permintaan)
            for i, kp in enumerate(keypoints):
                la, ra = kp[15], kp[16]
                la_conf, ra_conf = confidences[i][15], confidences[i][16]
                
                # Cek apakah kaki (ankle) terdeteksi dengan confidence cukup
                has_feet = (la_conf > 0.4 and not (la == 0).all()) or (ra_conf > 0.4 and not (ra == 0).all())
                
                if has_feet:
                    if (la_conf > 0.4 and not (la == 0).all()) and (ra_conf > 0.4 and not (ra == 0).all()):
                        # Pilih kaki yang posisinya paling bawah (Y paling besar) agar lompatan stabil
                        # (Mencegah fake jump dengan hanya mengangkat 1 kaki)
                        if la[1] > ra[1]:
                            cx, cy = la
                            conf = la_conf
                        else:
                            cx, cy = ra
                            conf = ra_conf
                    elif la_conf > 0.4 and not (la == 0).all():
                        cx, cy = la
                        conf = la_conf
                    else:
                        cx, cy = ra
                        conf = ra_conf
                else:
                    # Fallback ke pinggul (hip) jika kaki tidak terdeteksi
                    lh, rh = kp[11], kp[12]
                    if not (lh == 0).all() and not (rh == 0).all():
                        cx = (lh[0] + rh[0]) / 2
                        cy = (lh[1] + rh[1]) / 2
                        conf = (confidences[i][11] + confidences[i][12]) / 2
                    elif not (lh == 0).all():
                        cx, cy = lh
                        conf = confidences[i][11]
                    elif not (rh == 0).all():
                        cx, cy = rh
                        conf = confidences[i][12]
                    else:
                        cx, cy, conf = 0, 0, 0
                
                if cx == 0 and cy == 0:
                    continue
                    
                # Ukuran (height) dari bounding box untuk mengetahui siapa yang paling dekat dengan kamera
                size = boxes[i][3] if boxes is not None else 0
                
                players.append({
                    "cx": cx,
                    "cy": cy,
                    "conf": conf,
                    "size": size,
                    "has_feet": has_feet
                })

            # Pisahkan kandidat berdasarkan posisi kiri dan kanan
            left_candidates = [p for p in players if p["cx"] < mid]
            right_candidates = [p for p in players if p["cx"] >= mid]

            # Pilih player yang ukurannya paling besar (paling dekat kamera / fokus ke user) di tengah keramaian
            left_player = max(left_candidates, key=lambda p: p["size"]) if left_candidates else None
            right_player = max(right_candidates, key=lambda p: p["size"]) if right_candidates else None

            tracked = {
                "left": left_player,
                "right": right_player
            }

            for side, player_data in tracked.items():
                if getattr(self, 'game_mode', 'PVP') == "AI" and side == "right":
                    continue
                if player_data is None:
                    continue

                cx, cy, conf = player_data["cx"], player_data["cy"], player_data["conf"]
                has_feet = player_data["has_feet"]
                
                # Jika kaki tidak terdeteksi, tampilkan alert peringatan dan jangan proses lompatan
                if not has_feet:
                    alert_text = "PERINGATAN: Kaki Tidak Terdeteksi!"
                    text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Ensure alert text is within frame bounds
                    alert_x = max(10, min(int(cx - text_size[0] // 2), w - text_size[0] - 10))
                    alert_y = max(30, int(cy - 50))
                    
                    # Draw shadow for readability
                    cv2.putText(frame, alert_text, (alert_x+1, alert_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(frame, alert_text, (alert_x, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Tetap gambar indikator targetnya
                    cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                    continue

                self.history_y[side].append(cy)
                if len(self.history_y[side]) < HISTORY_LEN:
                    continue

                cy = sum(self.history_y[side]) / len(self.history_y[side])

                # Draw Confidence Bar
                bar_w = 60
                bar_h = 6
                bx, by = int(cx - bar_w // 2), int(cy - 25)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                bar_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                cv2.rectangle(frame, (bx, by), (bx + int(bar_w * conf), by + bar_h), bar_color, -1)
                
                # Glow effect for detected point
                cv2.circle(frame, (int(cx), int(cy)), 12, (text_color[0], text_color[1], text_color[2]), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, bar_color, -1)

                if side not in self.data_store:
                    self.data_store[side] = {
                        "ground_y": cy,
                        "state": "ground",
                        "peak_y": cy
                    }
                    continue

                d = self.data_store[side]

                if d["state"] == "ground":
                    # Adapt ground_y
                    if cy > d["ground_y"]:
                        d["ground_y"] = cy # Segera update jika kaki kembali menyentuh tanah (Y besar)
                    else:
                        d["ground_y"] = d["ground_y"] * 0.95 + cy * 0.05 # Perlahan ikuti gerakan kecil
                        
                    if cy < d["ground_y"] - MIN_RISE:
                        d["state"] = "up"
                        d["peak_y"] = cy

                elif d["state"] == "up":
                    if cy < d["peak_y"]:
                        d["peak_y"] = cy

                    if cy > d["peak_y"] + MIN_FALL:
                        # Hitung tinggi lompatan dari referensi ground_y yang stabil, bukan dari cy sesaat
                        jump_height = d["ground_y"] - d["peak_y"]

                        if jump_height > MIN_JUMP and self.is_playing:
                            self.snd_jump.play()

                            if side == "left":
                                self.left_score += 1
                            else:
                                self.right_score += 1

                            self.update_race()

                            if self.left_score >= FINISH_SCORE:
                                self.winner = "PLAYER 1 WIN"
                                self.snd_finish.play()
                                self.stop()

                            elif self.right_score >= FINISH_SCORE:
                                self.winner = "AI OPPONENT WIN" if getattr(self, 'game_mode', 'PVP') == "AI" else "PLAYER 2 WIN"
                                self.snd_finish.play()
                                self.stop()

                        d["state"] = "ground"

                # titik merah tetap muncul
                cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)

        # adaptive text color logic
        avg_brightness = cv2.mean(frame)[0]
        text_color = (0, 0, 0) if avg_brightness > 127 else (255, 255, 255)

        # RENDER AI CHARACTER IN VS AI MODE (Filling the right side)
        if getattr(self, 'game_mode', 'PVP') == "AI" and not self.winner:
            ai_x = int(w * 0.75)
            ai_y_base = int(h * 0.6)
            ai_y = ai_y_base
            
            ai_img = self.p2_data["neutral_path"]
            
            # Animasi Lompat Parabola
            time_since_jump = time.time() - self.ai_jump_anim_start
            if time_since_jump < self.ai_jump_duration:
                # Progress 0.0 -> 1.0
                t = time_since_jump / self.ai_jump_duration
                # Rumus parabola: y = 4 * height * t * (1-t)
                jump_height = 150 
                offset = int(jump_height * 4 * t * (1 - t))
                ai_y -= offset
                ai_img = self.p2_data["happy"] # Ganti muka jadi seneng pas lompat
            
            # Gambar karakter AI
            frame = self.overlay_icon(frame, ai_img, ai_x, ai_y, 300)
            
            # Label AI
            cv2.putText(frame, "AI OPPONENT", (ai_x - 80, ai_y_base + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # countdown
        if self.countdown is not None:
            text = str(self.countdown + 1) if self.countdown > 0 else "GO!"
            # Shadow for countdown text
            cv2.putText(frame, text, (int(w*0.4)+4, int(h*0.6)+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
            cv2.putText(frame, text, (int(w*0.4), int(h*0.6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, text_color, 8)

        # timer kiri kanan
        if self.left_start_time is not None:
            left_text = f"{self.left_time:.3f}s"
            right_text = f"{self.right_time:.3f}s"

            cv2.putText(frame, left_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            text_size = cv2.getTextSize(right_text,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

            cv2.putText(frame, right_text,
                        (w - text_size[0] - 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            # =========================
            # 🔥 WINNER CHECK + DISPLAY (TARUH DI SINI)
            # =========================
            if not self.winner:
                if self.left_score >= FINISH_SCORE:
                    self.winner = "PLAYER 1 WIN"
                elif self.right_score >= FINISH_SCORE:
                    self.winner = "AI OPPONENT WIN" if getattr(self, 'game_mode', 'PVP') == "AI" else "PLAYER 2 WIN"

            if self.winner:
                # Slightly lower the character to make room for text above
                left_pos = (int(w * 0.25), int(h * 0.55))
                right_pos = (int(w * 0.75), int(h * 0.55))
                char_size = 400

                def draw_status_text(text, center_pos, is_win):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 2.5
                    thickness = 6
                    
                    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
                    tx = center_pos[0] - text_size[0] // 2
                    ty = center_pos[1] - (char_size // 2) - 10
                    
                    # Colored border/shadow to differentiate WIN/LOSE
                    shadow_color = (0, 255, 0) if is_win else (0, 0, 255)
                    cv2.putText(frame, text, (tx, ty), font, scale, shadow_color, thickness + 8)
                    cv2.putText(frame, text, (tx, ty), font, scale, text_color, thickness)

                if self.winner == "PLAYER 1 WIN":
                    frame = self.overlay_icon(frame, self.p1_data["happy"], *left_pos, char_size)
                    frame = self.overlay_icon(frame, self.p2_data["sad"], *right_pos, char_size)

                    draw_status_text("WIN", left_pos, True)
                    draw_status_text("LOSE", right_pos, False)
                else:
                    frame = self.overlay_icon(frame, self.p1_data["sad"], *left_pos, char_size)
                    frame = self.overlay_icon(frame, self.p2_data["happy"], *right_pos, char_size)

                    draw_status_text("LOSE", left_pos, False)
                    draw_status_text("WIN", right_pos, True)

        cv2.line(frame, (mid,0), (mid,h), (255,255,255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(img))

app = QApplication(sys.argv)
window = JumpApp()
window.showMaximized()
sys.exit(app.exec())