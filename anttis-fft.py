import sys
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets, QtGui

class VisualCochlea(QtWidgets.QWidget):
    """
    This tool visualizes the "Cochlea" concept for a 2D image.
    It replaces the confusing 2D FFT with 256 separate 1D FFTs.
    
    - LEFT: The raw webcam image (Position Space).
    - RIGHT: The "Cochlea Map" (Hybrid Space).
        - Y-axis is still Y-Position.
        - X-axis is now HORIZONTAL FREQUENCY.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Visual Cochlea: The Intuitive FFT')
        self.setGeometry(50, 50, 1400, 700)
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("ERROR: Cannot open webcam")
            sys.exit(1)
        
        self.img_size = 256
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)

        # --- Left Panel: Position Space (The Input) ---
        pos_layout = QtWidgets.QVBoxLayout()
        pos_label = QtWidgets.QLabel("LEFT: Position Space (The Webcam Image)\n(Y-Position vs. X-Position)")
        pos_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pos_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        
        self.view_position = pg.ImageView()
        self.view_position.ui.histogram.hide()
        self.view_position.ui.roiBtn.hide()
        self.view_position.ui.menuBtn.hide()
        
        pos_layout.addWidget(pos_label)
        pos_layout.addWidget(self.view_position)
        main_layout.addLayout(pos_layout)

        # --- Right Panel: Cochlea Space (The Output) ---
        cochlea_layout = QtWidgets.QVBoxLayout()
        cochlea_label = QtWidgets.QLabel("RIGHT: Visual Cochlea (The Row-by-Row FFT)\n(Y-Position vs. Horizontal Frequency)")
        cochlea_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cochlea_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        
        self.view_cochlea = pg.ImageView()
        self.view_cochlea.ui.histogram.hide()
        self.view_cochlea.ui.roiBtn.hide()
        self.view_cochlea.ui.menuBtn.hide()
        
        cochlea_layout.addWidget(cochlea_label)
        cochlea_layout.addWidget(self.view_cochlea)
        main_layout.addLayout(cochlea_layout)
        
        # Start update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(33)  # ~30 FPS

    def update_views(self):
        # 1. Capture and preprocess webcam frame
        ret, frame = self.cap.read()
        if not ret:
            return
        
        h, w = frame.shape[:2]
        ch, cw = h // 2, w // 2
        s = self.img_size // 2
        
        gray = cv2.cvtColor(frame[ch-s:ch+s, cw-s:cw+s], cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # 2. --- THIS IS THE MAGIC ---
        # We run a 1D FFT on every single row (axis=1)
        # This is the "cochlea" / "buckets" you described.
        cochlea_map = np.fft.fft(gray, axis=1)
        
        # Shift the zero-frequency (DC) component to the center of each row
        cochlea_map_shifted = np.fft.fftshift(cochlea_map, axes=1)
        
        # Get the brightness (Amplitude)
        # This is our new map!
        display_map = np.log(1 + np.abs(cochlea_map_shifted))

        # 3. Display both
        
        # Left Panel: Show the normal image
        # (We transpose it to match the Y-axis of the plot)
        self.view_position.setImage(gray.T, autoRange=False, autoLevels=False, levels=(0, 1))

        # Right Panel: Show the new "Cochlea Map"
        # (Transposed so Y-axis is vertical)
        self.view_cochlea.setImage(display_map.T)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Dark theme
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    window = VisualCochlea()
    window.show()
    sys.exit(app.exec())
