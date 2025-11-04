#!/usr/bin/env python3
"""
Antti's Interactive Cochlea Editor
(Based on image2spectra.py)

Load an image, compute its "Cochlea Map" (row-wise FFT),
and then CLICK and DRAG on the map to "wipe" (zero-out)
frequencies.

The reconstructed image will update in real-time to show
what information you just destroyed.
"""

import sys
import os
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PIL import Image
from scipy.signal import windows
from scipy.fft import rfft, irfft

pg.setConfigOptions(imageAxisOrder='row-major')

class InteractiveCochleaEditor(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antti's Interactive Cochlea Editor (Wipe the Spectra!)")
        self.resize(1400, 800)

        # data holders
        self.image = None
        self.spectra = None # This is the "master" data we will modify
        self.spectra_display = None # This is the log-mag display
        self.window = None
        self.n_fft = None
        self.reconstructed = None
        
        self.brush_size = 10
        self.brush_strength = 1.0 # 1.0 = full wipe, 0.1 = partial

        # UI
        self._build_ui()
        
        self.is_wiping = False

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        left_v = QtWidgets.QVBoxLayout()
        right_v = QtWidgets.QVBoxLayout()
        layout.addLayout(left_v, 3)
        layout.addLayout(right_v, 4)

        # ---------------- Left: Controls + Original Image ----------------
        ctrl_box = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl_box.setLayout(ctrl_layout)

        # Buttons
        btn_load_img = QtWidgets.QPushButton("1. Load Image...")
        btn_load_img.clicked.connect(self.load_image)
        btn_compute = QtWidgets.QPushButton("2. Compute Cochlea Map")
        btn_compute.clicked.connect(self.compute_spectra)
        
        # Options
        self.chk_window = QtWidgets.QCheckBox("Apply Hann window (per row)")
        self.chk_window.setChecked(True)
        self.chk_log = QtWidgets.QCheckBox("Display cochlea map (log scale)")
        self.chk_log.setChecked(True)
        
        # --- NEW Brush Controls ---
        brush_label = QtWidgets.QLabel("--- Spectral Wiping Controls ---")
        brush_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        
        brush_size_label = QtWidgets.QLabel(f"Brush Size: {self.brush_size}")
        self.slider_brush_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(1, 100)
        self.slider_brush_size.setValue(self.brush_size)
        self.slider_brush_size.valueChanged.connect(
            lambda v: (setattr(self, 'brush_size', v), brush_size_label.setText(f"Brush Size: {v}"))
        )
        
        brush_strength_label = QtWidgets.QLabel(f"Wipe Strength: {self.brush_strength*100:.0f}%")
        self.slider_brush_strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_brush_strength.setRange(1, 100)
        self.slider_brush_strength.setValue(int(self.brush_strength * 100))
        self.slider_brush_strength.valueChanged.connect(
            lambda v: (setattr(self, 'brush_strength', v/100.0), brush_strength_label.setText(f"Wipe Strength: {v}%"))
        )
        
        self.label_instructions = QtWidgets.QLabel("3. Click and Drag on the Cochlea Map to Wipe!")
        self.label_instructions.setStyleSheet("color: #4CAF50; font-weight: bold; margin-top: 5px;")

        # Info label
        self.info_label = QtWidgets.QLabel("No image loaded.")

        # pack controls
        for w in (btn_load_img, btn_compute, self.chk_window, self.chk_log,
                  brush_label, brush_size_label, self.slider_brush_size,
                  brush_strength_label, self.slider_brush_strength,
                  self.label_instructions, self.info_label):
            ctrl_layout.addWidget(w)
        ctrl_layout.addStretch()

        left_v.addWidget(ctrl_box, 0)

        # Original Image view
        self.view_orig = pg.ImageView()
        self.view_orig.ui.histogram.hide()
        self.view_orig.ui.roiBtn.hide()
        left_v.addWidget(QtWidgets.QLabel("Original (grayscale)"))
        left_v.addWidget(self.view_orig, 1)

        # ---------------- Right: Cochlea view + Reconstructed ----------------
        right_v.addWidget(QtWidgets.QLabel("Cochlea map (Y-Pos vs. H-Freq) <-- WIPE HERE!"))
        self.view_cochlea = pg.ImageView()
        self.view_cochlea.ui.histogram.hide()
        self.view_cochlea.ui.roiBtn.hide()
        
        # --- FIX 1: Disable the default zoom/pan ---
        self.view_cochlea.view.setMouseEnabled(x=False, y=False)
        
        right_v.addWidget(self.view_cochlea, 3)

        right_v.addWidget(QtWidgets.QLabel("Reconstructed image (from Wiped Spectra)"))
        self.view_recon = pg.ImageView()
        self.view_recon.ui.histogram.hide()
        self.view_recon.ui.roiBtn.hide()
        right_v.addWidget(self.view_recon, 3)

        self.info_label.setWordWrap(True)
        
        # --- FIX 2: Install filter on the scene, not the widget ---
        self.view_cochlea.view.scene().installEventFilter(self)

    # --- Mouse Event Handling for Wiping ---
    
    # --- FIX 3: Use the correct eventFilter logic for GraphicsScene ---
    def eventFilter(self, source, event):
        """Intercept mouse events on the cochlea view for wiping."""
        if source == self.view_cochlea.view.scene():
            if event.type() == QtCore.QEvent.Type.GraphicsSceneMousePress and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.is_wiping = True
                self.wipe_spectra(event.scenePos())
                return True # We handled this event
            
            elif event.type() == QtCore.QEvent.Type.GraphicsSceneMouseRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.is_wiping = False
                return True # We handled this event
            
            elif event.type() == QtCore.QEvent.Type.GraphicsSceneMouseMove and self.is_wiping:
                self.wipe_spectra(event.scenePos())
                return True # We handled this event
        
        # Let parent class handle other events
        return super().eventFilter(source, event)

    # --- FIX 4: Update wipe_spectra to use scene_pos directly ---
    def wipe_spectra(self, scene_pos):
        if self.spectra is None or self.spectra_display is None:
            return

        if scene_pos is None:
            return
            
        # Map from Scene coordinates to ImageItem coordinates (pixels)
        pos_in_item = self.view_cochlea.getImageItem().mapFromScene(scene_pos)
        
        # Note: We transposed the display, so (x, y) in view is (freq, row)
        # --- THIS IS THE FIX ---
        # The axes were swapped. x() is the row, y() is the freq_bin.
        row = int(pos_in_item.x())
        freq_bin = int(pos_in_item.y())
        
        rows, cols = self.spectra.shape
        
        if not (0 <= row < rows and 0 <= freq_bin < cols):
            return

        # Define the "wipe" area
        s = self.brush_size
        y1 = max(0, row - s)
        y2 = min(rows, row + s)
        x1 = max(0, freq_bin - s)
        x2 = min(cols, freq_bin + s)
        
        # 1. Wipe the MASTER spectra data (the complex numbers)
        self.spectra[y1:y2, x1:x2] *= (1.0 - self.brush_strength)
        
        # 2. Wipe the DISPLAY data (the log-mag image)
        #    (We set it to 0 for a clear visual)
        self.spectra_display[y1:y2, x1:x2] = 0.0

        # 3. Update the cochlea view (non-blocking)
        self.view_cochlea.setImage(self.spectra_display.T, autoLevels=False, levels=(0, 1), autoRange=False)

        # 4. Trigger a reconstruction (non-blocking)
        QtCore.QTimer.singleShot(0, self.reconstruct_image)

    # ----------------- Actions -----------------
    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        try:
            pil = Image.open(path).convert("L")  # grayscale
            arr = np.asarray(pil).astype(np.float32) / 255.0
            self.image = arr
            self.view_orig.setImage(self.image, levels=(0, 1))
            self.spectra = None
            self.view_cochlea.clear()
            self.view_recon.clear()
            self._set_info(f"Loaded image: {os.path.basename(path)}  size={self.image.shape}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Couldn't load image:\n{e}")

    def compute_spectra(self):
        if self.image is None:
            QtWidgets.QMessageBox.warning(self, "No image", "Load an image first.")
            return

        rows, cols = self.image.shape
        self.n_fft = cols

        if self.chk_window.isChecked():
            w = windows.hann(cols, sym=False).astype(np.float64)
            self.window = w
            data = (self.image * w[np.newaxis, :]).astype(np.float64)
        else:
            self.window = None
            data = self.image.astype(np.float64)

        spec = rfft(data, axis=1)
        self.spectra = spec.astype(np.complex128) # Store master data

        # Create the display-only version
        mag = np.abs(spec)
        if self.chk_log.isChecked():
            display = np.log1p(mag)
        else:
            display = mag

        display = display.astype(np.float32)
        disp_min, disp_max = display.min(), display.max()
        if disp_max > disp_min:
            display = (display - disp_min) / (disp_max - disp_min)
        else:
            display = np.zeros_like(display)
            
        self.spectra_display = display # Store display data

        self.view_cochlea.setImage(self.spectra_display.T, autoLevels=False, levels=(0, 1))
        self.reconstruct_image() # Reconstruct on initial compute
        self._set_info(f"Computed spectra: shape={spec.shape}. Ready to wipe!")

    def reconstruct_image(self):
        if self.spectra is None:
            # Don't show a warning, as this is called in real-time
            return

        try:
            # Use the MASTER (wiped) spectra data
            recon = irfft(self.spectra, n=self.n_fft, axis=1)
            
            if self.window is not None:
                w = self.window
                eps = 1e-9
                denom = w[np.newaxis, :].copy()
                denom[denom < eps] = 1.0
                recon = recon / denom

            if self.image is not None and recon.shape[1] != self.image.shape[1]:
                recon = recon[:, :self.image.shape[1]]

            recon = np.real(recon)
            recon = np.clip(recon, 0.0, 1.0)

            self.view_recon.setImage(recon, autoLevels=False, levels=(0, 1))
            self.reconstructed = recon.astype(np.float32)
        except Exception as e:
            print(f"Reconstruction failed: {e}") # Log to console instead of popup

    def _set_info(self, txt):
        self.info_label.setText(txt)

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Dark theme
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(42, 42, 42))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(66, 66, 66))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.black)
    
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, QtGui.QColor(127, 127, 127))
    dark_palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(127, 127, 127))
    
    app.setPalette(dark_palette)
    
    win = InteractiveCochleaEditor()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


