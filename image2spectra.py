#!/usr/bin/env python3
"""
Antti's FFT/IFFT Visual Tool
- Loads an image (grayscale)
- Computes row-wise 1D FFT (rfft) per row -> "cochlea map"
- Displays original image and cochlea map
- Save/load full complex spectra (real + imag arrays + metadata) so exact inverse is possible
- Reconstructs image via irfft and shows/saves result

Requirements:
    pip install pyqt6 pyqtgraph numpy pillow scipy

Run:
    python antti_cochlea_gui.py
"""

import sys
import os
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PIL import Image
from scipy.signal import windows
from scipy.fft import rfft, irfft

pg.setConfigOptions(imageAxisOrder='row-major')  # makes setImage use (rows, cols) naturally


class AnttiCochleaApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antti's FFT / IFFT Visual Cochlea")
        self.resize(1200, 700)

        # data holders
        self.image = None           # original grayscale float32 (rows x cols) [0..1]
        self.spectra = None         # complex spectra (rows x freq_bins) as complex128
        self.window = None          # window used (if any)
        self.n_fft = None

        # UI
        self._build_ui()

        # timers / misc
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(50)

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
        btn_load_img = QtWidgets.QPushButton("Load Image...")
        btn_load_img.clicked.connect(self.load_image)
        btn_compute = QtWidgets.QPushButton("Compute Row-wise FFT")
        btn_compute.clicked.connect(self.compute_spectra)
        btn_save_spec = QtWidgets.QPushButton("Save Spectra...")
        btn_save_spec.clicked.connect(self.save_spectra)
        btn_load_spec = QtWidgets.QPushButton("Load Spectra...")
        btn_load_spec.clicked.connect(self.load_spectra)
        btn_recon = QtWidgets.QPushButton("Reconstruct Image from Spectra")
        btn_recon.clicked.connect(self.reconstruct_image)
        btn_save_recon = QtWidgets.QPushButton("Save Reconstructed Image...")
        btn_save_recon.clicked.connect(self.save_reconstructed_image)

        # Options
        self.chk_window = QtWidgets.QCheckBox("Apply Hann window (per row)")
        self.chk_window.setChecked(True)
        self.chk_log = QtWidgets.QCheckBox("Display cochlea map (log scale)")
        self.chk_log.setChecked(True)

        # Info label
        self.info_label = QtWidgets.QLabel("No image loaded.")

        # pack controls
        for w in (btn_load_img, btn_compute, btn_save_spec, btn_load_spec, btn_recon, btn_save_recon,
                  self.chk_window, self.chk_log, self.info_label):
            ctrl_layout.addWidget(w)

        left_v.addWidget(ctrl_box, 0)

        # Original Image view
        self.view_orig = pg.ImageView()
        self.view_orig.ui.histogram.hide()
        self.view_orig.ui.roiBtn.hide()
        left_v.addWidget(QtWidgets.QLabel("Original (grayscale)"))
        left_v.addWidget(self.view_orig, 1)

        # ---------------- Right: Cochlea view + Reconstructed ----------------
        right_v.addWidget(QtWidgets.QLabel("Cochlea map (rows vs horizontal frequency)"))
        self.view_cochlea = pg.ImageView()
        self.view_cochlea.ui.histogram.hide()
        self.view_cochlea.ui.roiBtn.hide()
        right_v.addWidget(self.view_cochlea, 3)

        right_v.addWidget(QtWidgets.QLabel("Reconstructed image (from spectra)"))
        self.view_recon = pg.ImageView()
        self.view_recon.ui.histogram.hide()
        self.view_recon.ui.roiBtn.hide()
        right_v.addWidget(self.view_recon, 3)

        # status bar style
        self.info_label.setWordWrap(True)

    # ----------------- Utilities -----------------
    def _set_info(self, txt):
        self.info_label.setText(txt)

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
        self.n_fft = cols  # using full width for FFT (keeps invertible easily)

        # optional window
        if self.chk_window.isChecked():
            w = windows.hann(cols, sym=False).astype(np.float64)
            self.window = w
            data = (self.image * w[np.newaxis, :]).astype(np.float64)
        else:
            self.window = None
            data = self.image.astype(np.float64)

        # compute rfft per row -> shape (rows, n_fft//2 + 1)
        spec = rfft(data, axis=1)
        self.spectra = spec  # complex128

        # magnitude for display
        mag = np.abs(spec)
        if self.chk_log.isChecked():
            display = np.log1p(mag)
        else:
            display = mag

        # normalize display to 0..1 for nicer viewing
        display = display.astype(np.float32)
        disp_min, disp_max = display.min(), display.max()
        if disp_max > disp_min:
            display = (display - disp_min) / (disp_max - disp_min)
        else:
            display = np.zeros_like(display)

        # set image (transpose so Y is vertical)
        self.view_cochlea.setImage(display.T, autoLevels=False, levels=(0, 1))
        self.view_recon.clear()
        self._set_info(f"Computed spectra: rows={rows} cols={cols} freqs={spec.shape[1]} (saved as rfft)")

    def save_spectra(self):
        if self.spectra is None:
            QtWidgets.QMessageBox.warning(self, "No spectra", "Compute or load spectra first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save spectra (.npz)", "spectra.npz", "NPZ (*.npz)")
        if not path:
            return
        # We'll save real and imag to avoid complex object issues, plus metadata and window if present
        try:
            np.savez_compressed(
                path,
                real=self.spectra.real.astype(np.float32),
                imag=self.spectra.imag.astype(np.float32),
                n_fft=int(self.n_fft if self.n_fft is not None else self.image.shape[1]),
                window=(self.window.astype(np.float32) if self.window is not None else None),
                orig_shape=(self.image.shape if self.image is not None else None)
            )
            self._set_info(f"Spectra saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Couldn't save spectra:\n{e}")

    def load_spectra(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open spectra (.npz)", "", "NPZ (*.npz)")
        if not path:
            return
        try:
            z = np.load(path, allow_pickle=True)
            real = z['real'].astype(np.float64)
            imag = z['imag'].astype(np.float64)
            spec = real + 1j * imag
            self.spectra = spec
            self.n_fft = int(z['n_fft'].tolist()) if 'n_fft' in z else spec.shape[1]*2 - 2
            win = z.get('window', None)
            self.window = (np.array(win) if win is not None and win is not None else None)
            orig_shape = z.get('orig_shape', None)
            self._set_info(f"Loaded spectra from {os.path.basename(path)}  spec.shape={spec.shape} n_fft={self.n_fft} orig_shape={orig_shape}")
            # show preview map
            mag = np.abs(self.spectra)
            display = np.log1p(mag) if self.chk_log.isChecked() else mag
            display = display.astype(np.float32)
            disp_min, disp_max = display.min(), display.max()
            if disp_max > disp_min:
                display = (display - disp_min) / (disp_max - disp_min)
            else:
                display = np.zeros_like(display)
            self.view_cochlea.setImage(display.T, autoLevels=False, levels=(0, 1))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Couldn't load spectra:\n{e}")

    def reconstruct_image(self):
        if self.spectra is None:
            QtWidgets.QMessageBox.warning(self, "No spectra", "Compute or load spectra first.")
            return

        # inverse rfft per row
        try:
            recon = irfft(self.spectra, n=self.n_fft, axis=1)  # rows x n_fft
            # if window was applied during forward, undo it (divide), careful with zeros
            if self.window is not None:
                w = self.window
                # avoid tiny divisors
                eps = 1e-9
                denom = w[np.newaxis, :].copy()
                denom[denom < eps] = 1.0  # don't divide where window is zero; will be inaccurate there
                recon = recon / denom

            # crop/trim to original width if we know orig shape
            if self.image is not None and recon.shape[1] != self.image.shape[1]:
                recon = recon[:, :self.image.shape[1]]

            # clamp and scale to 0..1
            recon = np.real(recon)
            recon = np.clip(recon, 0.0, 1.0)

            self.view_recon.setImage(recon, autoLevels=False, levels=(0, 1))
            self._set_info(f"Reconstructed image from spectra: shape={recon.shape}")
            self.reconstructed = recon.astype(np.float32)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Reconstruction failed:\n{e}")

    def save_reconstructed_image(self):
        if not hasattr(self, 'reconstructed') or self.reconstructed is None:
            QtWidgets.QMessageBox.warning(self, "No reconstruction", "Reconstruct an image first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save reconstructed image", "reconstructed.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        try:
            img_uint8 = (np.clip(self.reconstructed, 0, 1) * 255.0).astype(np.uint8)
            Image.fromarray(img_uint8).save(path)
            self._set_info(f"Saved reconstructed image to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Couldn't save image:\n{e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AnttiCochleaApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
