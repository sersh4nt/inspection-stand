import platform
import subprocess

import cv2
import numpy as np
import qimage2ndarray
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Camera(QObject):
    _DEFAULT_FPS = 60
    new_frame = pyqtSignal(np.ndarray)
    camera_err = pyqtSignal()

    def __init__(self, camera_id=0, mirrored=False, parent=None):
        super(Camera, self).__init__(parent)
        self.mirrored = mirrored
        self.frame = None

        if platform.system() == 'Linux':
            # command = f"v4l2-ctl -d {camera_id} -c exposure_auto=3"
            # subprocess.call(command, shell=True)
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        elif platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._query_frame)
        self.timer.setInterval(1000 // self.fps)
        self.paused = False

    @pyqtSlot()
    def _query_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            self.camera_err.emit()
            return
        if self.mirrored:
            self.frame = cv2.flip(self.frame, 1)
        h, w = self.frame.shape[:2]
        self.frame = cv2.rotate(self.frame[: h - 5, 20: w - 250], cv2.ROTATE_180)
        self.new_frame.emit(self.frame)

    @property
    def paused(self):
        return not self.timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self.timer.stop()
        else:
            self.timer.start()

    @property
    def frame_size(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = self._DEFAULT_FPS
        return fps


class CameraWidget(QLabel):
    new_frame = pyqtSignal(np.ndarray)
    new_frame_signal = pyqtSignal()

    def __init__(self, camera=None, parent=None):
        super(CameraWidget, self).__init__(parent)
        self.frame = None
        self.frame_ = None
        self.camera = camera if camera else None
        self.frame_size = self.camera.frame_size if camera else None
        if camera:
            self.camera.new_frame.connect(self._on_new_frame)

    def initialize(self, camera):
        self.camera = camera
        self.camera.new_frame.connect(self._on_new_frame)
        self.frame_size = self.camera.frame_size

    @pyqtSlot(np.ndarray)
    def _on_new_frame(self, frame):
        self.frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        self.new_frame.emit(self.frame)

    def changeEvent(self, e):
        if e.type() == QEvent.EnabledChange:
            if self.isEnabled():
                self.camera.new_frame.connect(self._on_new_frame)
            else:
                self.camera.new_frame.disconnect(self._on_new_frame)

    @pyqtSlot(np.ndarray)
    def acquire_frame(self, frame):
        self.frame_ = frame
        self.update()

    def paintEvent(self, e):
        if self.frame_ is None:
            return
        w, h = self.width(), self.height()
        scale = max(h / self.frame_.shape[0], w / self.frame_.shape[1])
        frame = cv2.resize(self.frame_, None, fx=scale, fy=scale)
        painter = QPainter(self)
        painter.drawImage(QPoint(0, 0), qimage2ndarray.array2qimage(frame))
