import sys

import cv2
import imutils
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from design import Ui_mainWindow
from libs.camera import Camera
from libs.network_handler import NetworkHandler
from libs.yolo.plots import *


class DefectsWindow(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = Camera(0)
        self.main_view.initialize(camera=self.camera)

        self.has_scratches = False
        self.has_holes = False
        self.has_bad_legs = False
        self.has_object = False
        self.cnt_img = 0
        self.detections = []

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(4, detectShadows=False)
        path = os.path.join(os.getcwd(), 'bg.jpg')
        bg_image = cv2.imread(path) if os.path.exists(path) else np.ones((1, 1, 3))
        self.bg_sub.apply(bg_image)

        self.main_view.new_frame.connect(self.new_frame)

        self.network_handler = NetworkHandler(os.path.join(os.getcwd(), 'weights'))

    @pyqtSlot(np.ndarray)
    def new_frame(self, frame):
        if frame is None:
            return
        mask = self.bg_sub.apply(frame)
        contours = len([c for c in imutils.grab_contours(cv2.findContours(mask, 1, cv2.CHAIN_APPROX_SIMPLE))
                        if cv2.contourArea(c) > 10000])
        if contours == 0:
            self.has_object = True
        else:
            self.has_object = False
            self.cnt_img = 0
            self.detections = []
            self.has_holes = False
            self.has_scratches = False
            self.has_bad_legs = False

        if self.has_object:
            if self.cnt_img < 10:
                self.cnt_img += 1
                detections = self.network_handler.detect(frame)
                if len(detections) > len(self.detections):
                    self.detections = detections
            for *xyxy, conf, cls in reversed(self.detections):
                label = f'{self.network_handler.names[int(cls)]} {conf:.2f}'
                plot_one_box(
                    xyxy,
                    frame,
                    label=label,
                    color=self.network_handler.colours[int(cls)],
                    line_thickness=2
                )
                if cls == 0:
                    self.has_holes = True
                elif cls == 1:
                    self.has_bad_legs = True
                elif cls == 2:
                    self.has_scratches = True

    def paintEvent(self, a0: QPaintEvent) -> None:
        super(DefectsWindow, self).paintEvent(a0)

        self.scratches.setStyleSheet(f'QLabel {{ background-color: {"red" if self.has_scratches else "green"}; }}')
        self.holes.setStyleSheet(f'QLabel {{ background-color: {"red" if self.has_holes else "green"}; }}')
        self.legs.setStyleSheet(f'QLabel {{ background-color: {"red" if self.has_bad_legs else "green"}; }}')

        pixmap = QPixmap(self.status.size())
        pixmap.fill(Qt.transparent)
        status_painter = QPainter(pixmap)
        clr = Qt.green if not self.has_holes and not self.has_scratches and not self.has_bad_legs else Qt.red
        status_painter.setBrush(QBrush(clr, Qt.SolidPattern))
        w = min(self.status.width() - 2, self.status.height() - 2)
        status_painter.drawEllipse(0, 0, w, w)
        status_painter.end()
        self.status.setPixmap(pixmap.scaled(self.status.width(), self.status.height(), Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widow = DefectsWindow()
    widow.show()
    app.exec_()
