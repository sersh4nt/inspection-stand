import os
import sys

import qimage2ndarray
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from design import Ui_mainWindow
import cv2
import numpy as np
from libs.network_handler import NetworkHandler
from libs.camera import *


class DefectsWindow(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = Camera(0)
        self.main_view.initialize(camera=self.camera)
        # self.main_view.new_frame_signal.connect(self.update)

        self.has_scratches = False
        self.has_holes = False
        self.has_bad_legs = False

        self.network_handler = NetworkHandler(os.path.join(os.getcwd(), 'weights'))

    def paintEvent(self, ev):
        scratch_painter = QPainter(self.scratch_indicator.pixmap())
        hole_painter = QPainter(self.hole_indicator.pixmap())
        leg_painter = QPainter(self.leg_indicator.pixmap())
        scratch_painter.setBrush(QColor(255, 0, 0) if self.has_scratches else QColor(0, 255, 0))
        hole_painter.setBrush(QColor(255, 0, 0) if self.has_scratches else QColor(0, 255, 0))
        leg_painter.setBrush(QColor(255, 0, 0) if self.has_scratches else QColor(0, 255, 0))
        scratch_painter.drawEllipse(10, 10, 2, 2)
        hole_painter.drawEllipse(10, 10, 2, 2)
        leg_painter.drawEllipse(10, 10, 2, 2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widow = DefectsWindow()
    widow.show()
    app.exec_()
