from PyQt5.QtCore import (QThread, pyqtSignal)
import cv2
import numpy as np


class streamCapture(QThread):
    getframe = pyqtSignal(np.ndarray)

    def __init__(self, cam):
        super().__init__()
        self.camip = cam
        self.cap = None
        self.stop = False

    def run(self):
        self.cap = cv2.VideoCapture(self.camip)
        while (not (self.stop)):
            while (not self.cap.isOpened()) and (not self.stop):
                self.sleep(2)
                self.cap = cv2.VideoCapture(self.camip)
            ret, frame = self.cap.read()
            if ret:
                self.getframe.emit(frame)

    def exit(self):
        self.stop = True

    def reopenStream(self, cam):
        self.camip = cam
        self.cap = cv2.VideoCapture(self.camip)

