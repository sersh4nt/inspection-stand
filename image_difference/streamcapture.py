from PyQt5.QtCore import (QThread, pyqtSignal)
import cv2
import numpy as np


class streamCapture(QThread):
    getframe = pyqtSignal(np.ndarray)

    def __init__(self, cam):
        super().__init__()
        self.camip = "v4l2src device=/dev/video"+ str(cam) +" ! jpegdec ! videoconvert ! appsink" 
        self.frame = None
        self.cap = None
        self.stop = False
        self.y_offset = 5
        self.x_offset_left = 20
        self.x_offset_right = 250

    def run(self):
        self.cap = cv2.VideoCapture(self.camip, cv2.CAP_GSTREAMER)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        while (not (self.stop)):
            while (not self.cap.isOpened()) and (not self.stop):
                self.sleep(2)
                self.cap = cv2.VideoCapture(self.camip, cv2.CAP_GSTREAMER)
            ret, self.frame = self.cap.read()
            if ret:
                self.getframe.emit(self.frame.copy())

    def exit(self):
        self.stop = True
    
    def get_current_frame(self):
        return self.frame.copy()

    def reopenStream(self, cam):
        self.camip = cam
        self.cap = cv2.VideoCapture(self.camip)

