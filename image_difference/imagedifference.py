from PyQt5.QtCore import (QThread, pyqtSlot, pyqtSignal, QMutex, QTimer)
import cv2
import numpy as np
import imutils

class imageDifference(QThread):
    output_image_defference = pyqtSignal(np.ndarray)
    set_template_picture = pyqtSignal(np.ndarray)
    set_diff_image = pyqtSignal(np.ndarray)
    set_orig_image = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.stop_thread = False
        self.template_image = None
        self.template_mask = None
        self.current_image = None
        self.current_image_grayscale = None
        self.diff_image = None
        self.background_substractor = cv2.createBackgroundSubtractorMOG2(1, 25, False)
        self.countourThresh = 6000
        self.transparency_weight = 0.4

    def run(self):
        self.mutex.lock()
        while (not self.stop_thread):
            self.mutex.lock()
            self.get_image_difference()

    def get_image_difference(self):
        buf_image = cv2.GaussianBlur(self.current_image, (21, 21), 0)
        buf_image = self.background_substractor.apply(buf_image, self.template_mask, 0)

        kernel = np.ones((3,3),np.uint8)
        buf_image = cv2.dilate(buf_image,kernel,iterations=2)

        self.set_mask_image.emit(buf_image.copy())
        original_image_copy = self.current_image.copy()
        countours = cv2.findContours(buf_image,  1, cv2.CHAIN_APPROX_SIMPLE)
        countours = imutils.grab_contours(countours)
        original_image_copy_grayscale = cv2.cvtColor(original_image_copy, cv2.COLOR_BGR2GRAY)
        original_image_copy_grayscale = cv2.equalizeHist(original_image_copy_grayscale)
        original_image_copy = cv2.cvtColor(original_image_copy_grayscale, cv2.COLOR_GRAY2RGB)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image_grayscale = cv2.equalizeHist(self.current_image)
        self.current_image = cv2.cvtColor(self.current_image_grayscale, cv2.COLOR_GRAY2RGB)
        self.set_orig_image.emit(self.current_image.copy())
        for countour in countours:
            if cv2.contourArea(countour) > self.countourThresh:
                cv2.fillPoly(original_image_copy, [countour], (0, 0, 255))
                self.set_diff_image.emit(original_image_copy.copy())
                self.current_image = cv2.addWeighted(original_image_copy, self.transparency_weight, self.current_image, 1 - self.transparency_weight, 0.8)
        self.output_image_defference.emit(self.current_image.copy())

    @pyqtSlot(str)
    def set_template_image(self, filename):
        self.template_image = cv2.imread(filename)
        self.set_template_picture.emit(self.template_image.copy())
        self.template_image = cv2.GaussianBlur(self.template_image, (21, 21), 0)
        self.template_mask = self.background_substractor.apply(self.template_image)
        
    
    @pyqtSlot(str)
    def get_image(self, filename):
        self.current_image = cv2.imread(filename)
        self.mutex.unlock()

    def exit(self):
        self.stop_thread = True
        self.mutex.unlock()
