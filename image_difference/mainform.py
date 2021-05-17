from PyQt5 import QtCore
import PyQt5
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout,
                             QVBoxLayout, QPushButton, QGridLayout, QFileDialog)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtCore import (pyqtSlot, pyqtSignal, Qt)
import cv2
import sys
import numpy as np
from imagedifference import imageDifference

class mainForm(QWidget):
    get_template_filename = pyqtSignal(str)
    get_compare_image_filename = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.screen_resolution = None
        self.grid_layout = None
        self.output_picture = None
        self.initUI()
    
    def initUI(self):
        self.screen_resolution = QApplication.desktop().screenGeometry()
        self.resize(self.screen_resolution.size())
        self.move(self.screen_resolution.left(), self.screen_resolution.top())
        self.grid_layout = QGridLayout()
        self.load_template_button = QPushButton("load template")
        self.load_template_button.clicked.connect(self.show_dialog)
        self.load_difference_img = QPushButton("load image to compare")
        self.load_difference_img.clicked.connect(self.show_dialog)
        self.output_picture = QLabel()
        self.grid_layout.addWidget(self.output_picture, 0, 0, 3, 3)
        self.grid_layout.addWidget(self.load_template_button, 0, 3, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        self.grid_layout.addWidget(self.load_difference_img, 1, 3, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(self.grid_layout)
    
    def show_dialog(self):
        dialog_text = ""
        template = False
        if self.sender() == self.load_template_button:
            dialog_text = "Choose template"
            template = True
        else:
            dialog_text = "Choose image to compare"
        filename = QFileDialog.getOpenFileName(self, dialog_text, "./examples")[0]
        if filename != "":
            if template:
                self.get_template_filename.emit(filename)
            else:
                self.get_compare_image_filename.emit(filename)

    pyqtSlot(np.ndarray)
    def mat2qimage(self, image):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgbImage = cv2.resize(rgbImage, (self.output_picture.size().width(),
                                         self.output_picture.size().height()))
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        result_image = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.output_picture.setPixmap(QPixmap.fromImage(result_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = mainForm()
    image_difference_thread = imageDifference()
    main.get_template_filename.connect(image_difference_thread.set_template_image)
    main.get_compare_image_filename.connect(image_difference_thread.get_image)
    image_difference_thread.output_image_defference.connect(main.mat2qimage)
    main.show()
    image_difference_thread.start()
    sys.exit(app.exec_())