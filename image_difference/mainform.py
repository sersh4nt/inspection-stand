from PyQt5 import QtCore
import PyQt5
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QLabel, QHBoxLayout, QDockWidget, QFileSystemModel,
                             QVBoxLayout, QPushButton, QGridLayout, QComboBox, QLineEdit, QTreeView)
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtCore import (pyqtSlot, pyqtSignal, Qt, QDir)
import cv2
import sys
import numpy as np
from imagedifference import imageDifference

class mainForm(QMainWindow):
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
        # self.resize(self.screen_resolution.size())
        # self.move(self.screen_resolution.left(), self.screen_resolution.top())
        self.grid_layout = QGridLayout()

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)
        self.central_widget.setMaximumSize(self.screen_resolution.width() // 4 * 3, self.screen_resolution.height() // 4 * 3)
        self.setCentralWidget(self.central_widget)
        
        self.camera_label = QLabel("Camera:")
        self.grid_layout.addWidget(self.camera_label, 0, 0, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        self.webcam_switcher = QComboBox()
        self.detect_webcam_devices(self.webcam_switcher)
        self.grid_layout.addWidget(self.webcam_switcher, 0, 1, 1, 3)
        
        self.output_picture = QLabel()
        self.grid_layout.addWidget(self.output_picture, 1, 0, 1, 4)

        self.right_dock_layout = QVBoxLayout()

        self.right_dock_widget = QDockWidget()
        self.right_dock_widget.setMinimumSize(self.screen_resolution.width() // 4, self.screen_resolution.height())
        right_dock = QWidget(self.right_dock_widget)
        right_dock.setMinimumSize(self.screen_resolution.width() // 4, self.screen_resolution.height())
        right_dock.setLayout(self.right_dock_layout)
        
        template_label = QLabel("Templates")
        template_label.setMinimumSize(50, 25)
        self.right_dock_layout.addWidget(template_label)
        
        self.filter_template_edit = QLineEdit()
        self.filter_template_edit.setPlaceholderText("Filter (Ctr + Alt + f)")
        template_label.setMinimumSize(90, 25)
        self.filter_template_edit.setStyleSheet("background-image: url(/home/dmitry/work/from_vlad/inspection-stand/image_difference/icons/searchIcon.png); background-repeat: no-repeat; background-position: right;")

        self.right_dock_layout.addWidget(self.filter_template_edit)

        self.file_system_model = QFileSystemModel()
        self.file_system_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.AllEntries)
        self.file_system_model.setRootPath(QDir.currentPath())

        self.directory_tree_view = QTreeView()
        self.directory_tree_view.setModel(self.file_system_model)
        self.directory_tree_view.setMinimumSize(200, 100)
        self.directory_tree_view.hideColumn(1)
        self.directory_tree_view.hideColumn(2)
        self.directory_tree_view.hideColumn(3)
        self.directory_tree_view.doubleClicked.connect(self.load_template)
        self.directory_tree_view.setRootIndex(self.file_system_model.index(QDir.currentPath()))
        
        self.right_dock_layout.addWidget(self.directory_tree_view)

        self.load_template_button = QPushButton("Select Template")
        self.load_template_button.clicked.connect(self.load_template)

        self.right_dock_layout.addWidget(self.load_template_button)

        self.create_template_button = QPushButton("Create Template")
        self.create_template_button.clicked.connect(self.create_template)

        self.template_image_widget = QWidget()
        self.template_image_widget.setMinimumSize(self.screen_resolution.width() // 4, self.screen_resolution.width() // 4)
        self.template_image = QLabel(self.template_image_widget)
        self.template_image.resize(self.template_image_widget.size())

        self.template_image_text = QLabel(self.template_image_widget, text="template")

        self.right_dock_layout.addWidget(self.template_image_widget)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock_widget)


    def detect_webcam_devices(self, combo_box):
        _video_capture = cv2.VideoCapture()
        _dev_id = 0
        while(True):
            if _video_capture.open(_dev_id):
                combo_box.addItem("Device #1")
                _dev_id += 1
            else:
                break
        _video_capture.release()
    
    def load_template(self):
        index = self.directory_tree_view.selectedIndexes()[0]
        print("load template, path:", self.file_system_model.filePath(index))
        self.get_template_filename.emit(self.file_system_model.filePath(index))

    def create_template(self):
        print("create template")

    pyqtSlot(np.ndarray)
    def mat2qimage(self, image):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rgbImage = cv2.resize(rgbImage, (self.output_picture.size().width(),
        #                                  self.output_picture.size().height()))
        rgbImage = imutils.resize(rgbImage, self.output_picture.size().width())                                 
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
    image_difference_thread.set_template_picture.connect(main.set_template_picture)
    # image_difference_thread.set_diff_image.connect(main.set_diff_picture)
    # image_difference_thread.set_orig_image.connect(main.set_orig_picture)
    main.showFullScreen()
    image_difference_thread.start()
    sys.exit(app.exec_())