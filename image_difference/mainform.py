from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QLabel, QDockWidget, QFileSystemModel,
                             QVBoxLayout, QPushButton, QGridLayout, QComboBox, QLineEdit, QTreeView, QSlider)
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPainterPath)
from PyQt5.QtCore import (pyqtSlot, pyqtSignal, Qt, QDir, QRectF, QFileInfo)
import cv2
import sys
import numpy as np
import imutils
# sys.path.append("/home/sersh4nt/presentation/image_difference/libs")
from libs.imagedifference import imageDifference
from libs.streamcapture import streamCapture


class mainForm(QMainWindow):
    get_template_filename = pyqtSignal(str)
    exit_program = pyqtSignal()
    camera_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.stream = None
        self.image_difference_thread = None
        self.template_set = False
        self._countour_max_tresh = 10
        self._countour_min_tresh = 1
        self._transparency_max = 10
        self._transparency_min = 0
        self._countour_gamma_max = 10
        self._countour_gamma_min = 1
        self._color_max = 255
        self._color_min = 0
        self.screen_resolution = None
        self.grid_layout = None
        self.output_picture = None
        self.initUI()
        self.init_image_difference()
        if self.webcam_switcher.count() > 0:
            self.stream = streamCapture(self.webcam_switcher.itemData(self.webcam_switcher.currentIndex()))
            self.stream.getframe.connect(self.mat2qimage)
            self.webcam_switcher.currentIndexChanged.connect(self.camera_switcher_index_changed)
            self.camera_changed.connect(self.stream.reopenStream)
            self.stream.start()
            self.exit_program.connect(self.stream.exit)

    ### отрисовка интерфейса
    def initUI(self):
        self.screen_resolution = QApplication.desktop().screenGeometry()
        # self.resize(self.screen_resolution.size())
        # self.move(self.screen_resolution.left(), self.screen_resolution.top())
        self.grid_layout = QGridLayout()

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)
        self.central_widget.setMaximumSize(self.screen_resolution.width() // 4 * 3,
                                           self.screen_resolution.height() // 4 * 3)
        self.setCentralWidget(self.central_widget)

        self.camera_label = QLabel("Camera:")
        self.grid_layout.addWidget(self.camera_label, 0, 0, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        self.webcam_switcher = QComboBox()
        self.detect_webcam_devices(self.webcam_switcher)
        self.grid_layout.addWidget(self.webcam_switcher, 0, 1, 1, 3)

        self.output_picture = QLabel()
        self.grid_layout.addWidget(self.output_picture, 1, 0, 1, 4)

        ### creating right dock
        self.right_dock_layout = QVBoxLayout()

        self.right_dock_widget = QDockWidget()
        self.right_dock_widget.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
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
        self.filter_template_edit.setStyleSheet(
            "background-image: url(../image_difference/icons/searchIcon.png); background-repeat: no-repeat; background-position: right;")

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
        # self.directory_tree_view.sortByColumn(0)
        self.directory_tree_view.setSortingEnabled(True)
        self.directory_tree_view.doubleClicked.connect(self.load_template)
        self.directory_tree_view.setRootIndex(self.file_system_model.index("../image_difference/"))

        self.right_dock_layout.addWidget(self.directory_tree_view)

        self.load_template_button = QPushButton("Select Template")
        self.load_template_button.setMaximumSize(self.screen_resolution.width() // 4 - 30, 30)
        self.load_template_button.clicked.connect(self.load_template)

        self.right_dock_layout.addWidget(self.load_template_button)

        self.create_template_button = QPushButton("Create Template")
        self.create_template_button.setMaximumSize(self.screen_resolution.width() // 4 - 30, 30)
        self.create_template_button.clicked.connect(self.create_template)

        self.right_dock_layout.addWidget(self.create_template_button)

        self.template_image_widget = QWidget()
        self.template_image_widget.setMinimumSize(self.screen_resolution.width() // 4 - 20,
                                                  self.screen_resolution.width() // 4 - 10)

        self.template_image_back = QLabel(self.template_image_widget)
        self.template_image_back.resize(self.screen_resolution.width() // 4 - 20,
                                        self.screen_resolution.width() // 4 - 10)
        pix = QPixmap(self.template_image_back.size())
        pix.fill(Qt.lightGray)
        rect = QRectF(0.0, 0.0, self.template_image_back.size().width(), self.template_image_back.size().height())
        painter = QPainter()
        painter.begin(pix)
        painter.setRenderHints(QPainter.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(rect, 5.0, 5.0)
        painter.drawPath(path)
        painter.end()
        self.template_image_back.setPixmap(pix)

        self.template_image = QLabel(self.template_image_widget)
        self.template_image.move(5, 5)
        self.template_image.resize(self.screen_resolution.width() // 4 - 30, self.screen_resolution.width() // 4 - 30)

        self.template_image_text = QLabel(self.template_image_widget, text="Current Template")
        self.template_image_text.setStyleSheet("font-weight: bold")
        self.template_image_text.move(self.screen_resolution.width() // 8 - 65, 20)

        self.right_dock_layout.addWidget(self.template_image_widget)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock_widget)

        ### creating bottom dock
        self.bottom_dock_layout = QGridLayout()
        self.bottom_dock_layout.setSpacing(10)

        self.bottom_dock_widget = QDockWidget()
        self.bottom_dock_widget.setMinimumSize(self.screen_resolution.width() // 4 * 3 - 10,
                                               self.screen_resolution.height() // 4 - 10)
        bottom_dock = QWidget(self.bottom_dock_widget)
        bottom_dock.setMinimumSize(self.screen_resolution.width() // 4 * 3 - 20,
                                   self.screen_resolution.height() // 4 - 20)
        bottom_dock.move(10, 10)
        bottom_dock.setLayout(self.bottom_dock_layout)

        settings_label = QLabel("Settings:")
        self.bottom_dock_layout.addWidget(settings_label, 0, 0, 1, 2,
                                          Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        countour_tresh_label = QLabel("Countour Tresh:")
        self.bottom_dock_layout.addWidget(countour_tresh_label, 1, 0, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.countour_tresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.countour_tresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.countour_tresh_slider.setRange(self._countour_min_tresh, self._countour_max_tresh)
        self.countour_tresh_slider.setValue(2)
        self.bottom_dock_layout.addWidget(self.countour_tresh_slider, 1, 1, 1, 1, Qt.AlignmentFlag.AlignTop)

        transparency_weight_label = QLabel("Transparency:")
        self.bottom_dock_layout.addWidget(transparency_weight_label, 2, 0, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.transparency_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.transparency_weight_slider.setValue(6)
        self.transparency_weight_slider.setRange(self._transparency_min, self._transparency_max)
        self.bottom_dock_layout.addWidget(self.transparency_weight_slider, 2, 1, 1, 1, Qt.AlignmentFlag.AlignTop)

        countour_gamma_label = QLabel("Countour Gamma:")
        self.bottom_dock_layout.addWidget(countour_gamma_label, 3, 0, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.countour_gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.countour_gamma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.countour_gamma_slider.setValue(8)
        self.countour_gamma_slider.setRange(self._countour_gamma_min, self._countour_gamma_max)
        self.bottom_dock_layout.addWidget(self.countour_gamma_slider, 3, 1, 1, 1, Qt.AlignmentFlag.AlignTop)

        ### right side of settings
        countour_color_label = QLabel("Countour Color:")
        self.bottom_dock_layout.addWidget(countour_color_label, 0, 2, 1, 2,
                                          Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        r_color_label = QLabel("R:")
        self.bottom_dock_layout.addWidget(r_color_label, 1, 2, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.r_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.r_color_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.r_color_slider.setRange(self._color_min, self._color_max)
        self.r_color_slider.setValue(255)
        self.bottom_dock_layout.addWidget(self.r_color_slider, 1, 3, 1, 1, Qt.AlignmentFlag.AlignTop)

        g_color_label = QLabel("G:")
        self.bottom_dock_layout.addWidget(g_color_label, 2, 2, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.g_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.g_color_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.g_color_slider.setRange(self._color_min, self._color_max)
        self.bottom_dock_layout.addWidget(self.g_color_slider, 2, 3, 1, 1, Qt.AlignmentFlag.AlignTop)

        b_color_label = QLabel("B:")
        self.bottom_dock_layout.addWidget(b_color_label, 3, 2, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.b_color_slider = QSlider(Qt.Orientation.Horizontal)
        self.b_color_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.b_color_slider.setRange(self._color_min, self._color_max)
        self.bottom_dock_layout.addWidget(self.b_color_slider, 3, 3, 1, 1, Qt.AlignmentFlag.AlignTop)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock_widget)

        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

    def init_image_difference(self):
        self.image_difference_thread = imageDifference()
        self.get_template_filename.connect(self.image_difference_thread.set_template_image)
        self.countour_tresh_slider.valueChanged.connect(self.image_difference_thread.set_countour_tresh_value)
        self.transparency_weight_slider.valueChanged.connect(self.image_difference_thread.set_transparency_weight_value)
        self.countour_gamma_slider.valueChanged.connect(self.image_difference_thread.set_countour_gamma_value)
        self.r_color_slider.valueChanged.connect(self.image_difference_thread.set_countour_color_r)
        self.g_color_slider.valueChanged.connect(self.image_difference_thread.set_countour_color_g)
        self.b_color_slider.valueChanged.connect(self.image_difference_thread.set_countour_color_b)
        self.image_difference_thread.output_image_defference.connect(self.mat2qimage)
        self.image_difference_thread.set_template_picture.connect(self.set_template_picture)
        self.image_difference_thread.start()

    def detect_webcam_devices(self, combo_box):
        _video_capture = cv2.VideoCapture()
        _dev_id = 0
        while (_dev_id < 3):
            if _video_capture.open(_dev_id):
                combo_box.addItem("Device #" + str(_dev_id + 1), _dev_id)
                _dev_id += 1
            else:
                _dev_id += 1
        _video_capture.release()

    def load_template(self):
        index = self.directory_tree_view.selectedIndexes()[0]
        if not QFileInfo(self.file_system_model.filePath(index)).isDir():
            # print("load template, path:", self.file_system_model.filePath(index))
            self.get_template_filename.emit(self.file_system_model.filePath(index))

    def create_template(self):
        if self.stream is not None:
            template_to_save = self.stream.get_current_frame()
            cv2.imwrite("../image_difference/examples/template.jpg", template_to_save)
            print("create template")

    pyqtSlot(np.ndarray)

    def mat2qimage(self, image):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgbImage = cv2.resize(rgbImage, (self.output_picture.size().width(),
                                         self.output_picture.size().height()))
        # rgbImage = imutils.resize(rgbImage, height=self.output_picture.height())
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        result_image = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.output_picture.setPixmap(QPixmap.fromImage(result_image))

    pyqtSlot(np.ndarray)

    def set_template_picture(self, image):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rgbImage = cv2.resize(rgbImage, (self.template_image.size().width(),
        #                                  self.template_image.size().height()))
        rgbImage = imutils.resize(rgbImage, self.template_image.size().width())
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        result_image = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.template_image.setPixmap(QPixmap.fromImage(result_image))
        if not self.template_set:
            self.template_set = True
            self.stream.getframe.disconnect(self.mat2qimage)
            self.stream.getframe.connect(self.image_difference_thread.get_image)
            self.image_difference_thread.output_image_defference.connect(self.mat2qimage)

    def camera_switcher_index_changed(self, index):
        self.camera_changed.emit(self.webcam_switcher.itemData(index))
        print("current index:", index)
        print("item data:", self.webcam_switcher.itemData(index))

    def closeEvent(self, event):
        self.exit_program.emit()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # stylesheetfile = "dark-orange-green.qss"
    # with open(stylesheetfile,"r") as fh:
    #     app.setStyleSheet(fh.read())
    main = mainForm()
    # image_difference_thread.set_diff_image.connect(main.set_diff_picture)
    # image_difference_thread.set_orig_image.connect(main.set_orig_picture)
    main.showFullScreen()
    sys.exit(app.exec_())