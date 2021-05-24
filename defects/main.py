import sys

import imutils
import qimage2ndarray
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from libs.camera import Camera
from libs.network_handler import NetworkHandler
from libs.yolo.plots import *
from new_design import Ui_MainWindow
from libs.database_editor import *

MARGIN = 20


class DefectsWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = Camera(0)
        self.main_view.initialize(camera=self.camera)

        self.detect_video_devices()

        self.has_scratches = False
        self.has_holes = False
        self.has_bad_legs = False
        self.has_object = False
        self.cnt_img = 0
        self.cnt_defect = 0
        self.detections = []

        self.detect_legs = True
        self.detect_holes = True
        self.detect_scratches = True

        self.network_handler = NetworkHandler(os.path.join(os.getcwd(), 'weights'))
        self.database_editor = DatabaseEditor(self.camera, os.path.join(os.getcwd(), 'data'))

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(4, detectShadows=False)
        path = os.path.join(os.getcwd(), 'bg.jpg')
        bg_image = cv2.imread(path) if os.path.exists(path) else np.ones((1, 1, 3))
        self.bg_sub.apply(bg_image)

        self.main_view.new_frame.connect(self.new_frame)
        self.next_button.clicked.connect(self.incr_cnt)
        self.prev_button.clicked.connect(self.decr_cnt)

        self.change_database.clicked.connect(self.show_database_editor)
        self.camera_swithcer.currentIndexChanged.connect(self.change_camera)
        self.legs_checkbox.stateChanged.connect(self.checkbox_changed)
        self.holes_checkbox.stateChanged.connect(self.checkbox_changed)
        self.scratches_checkbox.stateChanged.connect(self.checkbox_changed)

    def show_database_editor(self):
        self.database_editor.stream_enabled = True
        self.database_editor.show()

    def checkbox_changed(self):
        self.detect_legs = self.legs_checkbox.isChecked()
        self.detect_holes = self.holes_checkbox.isChecked()
        self.detect_scratches = self.scratches_checkbox.isChecked()

    def change_camera(self, index):
        self.camera = Camera(index)

    def detect_video_devices(self):
        _video_capture = cv2.VideoCapture()
        _dev_id = 0
        while _dev_id < 10:
            if _video_capture.open(_dev_id):
                self.camera_swithcer.addItem(f"Device #{_dev_id + 1}")
                _dev_id += 1
            else:
                break
        _video_capture.release()

    def incr_cnt(self):
        if self.has_object:
            self.cnt_defect += 1
            self.cnt_defect %= len(self.detections)

    def decr_cnt(self):
        if self.has_object:
            self.cnt_defect -= 1
            if self.cnt_defect < 0:
                self.cnt_defect += len(self.detections)

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
            self.cnt_defect = 0

        if self.has_object:
            self.name_label.setText('Microchip\nPIC18F6527')

            if self.cnt_img < 10:
                self.cnt_img += 1
                detections = self.network_handler.detect(frame)
                if len(detections) > len(self.detections):
                    self.detections = detections
            for *xyxy, conf, cls in reversed(self.detections):
                if cls == 0 and self.detect_holes or cls == 1 and self.detect_legs or cls == 2 and self.detect_scratches:
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

    def paintEvent(self, ev):
        super(DefectsWindow, self).paintEvent(ev)

        if self.has_object and len(self.detections):
            for *xyxy, _, cls in [self.detections[self.cnt_defect]]:
                he, wi = self.camera.frame.shape[:2]
                cutout = self.camera.frame[
                         max(0, int(xyxy[1]) - MARGIN): min(he - 1, int(xyxy[3]) + MARGIN),
                         max(0, int(xyxy[0]) - MARGIN): min(wi - 1, int(xyxy[2]) + MARGIN)
                         ]
                scale = min(self.defect_view.height() / cutout.shape[0], self.defect_view.width() / cutout.shape[1])
                cutout = cv2.resize(cutout, None, fx=scale, fy=scale)
                pmap = QPixmap(self.defect_view.size())
                pmap.fill(Qt.transparent)
                defect_painter = QPainter(pmap)
                defect_painter.drawImage(QPoint(0, 0), qimage2ndarray.array2qimage(cutout))
                self.defect_view.setPixmap(
                    pmap.scaled(self.defect_view.width(), self.defect_view.height(), Qt.KeepAspectRatio))
                defect_painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widow = DefectsWindow()
    widow.show()
    app.exec_()
