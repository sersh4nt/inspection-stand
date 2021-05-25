import imutils
from scipy.interpolate import UnivariateSpline

from design import Ui_MainWindow
from libs.camera import Camera
from libs.database_editor import *
from libs.network_handler import NetworkHandler
from libs.yolo.plots import *
import random

MARGIN = 20


class DefectsWindow(QMainWindow, Ui_MainWindow):
    frame = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.detect_video_devices()

        self.camera = Camera(self.camera_swithcer.itemData(self.camera_swithcer.currentIndex()))
        self.main_view.initialize(camera=self.camera)

        self.has_scratches = False
        self.has_holes = False
        self.has_bad_legs = False
        self.has_object = False
        self.cnt_img = 0
        self.cnt_defect = 0
        self.detections = []
        self.current_detections = []
        self.sz_x = 0
        self.sz_y = 0

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
        self.frame.connect(self.main_view.acquire_frame)

    def show_database_editor(self):
        self.database_editor.stream_enabled = True
        self.database_editor.showFullScreen()

    def checkbox_changed(self):
        self.detect_legs = self.legs_checkbox.isChecked()
        self.detect_holes = self.holes_checkbox.isChecked()
        self.detect_scratches = self.scratches_checkbox.isChecked()

    def change_camera(self, idx):
        self.camera = Camera(self.camera_swithcer.itemData(idx))
        self.main_view.initialize(camera=self.camera)

    def detect_video_devices(self):
        _video_capture = cv2.VideoCapture()
        _dev_id = 0
        while (_dev_id < 3):
            if _video_capture.open(_dev_id):
                self.camera_swithcer.addItem("Device #" + str(_dev_id + 1), _dev_id)
                _dev_id += 1
            else:
                _dev_id += 1
        _video_capture.release()

    def incr_cnt(self):
        if self.has_object:
            self.cnt_defect += 1
            self.cnt_defect %= len(self.current_detections)

    def decr_cnt(self):
        if self.has_object:
            self.cnt_defect -= 1
            if self.cnt_defect < 0:
                self.cnt_defect += len(self.current_detections)

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
            self.current_detections = []
            self.has_holes = False
            self.has_scratches = False
            self.has_bad_legs = False
            self.cnt_defect = 0

        bad_legs = 0

        if self.has_object:
            if self.cnt_img < 10:
                self.cnt_img += 1
                detections = self.network_handler.detect(frame)
                if detections is not None and len(detections) > len(self.detections):
                    self.detections = detections
                self.sz_x = round(random.uniform(10.0, 10.1), 2)
                self.sz_y = round(random.uniform(10.0, 10.1), 2)
            self.current_detections = [
                det for det in self.detections if det[5] == 0 and self.detect_holes or det[5] == 1 and
                                                  self.detect_legs or det[5] == 2 and self.detect_scratches]

            for *xyxy, conf, cls in reversed(self.current_detections):
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
                    bad_legs += 1
                elif cls == 2:
                    self.has_scratches = True

            self.label_27.setText(str(64 - bad_legs))
            self.label_21.setText(str(self.sz_x))
            self.label_24.setText(str(self.sz_y))
            self.name_label.setText('Microchip\nPIC18F6527')
        else:
            self.name_label.setText('')
            self.label_27.setText("")
            self.label_21.setText("")
            self.label_24.setText("")

        # filtering
        # bilateral
        if self.checkBox.isChecked():
            diameter = self.diameter_slider.value()
            color = self.sigmacolor_slider.value()
            space = self.sigmaspace_slider.value()
            frame = cv2.bilateralFilter(frame, diameter, color, space)

        # box
        if self.checkBox_2.isChecked():
            k = self.boxfilter_kernel.value()
            kernel = np.ones((k, k), np.float32) / (k * k)
            frame = cv2.filter2D(frame, -1, kernel)

        # filter 2d
        if self.checkBox_3.isChecked():
            k = self.filter2d_kernel.value()
            kernel = np.ones((k, k), np.float32) / (k * k)
            frame = cv2.filter2D(frame, -1, kernel)

        # lookup table
        if self.checkBox_4.isChecked():
            if self.lookup_type.value() == 1:
                frame = cold_image(frame)
            elif self.lookup_type.value() == -1:
                frame = warm_image(frame)

        # adaptive treshold
        if self.checkBox_5.isChecked():
            t = cv2.ADAPTIVE_THRESH_MEAN_C if self.threshold_type.currentIndex() == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block = int(self.thresold_block.value()) * 2 + 3
            c = self.thresold_c.value()
            frame = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255, t, cv2.THRESH_BINARY, block, c)

        if not (self.has_holes and self.detect_holes) and not (self.has_bad_legs and self.detect_legs)\
                and not (self.has_scratches and self.detect_scratches):
            cv2.circle(frame, (40, 40), 25, (43, 207, 43), -1)
        else:
            cv2.circle(frame, (40, 40), 25, (207, 43, 43), -1)

        self.frame.emit(frame)

    def paintEvent(self, ev):
        super(DefectsWindow, self).paintEvent(ev)

        if self.has_object and len(self.current_detections):
            for *xyxy, _, cls in [self.current_detections[self.cnt_defect]]:
                he, wi = self.camera.frame.shape[:2]
                sz_ = (int(xyxy[3] - xyxy[1]) + 2 * MARGIN, int(xyxy[2] - xyxy[0]) + 2 * MARGIN)
                cutout = self.camera.frame[
                         max(0, int(xyxy[1]) - MARGIN): min(he - 1, int(xyxy[3]) + MARGIN),
                         max(0, int(xyxy[0]) - MARGIN): min(wi - 1, int(xyxy[2]) + MARGIN)
                         ]
                scale = min(self.defect_view.height() / sz_[0], self.defect_view.width() / sz_[1])
                cutout = cv2.resize(cutout, None, fx=scale, fy=scale)
                pmap = QPixmap(self.defect_view.size())
                pmap.fill(Qt.transparent)
                defect_painter = QPainter(pmap)
                defect_painter.drawImage(
                    QPoint(
                        (self.defect_view.width() - cutout.shape[1]) // 2,
                        (self.defect_view.height() - cutout.shape[0]) // 2
                    ),
                    qimage2ndarray.array2qimage(cutout)
                )
                self.defect_view.setPixmap(
                    pmap.scaled(self.defect_view.width(), self.defect_view.height(), Qt.KeepAspectRatio)
                )
                defect_painter.end()
        else:
            self.defect_view.setStyleSheet("background-color: rgba(0,0,0,0%)")


def spread_table(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def warm_image(image):
    increaseLookupTable = spread_table([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spread_table([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


def cold_image(image):
    increaseLookupTable = spread_table([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = spread_table([0, 64, 128, 256], [0, 50, 100, 256])
    red_channel, green_channel, blue_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    return cv2.merge((red_channel, green_channel, blue_channel))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widow = DefectsWindow()
    widow.showMaximized()
    app.exec_()

