from libs.utils import *
from libs.yolo.general import *
from libs.yolo.plots import *
from libs.yolo.torch_utils import *
from models.yolo import *


class NetworkHandler:
    def __init__(self, path):
        set_logging()
        self.path = path
        self.data = []
        self.device = select_device('0' if torch.cuda.is_available() else 'cpu')
        self.half = False
        self.model = None
        self.stride = 0
        self.image_size = 640
        self.load_network()

        torch.multiprocessing.set_start_method('spawn')

    def load_network(self):
        path = os.path.join(self.path, 'last.pt')
        self.half = self.device.type != 'cpu'
        try:
            del self.model
            self.model = attempt_load(path, map_location=self.device)
            self.stride = int(self.model.stride.max())
            if self.half:
                self.model.half()
        except FileNotFoundError:
            msg = 'Cannot load neural network weights!\n' \
                  'The program would not support detection functions'
            QMessageBox.warning(None, 'Warning!', msg, QMessageBox.Ok)

    # this does work
    def detect(self, image):
        if self.model is None:
            return

        possible_result = []

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        self.image_size = check_img_size(640, s=self.stride)

        img = letterbox(image, self.image_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        predictions = self.model(img)[0]
        predictions = non_max_suppression(predictions, 0.25, 0.45)

        for detection in predictions:
            if len(detection):
                detection[:, :4] = scale_coords(img.shape[2:], detection[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(detection):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    possible_result.append(label)
                    plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=2)

        return possible_result, image
