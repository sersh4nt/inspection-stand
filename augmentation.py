from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import cv2
import glob


def read_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    lines = [line.strip('\n') for line in open(label_path)]
    boxes = []
    if lines != ['']:
        for line in lines:
            words = line.split(' ')
            id = words[0]
            x = int(float(words[1]) * w - float(words[3]) * w / 2)
            y = int(float(words[2]) * h - float(words[4]) * h / 2)
            he = int(float(words[4]) * h)
            wi = int(float(words[3]) * w)
            boxes.append((id, (x, y, wi, he)))
    return img, boxes

augmentor = createAugmentor('detection', 'yolo', 'yolo', 'linear', 'dataset', {'outputPath': 'dataset/augmented'})
transformer = transformerGenerator('detection')

v_flip = createTechnique('flip', {'flip': 0})
h_flip = createTechnique('flip', {'flip': 1})
hv_flip = createTechnique('flip', {'flip': -1})
avg_blur_3 = createTechnique('average_blurring', {'kernel': 3})
none = createTechnique('none', {})
augmentor.addTransformer(transformer(v_flip))
augmentor.addTransformer(transformer(h_flip))
augmentor.addTransformer(transformer(hv_flip))
augmentor.addTransformer(transformer(avg_blur_3))
augmentor.addTransformer(transformer(none))

augmentor.applyAugmentation()
