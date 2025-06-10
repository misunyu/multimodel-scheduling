import sys
import time
import os
import cv2
import numpy as np
import psutil
import onnxruntime as ort
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# 모델 및 데이터 경로 설정
target_model = "./yolov3_neubla.onnx"
resnet_model_path = "./resnet50.onnx"
resnet_image_dir = "./imagenet-sample-images"
input_video = "./stockholm_1280x720_1.mp4"
input_width = input_height = 608

# ImageNet 클래스 이름 로드
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# COCO 클래스 이름 정의
coco_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def preprocess_yolo(raw_input_img):
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, (raw_input_img.shape[1], raw_input_img.shape[0])

def preprocess_resnet(raw_input_img):
    img = cv2.resize(raw_input_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data

def draw_detections(img, box, score, class_id):
    class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"ID:{class_id}"
    label = f"{class_name} {score:.2f}"
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 빨간색 텍스트
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def postprocessing(output, original_img, img_width, img_height, confidence_thres=0.5, iou_thres=0.5):
    output_box = np.squeeze(output[0])
    output_label = np.squeeze(output[1])
    rows = output_box.shape[0]
    boxes, scores, class_ids = [], [], []
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    for i in range(rows):
        conf = output_box[i][4]
        if conf >= confidence_thres:
            left, top, right, bottom = output_box[i][:4]
            width = int((right - left) * x_factor)
            height = int((bottom - top) * y_factor)
            left = int(left * x_factor)
            top = int(top * y_factor)
            boxes.append([left, top, width, height])
            scores.append(conf)
            class_ids.append(int(output_label[i]))
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    if indices is not None:
        for idx in indices:
            i = int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])
            draw_detections(original_img, boxes[i], scores[i], class_ids[i])
    return original_img

def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qt_image)

class UnifiedViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO + ResNet Unified Viewer with CPU Monitor")
        self.resize(1280, 560)

        # 수평 영상 뷰어 레이아웃
        video_layout = QHBoxLayout()
        self.yolo_label = QLabel()
        self.resnet_label = QLabel()
        self.yolo_label.setFixedSize(640, 480)
        self.resnet_label.setFixedSize(640, 480)
        self.yolo_label.setScaledContents(True)
        self.resnet_label.setScaledContents(True)
        video_layout.addWidget(self.yolo_label)
        video_layout.addWidget(self.resnet_label)

        # CPU 사용률 라벨
        self.cpu_label = QLabel("CPU Usage: -- %")
        self.cpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cpu_label.setStyleSheet("font-size: 16px; color: blue;")

        # 전체 세로 레이아웃
        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.cpu_label)

        self.setLayout(main_layout)

        # 세션 및 영상/이미지 준비
        self.yolo_session = ort.InferenceSession(target_model)
        self.resnet_session = ort.InferenceSession(resnet_model_path)
        self.cap = cv2.VideoCapture(input_video)
        self.resnet_images = [os.path.join(resnet_image_dir, f)
                              for f in os.listdir(resnet_image_dir)
                              if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.resnet_index = 0

        # 타이머 설정
        self.yolo_timer = QTimer()
        self.yolo_timer.timeout.connect(self.update_yolo)
        self.yolo_timer.start(30)

        self.resnet_timer = QTimer()
        self.resnet_timer.timeout.connect(self.update_resnet)
        self.resnet_timer.start(1000)

        # YOLO FPS 계산을 위한 변수 초기화
        self.last_yolo_time = time.time()
        self.yolo_fps = 0.0
        self.yolo_infer_time = 0.0  # ms

        self.last_resnet_time = time.time()
        self.resnet_fps = 0.0
        self.resnet_infer_time = 0.0  # ms

        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_usage)
        self.cpu_timer.start(1000)

    def update_yolo(self):
        success, frame = self.cap.read()
        if not success:
            self.yolo_label.setText("YOLO: No video")
            return

        input_tensor, (w, h) = preprocess_yolo(frame)

        # 순수 추론 시간 측정 시작
        infer_start = time.time()
        output = self.yolo_session.run(None, {"input": input_tensor})
        infer_end = time.time()

        # 후처리 및 디스플레이
        result = postprocessing(output, frame, w, h)
        self.yolo_label.setPixmap(convert_cv_qt(result))

        # 추론 시간(ms) 및 처리량(FPS) 계산
        self.yolo_infer_time = (infer_end - infer_start) * 1000.0  # milliseconds
        self.yolo_fps = 1000.0 / self.yolo_infer_time if self.yolo_infer_time > 0 else 0.0

    def update_resnet(self):
        if not self.resnet_images:
            self.resnet_label.setText("No images found.")
            return

        if self.resnet_index >= len(self.resnet_images):
            self.resnet_index = 0

        img_path = self.resnet_images[self.resnet_index]
        img = cv2.imread(img_path)
        if img is None:
            self.resnet_label.setText(f"Failed to load {img_path}")
            return

        self.resnet_index += 1
        input_tensor = preprocess_resnet(img)

        # 순수 추론 시간 측정 시작
        infer_start = time.time()
        output = self.resnet_session.run(None, {"data": input_tensor})
        infer_end = time.time()

        # 결과 출력
        class_id = int(np.argmax(output[0]))
        class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
        cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.resnet_label.setPixmap(convert_cv_qt(img))

        # 추론 시간(ms) 및 처리량(FPS) 계산
        self.resnet_infer_time = (infer_end - infer_start) * 1000.0  # milliseconds
        self.resnet_fps = 1000.0 / self.resnet_infer_time if self.resnet_infer_time > 0 else 0.0

    def update_cpu_usage(self):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_label.setText(
            f"CPU Usage: {cpu_percent:.1f} % | "
            f"YOLO FPS: {self.yolo_fps:.1f} ({self.yolo_infer_time:.1f} ms) | "
            f"ResNet FPS: {self.resnet_fps:.1f} ({self.resnet_infer_time:.1f} ms)"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UnifiedViewer()
    viewer.show()
    sys.exit(app.exec())


