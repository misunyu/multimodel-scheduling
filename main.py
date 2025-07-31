import sys
import time
import os
import cv2
import numpy as np
import psutil
import onnxruntime as ort
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# 모델 및 데이터 경로 설정
target_model = "./yolov3_neubla.onnx"
resnet_model_path = "models/resnet50/model/resnet50.onnx"
resnet_image_dir = "./imagenet-sample-images"
input_video = "./stockholm_1280x720.mp4"
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
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)

def get_cpu_metrics(interval=0):
    cpu_percent = psutil.cpu_percent(interval=interval)
    load1, load5, load15 = os.getloadavg()
    cpu_stats = psutil.cpu_stats()
    ctx_switches = cpu_stats.ctx_switches
    interrupts = cpu_stats.interrupts
    return {
        "CPU_Usage_percent": cpu_percent,
        "Load_Average": (load1, load5, load15),
        "Context_Switches": ctx_switches,
        "Interrupts": interrupts
    }

class UnifiedViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO + ResNet Unified Viewer with CPU & NPU Monitor")
        self.resize(1280, 640)

        # -------------------- 상단 영상 뷰어 --------------------
        video_layout = QHBoxLayout()
        self.view1_label = QLabel()  # Previously yolo_label
        self.view2_label = QLabel()  # Previously resnet_label
        self.view1_label.setFixedSize(640, 480)
        self.view2_label.setFixedSize(640, 480)
        self.view1_label.setScaledContents(True)
        self.view2_label.setScaledContents(True)
        video_layout.addWidget(self.view1_label)
        video_layout.addWidget(self.view2_label)

        # -------------------- 상단 추론 정보 라벨 (YOLO + ResNet) --------------------
        self.model_performance_label = QLabel()
        self.model_performance_label.setAlignment(Qt.AlignLeft)
        self.model_performance_label.setStyleSheet("font-size: 14px; padding: 4px;")
        self.model_performance_label.setTextFormat(Qt.RichText)

        # -------------------- CPU / NPU 정보 라벨 --------------------
        self.cpu_info_label = QLabel()
        self.npu_info_label = QLabel()
        for label in [self.cpu_info_label, self.npu_info_label]:
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("font-size: 13px; padding: 4px;")
            label.setTextFormat(Qt.RichText)
            # 필요 시 고정 높이 제거 또는 줄이기
            label.setFixedHeight(100)

        cpu_npu_layout = QHBoxLayout()
        cpu_npu_layout.addWidget(self.cpu_info_label)
        cpu_npu_layout.addWidget(self.npu_info_label)

        # -------------------- 하단 정보 레이아웃 --------------------
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)  # YOLO FPS와 CPU info 사이 간격 최소화
        info_layout.addWidget(self.model_performance_label)
        info_layout.addLayout(cpu_npu_layout)

        # -------------------- 전체 레이아웃 --------------------
        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(info_layout)
        self.setLayout(main_layout)

        # -------------------- 모델 로딩 --------------------
        self.yolo_session = ort.InferenceSession(target_model)
        self.resnet_session = ort.InferenceSession(resnet_model_path)

        self.cap = cv2.VideoCapture(input_video)
        self.resnet_images = [os.path.join(resnet_image_dir, f)
                              for f in os.listdir(resnet_image_dir)
                              if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.resnet_index = 0

        # -------------------- 추론 성능 측정 초기화 --------------------
        self.yolo_total_infer_time = 0.0
        self.yolo_infer_count = 0
        self.yolo_avg_infer_time = 0.0
        self.yolo_avg_fps = 0.0

        self.resnet_total_infer_time = 0.0
        self.resnet_infer_count = 0
        self.resnet_avg_infer_time = 0.0
        self.resnet_avg_fps = 0.0

        # -------------------- CPU 통계 초기화 --------------------
        self.prev_cpu_stats = get_cpu_metrics(interval=0)

        # -------------------- 타이머 설정 --------------------
        self.yolo_timer = QTimer()
        self.yolo_timer.timeout.connect(self.update_yolo)
        self.yolo_timer.start(30)

        self.resnet_timer = QTimer()
        self.resnet_timer.timeout.connect(self.update_resnet)
        self.resnet_timer.start(1000)

        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)

    def update_yolo(self):
        success, frame = self.cap.read()
        if not success:
            self.view1_label.setText("View1 (YOLO): No video")
            return

        input_tensor, (w, h) = preprocess_yolo(frame)

        infer_start = time.time()
        try:
            output = self.yolo_session.run(None, {"input": input_tensor})
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
            return

        infer_end = time.time()

        result = postprocessing(output, frame, w, h)
        self.view1_label.setPixmap(convert_cv_qt(result))

        # 현재 추론 시간
        current_infer_time = (infer_end - infer_start) * 1000.0  # ms

        # 누적 계산
        self.yolo_total_infer_time += current_infer_time
        self.yolo_infer_count += 1
        self.yolo_avg_infer_time = self.yolo_total_infer_time / self.yolo_infer_count
        self.yolo_avg_fps = 1000.0 / self.yolo_avg_infer_time if self.yolo_avg_infer_time > 0 else 0.0

    def update_resnet(self):
        if not self.resnet_images:
            self.view2_label.setText("No images found.")
            return

        if self.resnet_index >= len(self.resnet_images):
            self.resnet_index = 0

        img_path = self.resnet_images[self.resnet_index]
        img = cv2.imread(img_path)
        if img is None:
            self.view2_label.setText(f"Failed to load {img_path}")
            return

        self.resnet_index += 1
        input_tensor = preprocess_resnet(img)

        infer_start = time.time()
        output = self.resnet_session.run(None, {"data": input_tensor})
        infer_end = time.time()

        class_id = int(np.argmax(output[0]))
        class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
        cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.view2_label.setPixmap(convert_cv_qt(img))

        # 현재 추론 시간
        current_infer_time = (infer_end - infer_start) * 1000.0  # ms

        # 누적 계산
        self.resnet_total_infer_time += current_infer_time
        self.resnet_infer_count += 1
        self.resnet_avg_infer_time = self.resnet_total_infer_time / self.resnet_infer_count
        self.resnet_avg_fps = 1000.0 / self.resnet_avg_infer_time if self.resnet_avg_infer_time > 0 else 0.0

    def update_cpu_npu_usage(self):
        # 현재 CPU 통계 획득
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]

        # ▶ View1 (YOLO) + View2 (ResNet) 정보를 한 줄로 통합 출력 (왼쪽 위)
        self.model_performance_label.setText(
            f"<b>View1 (YOLO)</b> Avg FPS: {self.yolo_avg_fps:.1f} "
            f"(<span style='color: gray;'>{self.yolo_avg_infer_time:.1f} ms</span>)"
            f" | "
            f"<b><span style='color: purple;'>View2 (ResNet)</span></b> Avg FPS: "
            f"<span style='color: purple;'>{self.resnet_avg_fps:.1f}</span> "
            f"(<span style='color: purple;'>{self.resnet_avg_infer_time:.1f} ms</span>)"
        )

        # ▶ CPU 정보 출력 (왼쪽 하단)
        self.cpu_info_label.setText(
            f"<b><span style='color: blue;'>CPU</span></b><br>"
            f"Usage: {current['CPU_Usage_percent']:.1f} %<br>"
            f"LoadAvg: {load1:.2f} / {load5:.2f} / {load15:.2f}<br>"
            f"CtxSwitches/sec: {delta_ctx} | Int/sec: {delta_int}"
        )

        # ▶ NPU 정보 출력 (오른쪽 하단) - 임의 값
        self.npu_info_label.setText(
            f"<b><span style='color: green;'>NPU</span></b><br>"
            f"Usage: 42.0 %<br>"
            f"LoadAvg: 0.12 / 0.10 / 0.08<br>"
            f"CtxSwitches/sec: 12 | Int/sec: 3"
        )

        # 이전 상태 갱신
        self.prev_cpu_stats = current


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = UnifiedViewer()
    viewer.show()
    sys.exit(app.exec_())
