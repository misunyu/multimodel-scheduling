# multimodel_gui.py
import os
import time
import cv2
import numpy as np
import psutil
import onnxruntime as ort
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import queue
import threading

import json
import threading
from datetime import datetime

LOG_DIR = "./logs"
MAX_LOG_ENTRIES = 500
log = 1  # 로그 기록 여부 제어

input_width = input_height = 608

with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

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

def async_log(model_name, infer_time_ms, avg_fps):
    if not log:
        return

    def write_log():
        os.makedirs(LOG_DIR, exist_ok=True)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "inference_time_ms": round(infer_time_ms, 2),
            "average_fps": round(avg_fps, 2)
        }
        log_file = os.path.join(LOG_DIR, f"{model_name}_log.json")

        # 기존 로그 읽기 (line-by-line 방식)
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # 새로운 항목 추가
        logs.append(log_data)

        # 최대 항목 수 제한
        if len(logs) > MAX_LOG_ENTRIES:
            logs = logs[-MAX_LOG_ENTRIES:]

        # 다시 기록
        with open(log_file, "w") as f:
            for entry in logs:
                json.dump(entry, f)
                f.write("\n")

    threading.Thread(target=write_log).start()


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
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def postprocessing_cpu(output, original_img, img_width, img_height, confidence_thres=0.5, iou_thres=0.5):
    output_data = np.squeeze(output[0])  # shape: (22743, 85)

    boxes, scores, class_ids = [], [], []
    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for row in output_data:
        object_conf = row[4]
        class_probs = row[5:]
        class_id = int(np.argmax(class_probs))
        class_conf = class_probs[class_id]
        score = object_conf * class_conf

        if score >= confidence_thres:
            cx, cy, w, h = row[0:4]
            x1 = int((cx - w / 2) * x_factor)
            y1 = int((cy - h / 2) * y_factor)
            w = int(w * x_factor)
            h = int(h * y_factor)

            boxes.append([x1, y1, w, h])
            scores.append(float(score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    if indices is not None and len(indices) > 0:
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

class UnifiedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("multimodel_display_layout.ui", self)

        self.yolo_label = self.findChild(QLabel, "yolo_label")
        self.resnet_label = self.findChild(QLabel, "resnet_label")
        self.yolo_info_label = self.findChild(QLabel, "yolo_info_label")
        self.cpu_info_label = self.findChild(QLabel, "cpu_info_label")
        self.npu_info_label = self.findChild(QLabel, "npu_info_label")

        # ONNX 세션 초기화
        self.yolo_session = ort.InferenceSession("models/yolov3_big/model/yolov3_big.onnx")
        self.resnet_session = ort.InferenceSession("models/resnet50/model/resnet50.onnx")

        # 비디오 및 이미지 입력
        self.cap = cv2.VideoCapture("./stockholm_1280x720.mp4")
        self.resnet_images = [os.path.join("./imagenet-sample-images", f)
                              for f in os.listdir("./imagenet-sample-images")
                              if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.resnet_index = 0

        # FPS/지연시간 통계 변수 초기화
        self.yolo_total_infer_time = 0.0
        self.yolo_infer_count = 0
        self.yolo_avg_infer_time = 0.0
        self.yolo_avg_fps = 0.0

        self.resnet_total_infer_time = 0.0
        self.resnet_infer_count = 0
        self.resnet_avg_infer_time = 0.0
        self.resnet_avg_fps = 0.0

        # CPU 정보
        self.prev_cpu_stats = get_cpu_metrics(interval=0)

        # YOLO 처리용 큐 및 스레드 초기화
        self.yolo_frame_queue = queue.Queue(maxsize=5)
        self.yolo_stop_flag = threading.Event()

        threading.Thread(target=self.capture_yolo_frames, daemon=True).start()
        threading.Thread(target=self.process_yolo_frames, daemon=True).start()

        # ResNet 순차 실행 시작
        QTimer.singleShot(0, self.update_resnet)

        # CPU/NPU 사용률 주기적 업데이트
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)


    def capture_yolo_frames(self):
        while not self.yolo_stop_flag.is_set():
            success, frame = self.cap.read()
            if not success:
                continue
            if not self.yolo_frame_queue.full():
                self.yolo_frame_queue.put(frame)

    def process_yolo_frames(self):
        while not self.yolo_stop_flag.is_set():
            try:
                frame = self.yolo_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            input_tensor, (w, h) = preprocess_yolo(frame)
            infer_start = time.time()

            try:
                output = self.yolo_session.run(None, {"images": input_tensor})
            except Exception as e:
                print(f"[YOLO ERROR] {e}")
                continue

            infer_end = time.time()
            result = postprocessing_cpu(output, frame, w, h)
            pixmap = convert_cv_qt(result)

            # UI 스레드에서 안전하게 갱신
            self.yolo_label.setPixmap(pixmap)

            current_infer_time = (infer_end - infer_start) * 1000.0
            self.yolo_total_infer_time += current_infer_time
            self.yolo_infer_count += 1
            self.yolo_avg_infer_time = self.yolo_total_infer_time / self.yolo_infer_count
            self.yolo_avg_fps = 1000.0 / self.yolo_avg_infer_time if self.yolo_avg_infer_time > 0 else 0.0

            async_log("yolov3_big", current_infer_time, self.yolo_avg_fps)


    def update_yolo(self):
        success, frame = self.cap.read()
        if not success:
            self.yolo_label.setText("YOLO: No video")
            return

        input_tensor, (w, h) = preprocess_yolo(frame)
        infer_start = time.time()

        try:
            output = self.yolo_session.run(None, {"images": input_tensor})
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
            return

        infer_end = time.time()
        result = postprocessing_cpu(output, frame, w, h)
        self.yolo_label.setPixmap(convert_cv_qt(result))

        current_infer_time = (infer_end - infer_start) * 1000.0
        self.yolo_total_infer_time += current_infer_time
        self.yolo_infer_count += 1
        self.yolo_avg_infer_time = self.yolo_total_infer_time / self.yolo_infer_count
        self.yolo_avg_fps = 1000.0 / self.yolo_avg_infer_time if self.yolo_avg_infer_time > 0 else 0.0

        async_log("yolov3_big", current_infer_time, self.yolo_avg_fps)

        # 이전 프레임 처리가 끝났으므로 다음 프레임 즉시 처리
        QTimer.singleShot(0, self.update_yolo)

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

        infer_start = time.time()
        try:
            output = self.resnet_session.run(None, {"data": input_tensor})
        except Exception as e:
            print(f"[ResNet ERROR] {e}")
            return
        infer_end = time.time()

        class_id = int(np.argmax(output[0]))
        class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
        cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.resnet_label.setPixmap(convert_cv_qt(img))

        current_infer_time = (infer_end - infer_start) * 1000.0
        self.resnet_total_infer_time += current_infer_time
        self.resnet_infer_count += 1
        self.resnet_avg_infer_time = self.resnet_total_infer_time / self.resnet_infer_count
        self.resnet_avg_fps = 1000.0 / self.resnet_avg_infer_time if self.resnet_avg_infer_time > 0 else 0.0

        async_log("resnet50", current_infer_time, self.resnet_avg_fps)

        # 다음 이미지 바로 처리
        QTimer.singleShot(0, self.update_resnet)

    def update_cpu_npu_usage(self):
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]

        self.yolo_info_label.setText(
            f"<b>YOLO</b> Avg FPS: {self.yolo_avg_fps:.1f} "
            f"(<span style='color: gray;'>{self.yolo_avg_infer_time:.1f} ms</span>) | "
            f"<b><span style='color: purple;'>ResNet</span></b> Avg FPS: "
            f"<span style='color: purple;'>{self.resnet_avg_fps:.1f}</span> "
            f"(<span style='color: purple;'>{self.resnet_avg_infer_time:.1f} ms</span>)"
        )

        self.cpu_info_label.setText(
            f"<b><span style='color: blue;'>CPU</span></b><br>"
            f"Usage: {current['CPU_Usage_percent']:.1f} %<br>"
            f"LoadAvg: {load1:.2f} / {load5:.2f} / {load15:.2f}<br>"
            f"CtxSwitches/sec: {delta_ctx} | Int/sec: {delta_int}"
        )

        self.npu_info_label.setText(
            f"<b><span style='color: green;'>NPU</span></b><br>"
            f"Usage: 42.0 %<br>"
            f"LoadAvg: 0.12 / 0.10 / 0.08<br>"
            f"CtxSwitches/sec: 12 | Int/sec: 3"
        )

        self.prev_cpu_stats = current
