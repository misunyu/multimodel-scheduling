# multimodel_gui.py
import os
import time
import cv2
import numpy as np
import psutil
import onnxruntime as ort
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import queue
import threading
import json
from datetime import datetime
from NeublaDriver import NeublaDriver


class ModelSignals(QObject):
    update_yolo_display = pyqtSignal(QPixmap)
    update_view1_display = pyqtSignal(QPixmap)
    update_view2_display = pyqtSignal(QPixmap)  # ✅ view2용 시그널 명확히 추가


LOG_DIR = "./logs"
MAX_LOG_ENTRIES = 500
log = 0  # 로그 기록 여부 제어

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

        need_trim = False
        line_count = 0

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                for _ in f:
                    line_count += 1
            need_trim = line_count >= MAX_LOG_ENTRIES

        if need_trim:
            logs = []
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            logs.append(log_data)
            logs = logs[-MAX_LOG_ENTRIES:]
            with open(log_file, "w") as f:
                for entry in logs:
                    json.dump(entry, f)
                    f.write("\n")
        else:
            with open(log_file, "a+") as f:
                json.dump(log_data, f)
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
    output_data = np.squeeze(output[0])
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
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    try:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    except Exception as e:
        print(f"[convert_cv_qt ERROR] {e}")
        return QPixmap()

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
        self.view1 = self.findChild(QLabel, "view1")
        self.view2 = self.findChild(QLabel, "view2")
        self.yolo_info_label = self.findChild(QLabel, "yolo_info_label")
        self.cpu_info_label = self.findChild(QLabel, "cpu_info_label")
        self.npu_info_label = self.findChild(QLabel, "npu_info_label")

        self.model_signals = ModelSignals()
        self.model_signals.update_yolo_display.connect(self.update_yolo_display)
        self.model_signals.update_view1_display.connect(self.update_view1_display)
        # self.model_signals.update_view2_display = pyqtSignal(QPixmap)  # 시그널 동적 생성
        self.model_signals.update_view2_display.connect(self.update_view2_display)

        self.yolo_session = ort.InferenceSession("models/yolov3_big/model/yolov3_big.onnx")
        self.view1_session = ort.InferenceSession("models/yolov3_small/model/yolov3_small.onnx")
        self.resnet_session = ort.InferenceSession("models/resnet50/model/resnet50.onnx")
        self.view2_session = ort.InferenceSession("models/resnet50/model/resnet50.onnx")  # view2 용

        self.cap = cv2.VideoCapture("./stockholm_1280x720.mp4")
        self.resnet_images = [os.path.join("./imagenet-sample-images", f)
                              for f in os.listdir("./imagenet-sample-images")
                              if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.resnet_index = 0
        self.view2_index = 0

        self.yolo_total_infer_time = 0.0
        self.yolo_infer_count = 0
        self.yolo_avg_infer_time = 0.0
        self.yolo_avg_fps = 0.0

        self.resnet_total_infer_time = 0.0
        self.resnet_infer_count = 0
        self.resnet_avg_infer_time = 0.0
        self.resnet_avg_fps = 0.0

        self.prev_cpu_stats = get_cpu_metrics(interval=0)

        self.yolo_frame_queue = queue.Queue(maxsize=5)
        self.yolo_result_queue = queue.Queue(maxsize=5)
        self.view1_frame_queue = queue.Queue(maxsize=5)
        self.view1_result_queue = queue.Queue(maxsize=5)
        self.view2_result_queue = queue.Queue(maxsize=5)
        self.yolo_stop_flag = threading.Event()

        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_yolo_frames, daemon=True).start()
        threading.Thread(target=self.process_view1_frames, daemon=True).start()
        threading.Thread(target=self.process_view2_frames, daemon=True).start()
        threading.Thread(target=self.display_yolo_frames, daemon=True).start()
        threading.Thread(target=self.display_view1_frames, daemon=True).start()
        threading.Thread(target=self.display_view2_frames, daemon=True).start()

        QTimer.singleShot(0, self.update_resnet)

        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)

    def closeEvent(self, event):
        self.yolo_stop_flag.set()
        time.sleep(0.2)  # 모든 쓰레드 종료 대기
        event.accept()


    def capture_frames(self):
        while not self.yolo_stop_flag.is_set():
            success, frame = self.cap.read()
            if not success:
                continue
            if not self.yolo_frame_queue.full():
                self.yolo_frame_queue.put(frame.copy())
            if not self.view1_frame_queue.full():
                self.view1_frame_queue.put(frame)

    def update_stats(self, model_name, current_infer_time):
        """모델별 평균 FPS 및 추론 시간 갱신 및 로그 기록"""
        if model_name == "yolov3_big":
            self.yolo_total_infer_time += current_infer_time
            self.yolo_infer_count += 1
            self.yolo_avg_infer_time = self.yolo_total_infer_time / self.yolo_infer_count
            self.yolo_avg_fps = 1000.0 / self.yolo_avg_infer_time if self.yolo_avg_infer_time > 0 else 0.0
        elif model_name == "resnet50":
            self.resnet_total_infer_time += current_infer_time
            self.resnet_infer_count += 1
            self.resnet_avg_infer_time = self.resnet_total_infer_time / self.resnet_infer_count
            self.resnet_avg_fps = 1000.0 / self.resnet_avg_infer_time if self.resnet_avg_infer_time > 0 else 0.0

        async_log(model_name, current_infer_time,
                  self.yolo_avg_fps if model_name == "yolov3_big" else self.resnet_avg_fps)

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
            current_infer_time = (infer_end - infer_start) * 1000.0
            if not self.yolo_result_queue.full():
                self.yolo_result_queue.put((result, current_infer_time))
            self.update_stats("yolov3_big", current_infer_time)


    def process_yolo_frames_npu(self):
        while not self.yolo_stop_flag.is_set():
            try:
                frame = self.yolo_frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            input_tensor, (w, h) = preprocess_yolo(frame)
            
            # infer_start = time.time()
            # try:
            #     output = self.yolo_session.run(None, {"images": input_tensor})
            # except Exception as e:
            #     print(f"[YOLO ERROR] {e}")
            #     continue
            # infer_end = time.time()
            # result = postprocessing_cpu(output, frame, w, h)
            # current_infer_time = (infer_end - infer_start) * 1000.0
            # if not self.yolo_result_queue.full():
            #     self.yolo_result_queue.put((result, current_infer_time))
            # self.update_stats("yolov3_big", current_infer_time)

    def process_view1_frames(self):
        while not self.yolo_stop_flag.is_set():
            try:
                frame = self.view1_frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            input_tensor, (w, h) = preprocess_yolo(frame)
            try:
                output = self.view1_session.run(None, {"images": input_tensor})
            except Exception as e:
                print(f"[VIEW1 ERROR] {e}")
                continue
            result = postprocessing_cpu(output, frame, w, h)
            if not self.view1_result_queue.full():
                self.view1_result_queue.put(result)

    def display_yolo_frames(self):
        while not self.yolo_stop_flag.is_set():
            try:
                result, _ = self.yolo_result_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(result)
            self.model_signals.update_yolo_display.emit(pixmap)

    def update_yolo_display(self, pixmap):
        self.yolo_label.setPixmap(pixmap)

    def display_view1_frames(self):
        while not self.yolo_stop_flag.is_set():
            try:
                result = self.view1_result_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(result)
            self.model_signals.update_view1_display.emit(pixmap)

    def update_view1_display(self, pixmap):
        self.view1.setPixmap(pixmap)

    def process_view2_frames(self):
        if not self.resnet_images:
            return
        while not self.yolo_stop_flag.is_set():
            if self.view2_index >= len(self.resnet_images):
                self.view2_index = 0
            img_path = self.resnet_images[self.view2_index]
            img = cv2.imread(img_path)
            self.view2_index += 1
            if img is None:
                continue
            input_tensor = preprocess_resnet(img)
            try:
                output = self.view2_session.run(None, {"data": input_tensor})
            except Exception as e:
                print(f"[View2 ERROR] {e}")
                continue
            class_id = int(np.argmax(output[0]))
            class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
            cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if not self.view2_result_queue.full():
                self.view2_result_queue.put(img)
            time.sleep(0.1)  # 너무 빠르게 순환되지 않도록 조정

    def display_view2_frames(self):
        while not self.yolo_stop_flag.is_set():
            try:
                result = self.view2_result_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(result)
            self.model_signals.update_view2_display.emit(pixmap)

    def update_view2_display(self, pixmap):
        self.view2.setPixmap(pixmap)

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
        self.update_stats("resnet50", current_infer_time)
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

        def load_yolo_npu(self, npu_num, o_path):

            try:
                driver = NeublaDriver()
                assert driver.Init(npu_num) == 0
                assert driver.LoadModel(o_path) == 0

            except Exception as e:
                try:
                    driver.Close()
                except:
                    pass
                print(f"[Error] NPU{npu_num}: {e}")
                exit()


        def process_yolo_npu(self, driver, input_data):
            try:
                # driver = NeublaDriver()
                # assert driver.Init(npu_num) == 0
                #
                # start_load = time.time()
                # assert driver.LoadModel(o_path) == 0
                # end_load = time.time()
                # load_time_ms = (end_load - start_load) * 1000.0
                #
                # random_input = np.random.rand(3, 608, 608).astype(np.uint8)
                # input_data = random_input.tobytes()
                #
                # start_infer = time.time()
                assert driver.SendInput(input_data, 3 * 608 * 608) == 0
                assert driver.Launch() == 0
                raw_outputs = driver.ReceiveOutputs()
                # end_infer = time.time()
                # infer_time_ms = (end_infer - start_infer) * 1000.0

                # assert driver.Close() == 0

            except Exception as e:
                try:
                    driver.Close()
                except:
                    pass

        def close_npu(self, driver):
            try:
                driver.Close()
            except:
                pass

        def process_resnet50_npu(self, npu_num, o_path):
            try:
                driver = NeublaDriver()
                assert driver.Init(npu_num) == 0

                start_load = time.time()
                assert driver.LoadModel(o_path) == 0
                end_load = time.time()
                load_time_ms = (end_load - start_load) * 1000.0

                random_input = np.random.rand(3, 224, 224).astype(np.uint8)
                input_data = random_input.tobytes()

                start_infer = time.time()
                assert driver.SendInput(input_data, 3 * 224 * 224) == 0
                assert driver.Launch() == 0
                raw_outputs = driver.ReceiveOutputs()
                end_infer = time.time()
                infer_time_ms = (end_infer - start_infer) * 1000.0

                assert driver.Close() == 0

            except Exception as e:
                try:
                    driver.Close()
                except:
                    pass
                print(f"[Error] NPU{npu_num}: {e}")
                exit()

            return load_time_ms, infer_time_ms
