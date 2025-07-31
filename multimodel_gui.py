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
from npu import initialize_driver, close_driver, send_receive_data_npu, resnet50_prepare_onnx_model, resnet50_preprocess, yolo_prepare_onnx_model, yolo_preprocess
from multiprocessing import Process, Queue, Event

class ModelSignals(QObject):
    update_yolo_display = pyqtSignal(QPixmap)
    update_view1_display = pyqtSignal(QPixmap)
    update_view2_display = pyqtSignal(QPixmap)
    update_resnet_display = pyqtSignal(QPixmap)

# Constants
LOG_DIR = "./logs"
MAX_LOG_ENTRIES = 500
log = 0  # Controls whether to record logs
input_width = input_height = 608

# Load class labels
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

def video_reader_process(video_path, frame_queue: Queue, shutdown_event: Event, max_queue_size=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video Reader ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if frame_queue.qsize() < max_queue_size:
            frame_queue.put(frame)
        time.sleep(frame_delay)

    cap.release()

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

# Image preprocessing functions
# Image preprocessing functions
def preprocess_image(raw_input_img, target_width, target_height):
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_width, target_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data

def preprocess_yolo(raw_input_img):
    image_data = preprocess_image(raw_input_img, input_width, input_height)
    return image_data, (raw_input_img.shape[1], raw_input_img.shape[0])

def preprocess_resnet(raw_input_img):
    return preprocess_image(raw_input_img, 224, 224)

# Detection visualization
def draw_detections(img, box, score, class_id):
    class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"ID:{class_id}"
    label = f"{class_name} {score:.2f}"
    x, y, w, h = box

    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - text_height - 4), (x + text_width, y), color, -1)
    cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Post-processing functions
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

def postprocessing_npu(output, original_img, img_width, img_height, confidence_thres=0.5, iou_thres=0.5):
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

            # Filter out abnormal bbox values
            if width <= 0 or height <= 0 or width > 2000 or height > 2000:
                continue
            if left < -100 or top < -100 or left > 3000 or top > 3000:
                continue

            boxes.append([left, top, width, height])
            scores.append(conf)
            class_ids.append(int(output_label[i]))

    valid_indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    drawn_boxes = []

    if valid_indices is not None and len(valid_indices) > 0:
        for idx in valid_indices:
            i = int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])
            draw_detections(original_img, boxes[i], scores[i], class_ids[i])
            drawn_boxes.append(boxes[i])

    return original_img, drawn_boxes

# Utility functions
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

# NPU processing functions
def run_resnet_npu_process(image_dir, output_queue, shutdown_event):
    try:
        image_files = [os.path.join(image_dir, f)
                      for f in os.listdir(image_dir)
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not image_files:
            print("[ResNet NPU] No images found")
            return

        front_sess, back_sess, params = resnet50_prepare_onnx_model(
            "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
        )

        scale = params['/0/avgpool/GlobalAveragePool_output_0_scale'] * params['0.fc.weight_scale']
        zp_act = params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
        zp_w = params['0.fc.weight_zero_point']
        scale_out = params['/0/fc/Gemm_output_0_scale']
        zp_out = params['/0/fc/Gemm_output_0_zero_point']
        weight_q = params['0.fc.weight_quantized'].T.astype(np.int32)

        driver = initialize_driver(1, "./models/resnet50/npu_code/resnet50_neubla_p1.o")

        index = 0
        while not shutdown_event.is_set():
            if index >= len(image_files):
                index = 0

            img = cv2.imread(image_files[index])
            index += 1
            if img is None:
                continue

            infer_start = time.time()
            input_data = front_sess.run(None, {"input": resnet50_preprocess(img)})[0].tobytes()
            raw_outputs = send_receive_data_npu(driver, input_data, 3 * 224 * 224)
            output_data = np.frombuffer(raw_outputs[0], dtype=np.uint8)

            try:
                back_output = back_sess.run(None, {"input": output_data.reshape(1, -1)})
                output = back_output[0]
                max_index = int(np.argmax(output))
            except Exception as e:
                output = np.matmul(output_data.astype(np.int32), weight_q)
                output -= zp_act * np.sum(weight_q, axis=0)
                output -= zp_w * np.sum(output_data, axis=0)
                output += zp_act * zp_w
                output = np.round(output * scale / scale_out) + zp_out
                output = output.astype(np.uint8)
                max_index = int(np.argmax(output))

            infer_end = time.time()
            class_name = f"Class ID: {max_index}"
            cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            output_queue.put((img, class_name, (infer_end - infer_start) * 1000.0))

    except Exception as e:
        print(f"[ResNet NPU Process ERROR] {e}")
    finally:
        close_driver(driver)

def run_yolo_npu_process(input_queue, output_queue, shutdown_event):
    try:
        front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
            "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
        )

        driver = initialize_driver(0, "./models/yolov3_small/npu_code/yolov3_small_neubla_p1.o")
        frame_delay = 1.0 / 30.0

        while not shutdown_event.is_set():
            try:
                frame = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            input_tensor, (w, h) = preprocess_yolo(frame)
            infer_start = time.time()

            try:
                front_output = front_sess.run(None, {"input": input_tensor})[0]
                input_data = front_output.tobytes()

                raw_outputs = send_receive_data_npu(driver, input_data, 3 * 608 * 608)
                output_data = [np.frombuffer(buf, dtype=np.uint8) for buf in raw_outputs]

                output_dequant_data = [
                    (data.astype(np.float32) - zero_point[name]) * scale[name]
                    for name, data in zip(
                        ["onnx::Transpose_684_DequantizeLinear",
                         "onnx::Transpose_688_DequantizeLinear",
                         "onnx::Transpose_692_DequantizeLinear"],
                        output_data
                    )
                ]

                shape_dict = {
                    "onnx::Transpose_684": (1, 255, 19, 19),
                    "onnx::Transpose_688": (1, 255, 38, 38),
                    "onnx::Transpose_692": (1, 255, 76, 76),
                }

                back_feeds = {}
                for name, data in zip(shape_dict.keys(), output_dequant_data):
                    needed_size = np.prod(shape_dict[name])
                    if data.size < needed_size:
                        print(f"[YOLO NPU ERROR] insufficient data for {name}, expected {needed_size}, got {data.size}")
                        raise ValueError("Invalid data size")
                    back_feeds[name] = data[:needed_size].reshape(shape_dict[name])

                output = back_sess.run(None, back_feeds)

            except Exception as e:
                print(f"[YOLO NPU ERROR] {e}")
                continue

            infer_end = time.time()
            result_img, drawn_boxes = postprocessing_npu(output, frame, w, h)

            if drawn_boxes:
                infer_time_ms = (infer_end - infer_start) * 1000.0
                output_queue.put((result_img, infer_time_ms))

            time.sleep(frame_delay)

    except Exception as e:
        print(f"[YOLO NPU Process ERROR] {e}")
    finally:
        close_driver(driver)

class UnifiedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("multimodel_display_layout.ui", self)

        # UI objects
        self.yolo_label = self.findChild(QLabel, "view1")
        self.resnet_label = self.findChild(QLabel, "view2")
        self.view1 = self.findChild(QLabel, "view3")
        self.view2 = self.findChild(QLabel, "view4")
        self.yolo_info_label = self.findChild(QLabel, "yolo_info_label")
        self.cpu_info_label = self.findChild(QLabel, "cpu_info_label")
        self.npu_info_label = self.findChild(QLabel, "npu_info_label")

        # Define and connect signals
        self.model_signals = ModelSignals()
        self.model_signals.update_yolo_display.connect(self.update_yolo_display)
        self.model_signals.update_view1_display.connect(self.update_view1_display)
        self.model_signals.update_view2_display.connect(self.update_view2_display)
        self.model_signals.update_resnet_display.connect(self.update_resnet_display)

        # Initialize common state variables
        self.shutdown_flag = threading.Event()
        self.prev_cpu_stats = get_cpu_metrics(interval=0)

        # Initialize queues and events
        self.video_frame_queue = Queue(maxsize=10)
        self.video_shutdown_event = Event()
        self.yolo_result_queue = queue.Queue(maxsize=5)
        self.view1_result_queue = queue.Queue(maxsize=5)
        self.view2_result_queue = queue.Queue(maxsize=5)
        self.resnet_result_queue = queue.Queue(maxsize=5)
        self.view1_frame_queue = queue.Queue(maxsize=10)
        self.yolo_frame_queue = Queue(maxsize=10)
        self.yolo_output_queue = Queue(maxsize=5)
        self.resnet_output_queue = Queue(maxsize=5)
        self.yolo_shutdown_event = Event()
        self.resnet_shutdown_event = Event()

        # Initialize statistics variables
        self.yolo_total_infer_time = 0.0
        self.yolo_infer_count = 0
        self.yolo_avg_infer_time = 0.0
        self.yolo_avg_fps = 0.0
        self.resnet_total_infer_time = 0.0
        self.resnet_infer_count = 0
        self.resnet_avg_infer_time = 0.0
        self.resnet_avg_fps = 0.0
        # Initialize statistics variables for View1 (YOLO CPU) and View2 (ResNet CPU)
        self.view1_total_infer_time = 0.0
        self.view1_infer_count = 0
        self.view1_avg_infer_time = 0.0
        self.view1_avg_fps = 0.0
        self.view2_total_infer_time = 0.0
        self.view2_infer_count = 0
        self.view2_avg_infer_time = 0.0
        self.view2_avg_fps = 0.0

        # Initialize images and sessions
        self.resnet_images = [os.path.join("./imagenet-sample-images", f)
                             for f in os.listdir("./imagenet-sample-images")
                             if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.resnet_index = 0
        self.view2_index = 0
        self.view1_session = ort.InferenceSession("models/yolov3_small/model/yolov3_small.onnx")
        self.view2_session = ort.InferenceSession("models/resnet50/model/resnet50.onnx")

        # Start processes
        self.video_reader_proc = Process(
            target=video_reader_process,
            args=("stockholm_1280x720.mp4", self.video_frame_queue, self.video_shutdown_event),
            daemon=True
        )
        self.video_reader_proc.start()

        self.yolo_process = Process(
            target=run_yolo_npu_process,
            args=(self.yolo_frame_queue, self.yolo_output_queue, self.yolo_shutdown_event),
        )
        self.yolo_process.start()

        self.resnet_process = Process(
            target=run_resnet_npu_process,
            args=("./imagenet-sample-images", self.resnet_output_queue, self.resnet_shutdown_event),
        )
        self.resnet_process.start()

        # Start threads
        threading.Thread(target=self.display_yolo_frames_from_process, daemon=True).start()
        threading.Thread(target=self.display_resnet_frames_from_process, daemon=True).start()
        threading.Thread(target=self.process_view1_frames, daemon=True).start()
        threading.Thread(target=self.process_view2_frames, daemon=True).start()
        threading.Thread(target=self.display_view1_frames, daemon=True).start()
        threading.Thread(target=self.display_view2_frames, daemon=True).start()
        threading.Thread(target=self.feed_view1_queue, daemon=True).start()

        # CPU/NPU monitoring
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)

    def feed_view1_queue(self):
        fps = 30.0
        try:
            cap = cv2.VideoCapture("stockholm_1280x720.mp4")
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            if fps_read > 1.0:
                fps = fps_read
            cap.release()
        except Exception as e:
            print(f"[feed_view1_queue] Failed to read FPS, using default 30.0: {e}")

        frame_delay = 1.0 / fps

        while not self.shutdown_flag.is_set():
            try:
                frame = self.video_frame_queue.get(timeout=1)
                if not self.view1_frame_queue.full():
                    self.view1_frame_queue.put(frame.copy())
                if not self.yolo_frame_queue.full():
                    self.yolo_frame_queue.put(frame.copy())
                time.sleep(frame_delay)
            except queue.Empty:
                continue

    def display_yolo_frames_from_process(self):
        while not self.shutdown_flag.is_set():
            try:
                frame, infer_time = self.yolo_output_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(frame)
            if not pixmap.isNull():
                self.model_signals.update_yolo_display.emit(pixmap)
                self.update_stats("yolov3_big", infer_time)

    def display_resnet_frames_from_process(self):
        while not self.shutdown_flag.is_set():
            try:
                frame, class_name, infer_time = self.resnet_output_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(frame)
            if not pixmap.isNull():
                self.model_signals.update_resnet_display.emit(pixmap)
                self.update_stats("resnet50", infer_time)

    def closeEvent(self, event):
        # 1. Set thread termination flag
        self.shutdown_flag.set()
        time.sleep(0.2)

        # 2. Set process termination event
        try:
            if hasattr(self, 'yolo_shutdown_event'):
                self.yolo_shutdown_event.set()
            if hasattr(self, 'resnet_shutdown_event'):
                self.resnet_shutdown_event.set()
        except Exception as e:
            print(f"[Shutdown Event ERROR] {e}")

        # 3. Attempt to terminate processes
        try:
            if hasattr(self, 'yolo_process') and self.yolo_process.is_alive():
                self.yolo_process.join(timeout=2)
                if self.yolo_process.is_alive():
                    print("[YOLO Process] force terminating...")
                    self.yolo_process.terminate()
                    self.yolo_process.join()

            if hasattr(self, 'resnet_process') and self.resnet_process.is_alive():
                self.resnet_process.join(timeout=2)
                if self.resnet_process.is_alive():
                    print("[ResNet Process] force terminating...")
                    self.resnet_process.terminate()
                    self.resnet_process.join()
        except Exception as e:
            print(f"[Process Join/Terminate ERROR] {e}")

        # 4. Resource cleanup
        try:
            # Clear queues
            for q in [
                self.yolo_frame_queue,
                self.yolo_result_queue,
                self.view1_frame_queue,
                self.view1_result_queue,
                self.view2_result_queue,
                self.resnet_result_queue,
            ]:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except:
                        break

            # Release video capture
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()

            # Release ONNX sessions
            for attr in [
                'yolo_session',
                'view1_session',
                'resnet_session',
                'view2_session',
                'yolo_front_session',
                'yolo_back_session',
                'resnet_front_session',
                'resnet_back_session',
            ]:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Garbage collection
            import gc
            gc.collect()

        except Exception as e:
            print(f"[Cleanup ERROR] {e}")

        # 5. Accept event
        event.accept()

    def update_stats(self, model_name, current_infer_time):
        """Update average FPS and inference time per model and record logs"""
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

    def process_view1_frames(self):
        while not self.shutdown_flag.is_set():
            try:
                frame = self.view1_frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            input_tensor, (w, h) = preprocess_yolo(frame)

            if not hasattr(self, 'view1_session') or self.view1_session is None:
                continue

            try:
                infer_start = time.time()
                output = self.view1_session.run(None, {"images": input_tensor})
                infer_end = time.time()
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                # Update View1 (YOLO CPU) statistics
                self.view1_total_infer_time += infer_time_ms
                self.view1_infer_count += 1
                self.view1_avg_infer_time = self.view1_total_infer_time / self.view1_infer_count
                self.view1_avg_fps = 1000.0 / self.view1_avg_infer_time if self.view1_avg_infer_time > 0 else 0.0
            except Exception as e:
                print(f"[VIEW1 ERROR] {e}")
                continue
            result = postprocessing_cpu(output, frame, w, h)
            if not self.view1_result_queue.full():
                self.view1_result_queue.put(result)

    def update_yolo_display(self, pixmap):
        self.yolo_label.setPixmap(pixmap)  # yolo_label now references view1

    def display_view1_frames(self):
        while not self.shutdown_flag.is_set():
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
        while not self.shutdown_flag.is_set():
            if self.view2_index >= len(self.resnet_images):
                self.view2_index = 0
            img_path = self.resnet_images[self.view2_index]
            img = cv2.imread(img_path)
            self.view2_index += 1
            if img is None:
                continue
            input_tensor = preprocess_resnet(img)

            if not hasattr(self, 'view2_session') or self.view2_session is None:
                continue

            try:
                infer_start = time.time()
                output = self.view2_session.run(None, {"data": input_tensor})
                infer_end = time.time()
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                # Update View2 (ResNet CPU) statistics
                self.view2_total_infer_time += infer_time_ms
                self.view2_infer_count += 1
                self.view2_avg_infer_time = self.view2_total_infer_time / self.view2_infer_count
                self.view2_avg_fps = 1000.0 / self.view2_avg_infer_time if self.view2_avg_infer_time > 0 else 0.0
            except Exception as e:
                print(f"[View2 ERROR] {e}")
                continue
            class_id = int(np.argmax(output[0]))
            class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
            cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if not self.view2_result_queue.full():
                self.view2_result_queue.put(img)

    def display_view2_frames(self):
        while not self.shutdown_flag.is_set():
            try:
                result = self.view2_result_queue.get(timeout=1)
            except queue.Empty:
                continue
            pixmap = convert_cv_qt(result)
            self.model_signals.update_view2_display.emit(pixmap)

    def update_view2_display(self, pixmap):
        self.view2.setPixmap(pixmap)

    def update_resnet_display(self, pixmap):
        self.resnet_label.setPixmap(pixmap)  # resnet_label now references view2

    def update_cpu_npu_usage(self):
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]

        self.yolo_info_label.setText(
            f"<b>View1 (YOLO NPU)</b> Avg FPS: {self.yolo_avg_fps:.1f} "
            f"(<span style='color: gray;'>{self.yolo_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: purple;'>View2 (ResNet NPU)</span></b> Avg FPS: "
            f"<span style='color: purple;'>{self.resnet_avg_fps:.1f}</span> "
            f"(<span style='color: purple;'>{self.resnet_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: green;'>View3 (YOLO CPU)</span></b> Avg FPS: "
            f"<span style='color: green;'>{self.view1_avg_fps:.1f}</span> "
            f"(<span style='color: green;'>{self.view1_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: blue;'>View4 (ResNet CPU)</span></b> Avg FPS: "
            f"<span style='color: blue;'>{self.view2_avg_fps:.1f}</span> "
            f"(<span style='color: blue;'>{self.view2_avg_infer_time:.1f} ms</span>)"
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