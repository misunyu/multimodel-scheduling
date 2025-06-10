import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
from ultralytics.utils.files import increment_path

REPEAT = 1

target_model = "./yolov3_neubla.onnx"
input_img_name = "./dog.jpg"
input_width = input_height = 608
input_type = "video"
input_video = "./stockholm_1280x720.mp4"

coco_id_to_label = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra",
    23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
}

def preprocess(raw_input_img):
    img_height, img_width = raw_input_img.shape[:2]
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data, (img_width, img_height)

def draw_detections(img, box, score, class_id):
    color = (0, 255, 0)
    x, y, w, h = box
    label = coco_id_to_label.get(class_id, f"ID:{class_id}")
    text = f"{label} {score:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - text_height - 4), (x + text_width, y), color, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
    if indices is not None and len(indices) > 0:
        for idx in indices:
            i = int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])
            draw_detections(original_img, boxes[i], scores[i], class_ids[i])
    return original_img

def process_onnxruntime_cpu(raw_input_img):
    input_img, (img_width, img_height) = preprocess(raw_input_img)
    output = viewer.cpu_total_session.run(None, {"input": input_img})
    result_img = postprocessing(output, raw_input_img, img_width, img_height)
    return result_img

def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qt_image)

class YoloViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv3 Result Viewer")
        self.resize(800, 600)
        layout = QVBoxLayout()
        self.label = QLabel("Starting YOLO Inference...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

        if not Path(input_video).exists():
            raise FileNotFoundError(f"Source path '{input_video}' does not exist.")

        self.cap = cv2.VideoCapture(input_video)
        self.cpu_total_session = ort.InferenceSession(target_model)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_running = True

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Start processing frames automatically
        self.timer.start(30)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return:
            QApplication.quit()
        elif event.key() == Qt.Key.Key_Space:
            if self.is_running:
                self.timer.stop()
                self.is_running = False
            else:
                self.timer.start(30)
                self.is_running = True

    def next_frame(self):
        success, raw_input_img = self.cap.read()
        if success:
            result_img = self.run_inference(raw_input_img)
            pixmap = convert_cv_qt(result_img)
            self.label.setPixmap(pixmap)

    def run_inference(self, raw_input_img):
        input_img, (img_width, img_height) = preprocess(raw_input_img)
        output = self.cpu_total_session.run(None, {"input": input_img})
        result_img = postprocessing(output, raw_input_img, img_width, img_height)
        return result_img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = YoloViewer()
    viewer.show()
    sys.exit(app.exec())

# import time
#
# import cv2
# import numpy as np
# import onnxruntime as ort
# from pathlib import Path
# from ultralytics.utils.files import increment_path
#
#
# REPEAT = 1
#
# target_model = "./yolov3_neubla.onnx"
# input_img_name = "./dog.jpg"
# input_width = input_height = 608
# input_type = "video"
#
# input_video = "./stockholm_1280x720.mp4"
#
# def preprocess(raw_input_img):
#     img_height, img_width = raw_input_img.shape[:2]
#     img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (input_width, input_height))
#     image_data = np.array(img) / 255.0  # normalize
#     image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
#     image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
#
#     return image_data, (img_width, img_height)
#
# coco_id_to_label = {
#     0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
#     8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
#     15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra",
#     23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
#     30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 36: "skateboard",
#     37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
#     45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
#     52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
#     60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
#     67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
#     73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
# }
#
#
# def draw_detections(img, box, score, class_id):
#     """Draw bounding box and English label"""
#     color = (0, 255, 0)
#     x, y, w, h = box
#     label = coco_id_to_label.get(class_id, f"ID:{class_id}")
#     text = f"{label} {score:.2f}"
#
#     cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#
#     # Draw filled rectangle background for text
#     (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     cv2.rectangle(img, (x, y - text_height - 4), (x + text_width, y), color, -1)
#     cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#
#
# def postprocessing(
#     output, original_img, img_width, img_height, save_path="output.jpg",
#     confidence_thres=0.5, iou_thres=0.5
# ):
#     output_box = np.squeeze(output[0])
#     output_label = np.squeeze(output[1])
#
#     rows = output_box.shape[0]
#     boxes, scores, class_ids = [], [], []
#
#     x_factor = img_width / input_width
#     y_factor = img_height / input_height
#
#     for i in range(rows):
#         conf = output_box[i][4]
#         if conf >= confidence_thres:
#             left, top, right, bottom = output_box[i][:4]
#             width = int((right - left) * x_factor)
#             height = int((bottom - top) * y_factor)
#             left = int(left * x_factor)
#             top = int(top * y_factor)
#             boxes.append([left, top, width, height])
#             scores.append(conf)
#             class_ids.append(int(output_label[i]))
#
#     indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
#
#     if indices is not None and len(indices) > 0:
#         for idx in indices:
#             i = int(idx) if isinstance(idx, (int, np.integer)) else int(idx[0])
#             draw_detections(original_img, boxes[i], scores[i], class_ids[i])
#
#     cv2.imwrite(save_path, original_img)
#     print(f"\nResult image saved to: {save_path}")
#
# def process_onnxruntime_cpu(raw_input_img):
#     print("==== process_onnxruntime_cpu() ====")
#
#     start_time = time.time()
#     cpu_total_session = ort.InferenceSession(target_model)
#     end_time = time.time()
#     elapsed_time = (end_time - start_time) * 1000
#     print(f"session creation time: {elapsed_time:.2f} ms")
#
#     for i in range(REPEAT):
#         start_time = time.time()
#         input_img, (img_width, img_height) = preprocess(raw_input_img)
#
#         end_time = time.time()
#         prep_time = (end_time - start_time) * 1000
#         print(f"preprocessing time: {prep_time:.2f} ms")
#
#         cpu_start_time = time.time()
#         output = cpu_total_session.run(None, {"input": input_img})
#         end_time = time.time()
#         elapsed_cpu_time = (end_time - cpu_start_time) * 1000
#         print(f"inference time: {elapsed_cpu_time:.2f} ms\n")
#
#         postp_start_time = time.time()
#         postprocessing(output, raw_input_img, img_width, img_height, save_path=f"yolo_result.jpg")
#         end_time = time.time()
#         postp_elapsed_time = ((end_time - postp_start_time) * 1000)/REPEAT
#         print(f"Yolov3 Postprocessing time: {postp_elapsed_time} ms")
#
#     # 출력 결과 확인
#     #for i, output in enumerate(outputs):
#     #    print(f"Output {i}: {output.shape}")
#
# def main():
#
#     if not Path(input_video).exists():
#         raise FileNotFoundError(f"Source path '{input_video}' does not exist.")
#
#     # Video setup
#     videocapture = cv2.VideoCapture(input_video)
#     frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
#     fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
#
#     # Output setup
#     do_overwrite = False
#     save_dir = increment_path(Path("test_video_output") / "exp", do_overwrite)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_writer = cv2.VideoWriter(
#         str(save_dir / f"{Path(input_video).stem}.mp4"),
#         fourcc,
#         fps,
#         (frame_width, frame_height),
#     )
#
#     def get_next_img():
#         if input_type in ("video"):
#             if videocapture.isOpened():
#                 success, raw_input_img = videocapture.read()
#                 #breakpoint()
#             else:
#                 success, raw_input_img = False, None
#         elif input_type == "image":
#             raw_input_img = cv2.imread(input_img_name)
#             success = True
#
#         if not success:
#             return None
#         return raw_input_img
#
#     input_type = "video"
#
#     raw_input_img = get_next_img()
#     process_onnxruntime_cpu(raw_input_img)
#
#
# if __name__ == "__main__":
#     main()



