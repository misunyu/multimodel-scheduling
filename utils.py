"""
Utility functions for the multimodel scheduling application.
"""
import os
import time
import json
import threading
import psutil
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime

# Constants
LOG_DIR = "./logs"
MAX_LOG_ENTRIES = 500

def async_log(model_name, infer_time_ms, avg_fps, log_enabled=0):
    """
    Asynchronously log model performance data to a JSON file.
    
    Args:
        model_name: Name of the model
        infer_time_ms: Inference time in milliseconds
        avg_fps: Average frames per second
        log_enabled: Flag to control whether logging is enabled
    """
    if not log_enabled:
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

def create_x_image(width=640, height=480):
    """
    Create an image with a black background and a white X across it.
    
    Args:
        width: Width of the image
        height: Height of the image
        
    Returns:
        A numpy array representing the image
    """
    # Create a black image
    img = np.zeros((height, width, 3), np.uint8)
    
    # Draw a white X
    cv2.line(img, (0, 0), (width, height), (255, 255, 255), 5)
    cv2.line(img, (0, height), (width, 0), (255, 255, 255), 5)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "No model specified"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    return img

def convert_cv_to_qt(cv_img):
    """
    Convert OpenCV image to Qt pixmap.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QPixmap object
    """
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    try:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    except Exception as e:
        print(f"[convert_cv_to_qt ERROR] {e}")
        return QPixmap()

def get_cpu_metrics(interval=0):
    """
    Get CPU performance metrics.
    
    Args:
        interval: Time interval for CPU percent calculation
        
    Returns:
        Dictionary containing CPU metrics
    """
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