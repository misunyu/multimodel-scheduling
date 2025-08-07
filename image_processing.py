"""
Image processing functions for the multimodel scheduling application.
"""
import cv2
import numpy as np

# Constants
input_width = input_height = 608

# COCO class labels
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

def image_preprocess(raw_input_img, target_width, target_height):
    """
    Preprocess an image for neural network input.
    
    Args:
        raw_input_img: Raw input image (numpy array)
        target_width: Target width for resizing
        target_height: Target height for resizing
        
    Returns:
        Preprocessed image data as numpy array
    """
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_width, target_height))
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data

def yolo_preprocess_local(raw_input_img):
    """
    Preprocess an image for YOLO model input.
    
    Args:
        raw_input_img: Raw input image (numpy array)
        
    Returns:
        Tuple of (preprocessed image data, original dimensions)
    """
    image_data = image_preprocess(raw_input_img, input_width, input_height)
    return image_data, (raw_input_img.shape[1], raw_input_img.shape[0])

def resnet50_preprocess_local(raw_input_img):
    """
    Preprocess an image for ResNet50 model input.
    
    Args:
        raw_input_img: Raw input image (numpy array)
        
    Returns:
        Preprocessed image data as numpy array
    """
    return image_preprocess(raw_input_img, 224, 224)

def draw_detection_boxes(img, box, score, class_id):
    """
    Draw detection boxes on an image.
    
    Args:
        img: Image to draw on (numpy array)
        box: Bounding box coordinates [x, y, w, h]
        score: Confidence score
        class_id: Class ID
        
    Returns:
        None (modifies img in-place)
    """
    class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"ID:{class_id}"
    label = f"{class_name} {score:.2f}"
    x, y, w, h = box

    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - text_height - 4), (x + text_width, y), color, -1)
    cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def yolo_postprocess_cpu(output, original_img, img_width, img_height, confidence_thres=0.5, iou_thres=0.5):
    """
    Post-process YOLO model output (CPU version).
    
    Args:
        output: Model output
        original_img: Original image (numpy array)
        img_width: Original image width
        img_height: Original image height
        confidence_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        
    Returns:
        Image with detection boxes drawn
    """
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
            draw_detection_boxes(original_img, boxes[i], scores[i], class_ids[i])
    return original_img

def yolo_postprocess_npu(output, original_img, img_width, img_height, confidence_thres=0.5, iou_thres=0.5):
    """
    Post-process YOLO model output (NPU version).
    
    Args:
        output: Model output
        original_img: Original image (numpy array)
        img_width: Original image width
        img_height: Original image height
        confidence_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        
    Returns:
        Tuple of (image with detection boxes drawn, list of drawn boxes)
    """
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
            draw_detection_boxes(original_img, boxes[i], scores[i], class_ids[i])
            drawn_boxes.append(boxes[i])

    return original_img, drawn_boxes