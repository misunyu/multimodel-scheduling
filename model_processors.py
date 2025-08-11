"""
Model processing functions for the multimodel scheduling application.
"""
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import queue

import npu

# Import local modules
from image_processing import (
    yolo_preprocess_local, 
    resnet50_preprocess_local, 
    yolo_postprocess_cpu, 
    yolo_postprocess_npu
)

# Load ImageNet class labels
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

def video_reader_process(video_path, frame_queue, shutdown_event, max_queue_size=10):
    """
    Process for reading video frames and putting them in a queue.
    
    Args:
        video_path: Path to the video file
        frame_queue: Queue to put frames into
        shutdown_event: Event to signal shutdown
        max_queue_size: Maximum size of the queue
    """
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

        try:
            # Avoid qsize() which may not be implemented on some platforms (e.g., macOS)
            frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame if queue is full to avoid blocking
            pass
        time.sleep(frame_delay)

    cap.release()

def run_yolo_cpu_process(input_queue, output_queue, shutdown_event):
    """
    Process for running YOLO model on CPU.
    
    Args:
        input_queue: Queue to get input frames from
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
    """
    try:
        # Load the YOLO model
        print(f"[YOLO CPU] Loading model...")
        session = ort.InferenceSession("models/yolov3_small/model/yolov3_small.onnx")
        print(f"[YOLO CPU] Model loaded successfully")
        
        while not shutdown_event.is_set():
            try:
                frame = input_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            input_tensor, (w, h) = yolo_preprocess_local(frame)
            
            try:
                infer_start = time.time()
                output = session.run(None, {"images": input_tensor})
                infer_end = time.time()
                
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                result = yolo_postprocess_cpu(output, frame, w, h)
                output_queue.put((result, infer_time_ms))
                
            except Exception as e:
                print(f"[YOLO CPU Process ERROR] {e}")
                continue
                
    except Exception as e:
        print(f"[YOLO CPU Process ERROR] {e}")

def run_resnet_cpu_process(image_dir, output_queue, shutdown_event):
    """
    Process for running ResNet model on CPU.
    
    Args:
        image_dir: Directory containing images to process
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
    """
    try:
        # Load the ResNet model
        session = ort.InferenceSession("models/resnet50_small/model/resnet50_small.onnx")
        
        # Get list of image files
        image_files = [os.path.join(image_dir, f)
                      for f in os.listdir(image_dir)
                      if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not image_files:
            print("[ResNet CPU] No images found")
            return
            
        index = 0
        while not shutdown_event.is_set():
            if index >= len(image_files):
                index = 0
                
            img = cv2.imread(image_files[index])
            index += 1
            if img is None:
                continue
                
            input_tensor = resnet50_preprocess_local(img)
            
            try:
                infer_start = time.time()
                output = session.run(None, {"data": input_tensor})
                infer_end = time.time()
                
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                class_id = int(np.argmax(output[0]))
                class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
                cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                output_queue.put((img, class_name, infer_time_ms))
                
            except Exception as e:
                print(f"[ResNet CPU Process ERROR] {e}")
                continue
                
    except Exception as e:
        print(f"[ResNet CPU Process ERROR] {e}")

def run_yolo_npu_process(input_queue, output_queue, shutdown_event, npu_id=0):
    """
    Process for running YOLO model on NPU.
    
    Args:
        input_queue: Queue to get input frames from
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        npu_id: NPU device ID
    """
    try:
        # Import NPU-specific functions only when needed
        from npu import (
            initialize_driver, 
            close_driver, 
            send_receive_data_npu, 
            yolo_prepare_onnx_model
        )
        
        front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
            "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
        )

        driver = initialize_driver(npu_id, "./models/yolov3_small/npu_code/yolov3_small_neubla_p1.o")
        frame_delay = 1.0 / 30.0

        while not shutdown_event.is_set():
            try:
                frame = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            input_tensor, (w, h) = yolo_preprocess_local(frame)
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
            result_img, drawn_boxes = yolo_postprocess_npu(output, frame, w, h)

            if drawn_boxes:
                infer_time_ms = (infer_end - infer_start) * 1000.0
                output_queue.put((result_img, infer_time_ms))

    except Exception as e:
        print(f"[YOLO NPU Process ERROR] {e}")
    finally:
        # Import close_driver only when needed
        try:
            from npu import close_driver
            close_driver(driver)
        except:
            pass

def run_resnet_npu_process(image_dir, output_queue, shutdown_event, npu_id=1):
    """
    Process for running ResNet model on NPU.
    
    Args:
        image_dir: Directory containing images to process
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        npu_id: NPU device ID
    """
    try:
        # Import NPU-specific functions only when needed
        from npu import (
            initialize_driver, 
            close_driver, 
            send_receive_data_npu, 
            resnet50_prepare_onnx_model,
            resnet50_preprocess
        )
        
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

        driver = initialize_driver(npu_id, "./models/resnet50_small/npu_code/resnet50_small_neubla_p1.o")

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
            class_name = imagenet_classes[max_index] if max_index < len(imagenet_classes) else f"Class ID: {max_index}"
            cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            output_queue.put((img, class_name, (infer_end - infer_start) * 1000.0))

    except Exception as e:
        print(f"[ResNet NPU Process ERROR] {e}")
    finally:
        # Import close_driver only when needed
        try:
            from npu import close_driver
            close_driver(driver)
        except:
            pass