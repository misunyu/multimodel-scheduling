"""
Model processing functions for the multimodel scheduling application.
"""
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

# Modularized timing/logging utilities (moved to dedicated module)
from timing_utils import log_model_load, log_inference

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

def run_yolo_cpu_process(input_queue, output_queue, shutdown_event, view_name=None):
    """
    Process for running YOLO model on CPU.
    
    Args:
        input_queue: Queue to get input frames from
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        view_name: Optional view identifier for logging
    """
    try:
        # Load the YOLO model
        print(f"[YOLO CPU] Loading model...")
        load_start = time.time()
        # Create ONNX Runtime session with reduced log verbosity to suppress shape merge warnings
        so = ort.SessionOptions()
        # 0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL
        so.log_severity_level = 3
        try:
            session = ort.InferenceSession(
                "models/yolov3_small/model/yolov3_small.onnx",
                sess_options=so,
                providers=["CPUExecutionProvider"]
            )
        except TypeError:
            # Fallback for older onnxruntime without providers argument
            session = ort.InferenceSession(
                "models/yolov3_small/model/yolov3_small.onnx",
                sess_options=so
            )
        # Determine input/output dynamically
        try:
            inputs = session.get_inputs()
            # Pick the 4D tensor input as image input if available
            img_input = None
            for inp in inputs:
                shp = inp.shape
                if isinstance(shp, (list, tuple)) and len(shp) == 4:
                    img_input = inp
                    break
            if img_input is None and inputs:
                img_input = inputs[0]
            input_name = img_input.name if img_input else "images"
            in_shape = img_input.shape if img_input else [1, 3, 608, 608]
            # expect NCHW by default
            input_w = int(in_shape[3]) if len(in_shape) == 4 and isinstance(in_shape[3], int) else 608
            input_h = int(in_shape[2]) if len(in_shape) == 4 and isinstance(in_shape[2], int) else 608
            # Detect optional image_shape input
            image_shape_input = None
            image_shape_dtype = np.float32
            for inp in inputs:
                if 'image_shape' in inp.name:
                    image_shape_input = inp
                    # map ORT type string to numpy dtype
                    t = (inp.type or '').lower()
                    if 'int64' in t:
                        image_shape_dtype = np.int64
                    elif 'int32' in t:
                        image_shape_dtype = np.int32
                    else:
                        image_shape_dtype = np.float32
                    break
        except Exception:
            input_name = "images"
            input_w, input_h = 608, 608
            image_shape_input = None
            image_shape_dtype = np.float32
        load_end = time.time()
        load_time_ms = (load_end - load_start) * 1000.0
        print(f"[YOLO CPU] Model loaded successfully")

        # Log model load
        log_model_load(
            pipeline="yolo",
            device="CPU",
            view=view_name,
            model="yolov3_small",
            model_load_time_ms=load_time_ms,
        )
        
        while not shutdown_event.is_set():
            try:
                item = input_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
                    frame, enqueue_ts = item
                else:
                    frame = item
                    enqueue_ts = None
            except queue.Empty:
                continue
                
            pre_s = time.time()
            # Waiting time until preprocessing begins
            wait_ms = ((pre_s - enqueue_ts) * 1000.0) if enqueue_ts else 0.0
            input_tensor, meta = yolo_preprocess_local(frame, (input_w, input_h))
            pre_e = time.time()
            pre_ms = (pre_e - pre_s) * 1000.0
            
            try:
                # Build input feed dict, include image_shape if required by model
                feeds = {input_name: input_tensor}
                if image_shape_input is not None:
                    h0 = int(meta.get('orig_h', frame.shape[0]))
                    w0 = int(meta.get('orig_w', frame.shape[1]))
                    img_shape_val = np.array([[h0, w0]], dtype=image_shape_dtype)
                    feeds[image_shape_input.name] = img_shape_val
                infer_start = time.time()
                output = session.run(None, feeds)
                infer_end = time.time()
                
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                post_s = time.time()
                result = yolo_postprocess_cpu(output, frame, meta)
                post_e = time.time()
                post_ms = (post_e - post_s) * 1000.0

                # Log per-frame inference timing
                log_inference(
                    pipeline="yolo",
                    device="CPU",
                    view=view_name,
                    model="yolov3_small",
                    preprocess_time_ms=pre_ms,
                    inference_time_ms=infer_time_ms,
                    postprocess_time_ms=post_ms,
                    wait_to_preprocess_ms=wait_ms,
                )
                
                output_queue.put((result, infer_time_ms, wait_ms))
                
            except Exception as e:
                print(f"[YOLO CPU Process ERROR] {e}")
                continue
                
    except Exception as e:
        print(f"[YOLO CPU Process ERROR] {e}")

def run_resnet_cpu_process(input_queue, output_queue, shutdown_event, view_name=None):
    """
    Process for running ResNet model on CPU.
    
    Args:
        image_dir: Directory containing images to process
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        view_name: Optional view identifier for logging
    """
    try:
        # Load the ResNet model
        load_start = time.time()
        session = ort.InferenceSession("models/resnet50_small/model/resnet50_small.onnx")
        load_end = time.time()
        load_time_ms = (load_end - load_start) * 1000.0

        # Determine input/output names and expected layout dynamically
        try:
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            input_name = inputs[0].name if inputs else "data"
            output_name = outputs[0].name if outputs else None
            input_shape = inputs[0].shape if inputs else None
            # Heuristic for layout: NCHW if channel dim is 3 at index 1; NHWC if last dim is 3
            layout = "NCHW"
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
                c_dim = input_shape[1]
                last_dim = input_shape[3]
                if last_dim == 3 or (isinstance(last_dim, str) and last_dim.upper() in ("C", "CHANNEL", "CHANNELS")):
                    layout = "NHWC"
                elif c_dim == 3 or (isinstance(c_dim, str) and c_dim.upper() in ("C", "CHANNEL", "CHANNELS")):
                    layout = "NCHW"
            print(f"[ResNet CPU] Model IO - input_name={input_name}, output_name={output_name}, input_shape={input_shape}, layout={layout}")
        except Exception as e:
            print(f"[ResNet CPU] Warning: could not introspect model IO names: {e}")
            input_name = "data"
            output_name = None
            layout = "NCHW"

        # Log model load
        log_model_load(
            pipeline="resnet50",
            device="CPU",
            view=view_name,
            model="resnet50_small",
            model_load_time_ms=load_time_ms,
        )
        while not shutdown_event.is_set():
            try:
                item = input_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
                    img, enqueue_ts = item
                else:
                    img = item
                    enqueue_ts = None
            except queue.Empty:
                continue

            pre_s = time.time()
            # Waiting time until preprocessing begins (for parity with YOLO)
            wait_ms = ((pre_s - enqueue_ts) * 1000.0) if enqueue_ts else 0.0
            tensor_nchw = resnet50_preprocess_local(img)
            # Adapt to model's expected layout
            if layout == "NHWC":
                input_tensor = np.transpose(tensor_nchw, (0, 2, 3, 1))
            else:
                input_tensor = tensor_nchw
            pre_e = time.time()
            pre_ms = (pre_e - pre_s) * 1000.0
            
            try:
                infer_start = time.time()
                outputs = session.run([output_name] if output_name else None, {input_name: input_tensor})
                infer_end = time.time()
                
                infer_time_ms = (infer_end - infer_start) * 1000.0
                
                post_s = time.time()
                logits = outputs[0]
                logits = np.squeeze(logits)
                class_id = int(np.argmax(logits))
                class_name = imagenet_classes[class_id] if class_id < len(imagenet_classes) else f"Class ID: {class_id}"
                cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                post_e = time.time()
                post_ms = (post_e - post_s) * 1000.0

                # Log per-frame inference timing
                log_inference(
                    pipeline="resnet50",
                    device="CPU",
                    view=view_name,
                    model="resnet50_small",
                    preprocess_time_ms=pre_ms,
                    inference_time_ms=infer_time_ms,
                    postprocess_time_ms=post_ms,
                    wait_to_preprocess_ms=wait_ms,
                )
                
                output_queue.put((img, class_name, infer_time_ms))
                
            except Exception as e:
                print(f"[ResNet CPU Process ERROR] {e}")
                continue
                
    except Exception as e:
        print(f"[ResNet CPU Process ERROR] {e}")

def run_yolo_npu_process(input_queue, output_queue, shutdown_event, npu_id=0, view_name=None):
    """
    Process for running YOLO model on NPU.
    
    Args:
        input_queue: Queue to get input frames from
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        npu_id: NPU device ID
        view_name: Optional view identifier for logging
    """
    try:
        # Import NPU-specific functions only when needed
        from npu import (
            initialize_driver, 
            close_driver, 
            send_receive_data_npu, 
            yolo_prepare_onnx_model
        )
        
        driver = None
        try:
            host_load_s = time.time()
            front_sess, back_sess, (scale, zero_point) = yolo_prepare_onnx_model(
                "../yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx"
            )
            host_load_e = time.time()
            host_model_load_ms = (host_load_e - host_load_s) * 1000.0
        except Exception as e:
            print(f"[YOLO NPU INIT ERROR] npu_id = {npu_id}, Host model preparation failed: {e}")
            raise

        try:
            npu_load_s = time.time()
            driver = initialize_driver(npu_id, "./models/yolov3_small/npu_code/yolov3_small_neubla_p1.o")
            npu_load_e = time.time()
            npu_memory_load_time_ms = (npu_load_e - npu_load_s) * 1000.0
        except Exception as e:
            print(f"[YOLO NPU INIT ERROR] NPU driver initialization failed: {e}")
            raise

        # Log model load (host + NPU memory)
        log_model_load(
            pipeline="yolo",
            device=f"NPU{npu_id}",
            view=view_name,
            model="yolov3_small",
            model_load_time_ms=host_model_load_ms,
            npu_memory_load_time_ms=npu_memory_load_time_ms,
        )


        while not shutdown_event.is_set():
            try:
                item = input_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
                    frame, enqueue_ts = item
                else:
                    frame = item
                    enqueue_ts = None
            except queue.Empty:
                continue

            pre_s = time.time()
            # Waiting time until preprocessing begins
            wait_ms = ((pre_s - enqueue_ts) * 1000.0) if enqueue_ts else 0.0
            # Determine input size from front_sess
            try:
                finp = front_sess.get_inputs()[0]
                fshape = finp.shape if finp else [1,3,608,608]
                f_w = int(fshape[3]) if len(fshape)==4 and isinstance(fshape[3], int) else 608
                f_h = int(fshape[2]) if len(fshape)==4 and isinstance(fshape[2], int) else 608
            except Exception:
                f_w, f_h = 608, 608
            input_tensor, meta = yolo_preprocess_local(frame, (f_w, f_h))
            pre_e = time.time()
            pre_ms = (pre_e - pre_s) * 1000.0
            infer_start = time.time()

            try:
                # Host front inference
                front_output = front_sess.run(None, {"input": input_tensor})[0]
                input_data = front_output.tobytes()
            except Exception as e:
                print(f"[YOLO NPU INFERENCE ERROR] Front session failed: {e}")
                continue

            try:
                # Data transfer to/from NPU
                raw_outputs = send_receive_data_npu(driver, input_data, 3 * f_w * f_h)
                output_data = [np.frombuffer(buf, dtype=np.uint8) for buf in raw_outputs]
            except Exception as e:
                print(f"[YOLO NPU DATA TRANSFER ERROR] send/receive failed: {e}")
                continue

            try:
                # Dequantization and back session
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
                        print(f"[YOLO NPU BACKEND ERROR] insufficient data for {name}, expected {needed_size}, got {data.size}")
                        raise ValueError("Invalid data size")
                    back_feeds[name] = data[:needed_size].reshape(shape_dict[name])

                output = back_sess.run(None, back_feeds)
            except Exception as e:
                print(f"[YOLO NPU BACKEND ERROR] {e}")
                continue

            infer_end = time.time()
            post_s = time.time()
            result_img, drawn_boxes = yolo_postprocess_npu(output, frame, meta)
            post_e = time.time()
            post_ms = (post_e - post_s) * 1000.0
            
            infer_time_ms = (infer_end - infer_start) * 1000.0
            if drawn_boxes:
                # Log per-frame inference timing only when we have valid detections
                log_inference(
                    pipeline="yolo",
                    device=f"NPU{npu_id}",
                    view=view_name,
                    model="yolov3_small",
                    preprocess_time_ms=pre_ms,
                    inference_time_ms=infer_time_ms,
                    postprocess_time_ms=post_ms,
                    wait_to_preprocess_ms=wait_ms,
                )
                output_queue.put((result_img, infer_time_ms, wait_ms))

    except Exception as e:
        print(f"[YOLO NPU Process ERROR] {e}")
    finally:
        # Import close_driver only when needed
        try:
            from npu import close_driver
            close_driver(driver)
        except:
            pass

def run_resnet_npu_process(input_queue, output_queue, shutdown_event, npu_id=1, view_name=None):
    """
    Process for running ResNet model on NPU.
    
    Args:
        image_dir: Directory containing images to process
        output_queue: Queue to put output results into
        shutdown_event: Event to signal shutdown
        npu_id: NPU device ID
        view_name: Optional view identifier for logging
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
        driver = None
        try:
            host_load_s = time.time()
            front_sess, back_sess, params = resnet50_prepare_onnx_model(
                "../resnet/resnet50-0676ba61_opset12.neubla_u8_lwq_percentile.onnx"
            )
            host_load_e = time.time()
            host_model_load_ms = (host_load_e - host_load_s) * 1000.0
        except Exception as e:
            print(f"[ResNet NPU INIT ERROR] Host model preparation failed: {e}")
            raise

        scale = params['/0/avgpool/GlobalAveragePool_output_0_scale'] * params['0.fc.weight_scale']
        zp_act = params['/0/avgpool/GlobalAveragePool_output_0_zero_point']
        zp_w = params['0.fc.weight_zero_point']
        scale_out = params['/0/fc/Gemm_output_0_scale']
        zp_out = params['/0/fc/Gemm_output_0_zero_point']
        weight_q = params['0.fc.weight_quantized'].T.astype(np.int32)

        try:
            npu_load_s = time.time()
            driver = initialize_driver(npu_id, "./models/resnet50_small/npu_code/resnet50_small_neubla_p1.o")
            npu_load_e = time.time()
            npu_memory_load_time_ms = (npu_load_e - npu_load_s) * 1000.0
        except Exception as e:
            print(f"[ResNet NPU INIT ERROR] NPU driver initialization failed: {e}")
            raise

        # Log model load (host + NPU memory)
        log_model_load(
            pipeline="resnet50",
            device=f"NPU{npu_id}",
            view=view_name,
            model="resnet50_small",
            model_load_time_ms=host_model_load_ms,
            npu_memory_load_time_ms=npu_memory_load_time_ms,
        )

        while not shutdown_event.is_set():
            try:
                item = input_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], float):
                    img, enqueue_ts = item
                else:
                    img = item
                    enqueue_ts = None
            except queue.Empty:
                continue

            pre_s = time.time()
            # Waiting time until preprocessing begins
            wait_ms = ((pre_s - enqueue_ts) * 1000.0) if enqueue_ts else 0.0
            input_data = front_sess.run(None, {"input": resnet50_preprocess(img)})[0].tobytes()
            pre_e = time.time()
            pre_ms = (pre_e - pre_s) * 1000.0
            infer_start = time.time()
            try:
                raw_outputs = send_receive_data_npu(driver, input_data, 3 * 224 * 224)
                output_data = np.frombuffer(raw_outputs[0], dtype=np.uint8)
            except Exception as e:
                print(f"[ResNet NPU DATA TRANSFER ERROR] send/receive failed: {e}")
                # Skip this frame and continue
                continue

            try:
                back_output = back_sess.run(None, {"input": output_data.reshape(1, -1)})
                output = back_output[0]
                max_index = int(np.argmax(output))
            except Exception as e:
                # Fallback to manual computation if back session fails
                output = np.matmul(output_data.astype(np.int32), weight_q)
                output -= zp_act * np.sum(weight_q, axis=0)
                output -= zp_w * np.sum(output_data, axis=0)
                output += zp_act * zp_w
                output = np.round(output * scale / scale_out) + zp_out
                output = output.astype(np.uint8)
                max_index = int(np.argmax(output))

            infer_end = time.time()
            post_s = time.time()
            class_name = imagenet_classes[max_index] if max_index < len(imagenet_classes) else f"Class ID: {max_index}"
            cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            post_e = time.time()
            post_ms = (post_e - post_s) * 1000.0

            infer_ms = (infer_end - infer_start) * 1000.0
            # Log per-frame inference timing
            log_inference(
                pipeline="resnet50",
                device=f"NPU{npu_id}",
                view=view_name,
                model="resnet50_small",
                preprocess_time_ms=pre_ms,
                inference_time_ms=infer_ms,
                postprocess_time_ms=post_ms,
                wait_to_preprocess_ms=wait_ms,
            )

            output_queue.put((img, class_name, infer_ms))

    except Exception as e:
        print(f"[ResNet NPU Process ERROR] {e}")
    finally:
        # Import close_driver only when needed
        try:
            from npu import close_driver
            close_driver(driver)
        except:
            pass