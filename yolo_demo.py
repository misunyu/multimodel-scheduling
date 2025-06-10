import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from NeublaDriver import NeublaDriver
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import colors

parser = argparse.ArgumentParser(prog="Yolo Demo")
parser.add_argument(
    "-b",
    "--binary",
    default="yolov3/yolov3_half.o",
    help="Neubla binary path to inference",
)
parser.add_argument(
    "-m",
    "--model",
    default="yolov3/yolov3_d53_mstrain-608_273e_coco_optim_opset12.neubla_u8_lwq_movingaverage.onnx",
    help="ONNX model to inference",
)
parser.add_argument("-i", "--image", default="yolov3/dog.jpg", help="Target image")
parser.add_argument(
    "-v", "--video", default="yolov3/store-aisle-detection.mp4", help="Target video"
)
parser.add_argument(
    "-t",
    "--type",
    default="camera",
    choices=["video", "image", "camera"],
    help="Target input type",
)

args = parser.parse_args()
obj_file_name = args.binary
target_model = args.model
input_image_name = args.image
input_video = args.video
input_type = args.type
view_img = True
save_img = False
MAX_FRAME = -1


classes = yaml_load(check_yaml("coco128.yaml"))["names"]
color_palette = [colors(i, True) for i in range(len(classes))]
input_width = input_height = 608


def prepare_onnx_model(
    target_model,
    front_input_names=["input"],
    front_output_names=["input_quantized"],
    target_dequant_layer_name=[
        "onnx::Transpose_684_DequantizeLinear",
        "onnx::Transpose_688_DequantizeLinear",
        "onnx::Transpose_692_DequantizeLinear",
    ],
    back_input_names=[
        "onnx::Transpose_684",
        "onnx::Transpose_688",
        "onnx::Transpose_692",
    ],
    back_output_names=["dets", "labels"],
):

    ## Prepare quantize layer(front model), postprocess layers(back model) and scale factors of dequant layers

    onnx_model = onnx.load(target_model)
    onnx.checker.check_model(onnx_model)

    onnx_graph = onnx_model.graph
    onnx_graph_nodes = onnx_graph.node

    front_output_path = target_model + ".front"
    back_output_path = target_model + ".back"

    # Get front model
    onnx.utils.extract_model(
        target_model, front_output_path, front_input_names, front_output_names
    )
    front_sess = ort.InferenceSession(front_output_path)

    # Get scale factor
    scale = {}
    zero_point = {}
    for idx, node in enumerate(onnx_graph_nodes):
        if node.name in target_dequant_layer_name:
            factors = onnx_model.graph.initializer
            for idx, init in enumerate(factors):
                if factors[idx].name == node.input[1]:
                    scale[node.name] = onnx.numpy_helper.to_array(factors[idx])
                elif factors[idx].name == node.input[2]:
                    zero_point[node.name] = onnx.numpy_helper.to_array(factors[idx])

    # Get back model
    onnx.utils.extract_model(
        target_model, back_output_path, back_input_names, back_output_names
    )
    back_sess = ort.InferenceSession(back_output_path)

    return front_sess, back_sess, (scale, zero_point)


def draw_detections(img, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """

    # Extract the coordinates of the bounding box
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    color = color_palette[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{classes[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


## ref: `ultralytics/examples/YOLOv8-ONNXRuntime/main.py`
def preprocess(raw_input_img):
    img_height, img_width = raw_input_img.shape[:2]
    img = cv2.cvtColor(raw_input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    image_data = np.array(img) / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    return image_data, (img_width, img_height)


def postprocess(
    input_image, output, img_width, img_height, confidence_thres=0.5, iou_thres=0.5
):
    # output layout: ( (List[ (left, top, right, bottom, confidence) ]), (List[ label ]) )
    output_box = np.squeeze(output[0])
    output_label = np.squeeze(output[1])

    rows = output_box.shape[0]

    boxes = []
    scores = []
    class_ids = []

    x_factor = img_width / input_width
    y_factor = img_height / input_height

    for i in range(rows):
        max_score = output_box[i][4]
        if max_score >= confidence_thres:
            class_id = output_label[i]
            left, top, right, bottom = (
                output_box[i][0],
                output_box[i][1],
                output_box[i][2],
                output_box[i][3],
            )
            width = int((right - left) * x_factor)
            height = int((bottom - top) * y_factor)
            left = int(left * x_factor)
            top = int(top * y_factor)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        draw_detections(input_image, box, score, class_id)

    return input_image


def main():  # noqa: C901
    if input_type == "video":
        # Prepare video
        # Check video input path
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Source path '{input_video}' does not exist.")

        # Video setup
        videocapture = cv2.VideoCapture(input_video)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

        # Output setup
        do_overwrite = False
        save_dir = increment_path(Path("test_video_output") / "exp", do_overwrite)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(input_video).stem}.mp4"),
            fourcc,
            fps,
            (frame_width, frame_height),
        )
    elif input_type == "camera":
        videocapture = cv2.VideoCapture(0)
        setting_num = 2
        if setting_num == 1:
            videocapture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )
            videocapture.set(cv2.CAP_PROP_FPS, 60)
        elif setting_num == 2:
            # w, h = 1280, 720 #1920, 1080 #640, 480
            #w, h = 1920, 1080
            w, h = 1280, 720
            #videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
            videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            videocapture.set(cv2.CAP_PROP_BUFFERSIZE, 2.0)
            videocapture.set(cv2.CAP_PROP_FPS, 120)
        print(videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(videocapture.get(cv2.CAP_PROP_FPS))
        print(videocapture.get(cv2.CAP_PROP_BUFFERSIZE))

    def get_next_img():
        if input_type in ("video", "camera"):
            if videocapture.isOpened():
                success, raw_input_img = videocapture.read()
                #breakpoint()
            else:
                success, raw_input_img = False, None
        elif input_type == "image":
            raw_input_img = cv2.imread(input_image_name)
            success = True

        if not success:
            return None
        return raw_input_img

    # Inference
    front_sess, back_sess, (scale, zero_point) = prepare_onnx_model(target_model)

    driver = NeublaDriver()

    assert driver.Init() == 0

    try:
        assert driver.LoadModel(obj_file_name) == 0
        print("LoadModel Compelete")

        # warm up
        raw_input_img = get_next_img()
        input_img, (img_width, img_height) = preprocess(raw_input_img)
        input_data = front_sess.run(None, {"input": input_img})[
            0
        ].tostring()  # to binary string
        assert driver.SendInput(input_data, 3 * 608 * 608) == 0
        assert driver.Launch() == 0

        fps_log = [0 for i in range(150)]
        frame_start = time.time()
        start = frame_start
        delay_print_ui = 0

        def get_elapsed(start):
            now = time.time()
            elapsed = now - start
            return elapsed * 1000, now

        frame_num = 0
        while True:
            frame_num += 1
            if MAX_FRAME > 0 and frame_num >= MAX_FRAME:
                break
            raw_input_img_prev = raw_input_img
            raw_input_img = get_next_img()
            if raw_input_img is None:
                break
            delay_video_open, start = get_elapsed(start)

            input_img, (img_width, img_height) = preprocess(raw_input_img)
            delay_preprocess, start = get_elapsed(start)

            input_data = front_sess.run(None, {"input": input_img})[
                0
            ].tostring()  # to binary string
            delay_front_process, start = get_elapsed(start)

            assert driver.Wait() == 0
            delay_front_to_wait, start = get_elapsed(start)

            assert driver.SendInput(input_data, 3 * 608 * 608) == 0
            delay_wait_to_data_sending, start = get_elapsed(start)

            assert driver.Launch() == 0

            raw_outputs = driver.ReceiveOutputs()
            delay_output_receiveing, start = get_elapsed(start)

            output_data = [
                np.frombuffer(output, dtype=np.uint8) for output in raw_outputs
            ]
            print(output_data[0])
            output_dequant_data = [
                (data.astype(np.float32) - zero_point[name]) * scale[name]
                for name, data in zip(
                    [
                        "onnx::Transpose_684_DequantizeLinear",
                        "onnx::Transpose_688_DequantizeLinear",
                        "onnx::Transpose_692_DequantizeLinear",
                    ],
                    output_data,
                )
            ]
            delay_dequant, start = get_elapsed(start)

            back_feeds = {
                "onnx::Transpose_684": (
                    output_dequant_data[0][: 255 * 19 * 19]
                ).reshape(1, 255, 19, 19),
                "onnx::Transpose_688": (
                    output_dequant_data[1][: 255 * 38 * 38]
                ).reshape(1, 255, 38, 38),
                "onnx::Transpose_692": (
                    output_dequant_data[2][: 255 * 76 * 76]
                ).reshape(1, 255, 76, 76),
            }

            output = back_sess.run(None, back_feeds)
            delay_back_process, start = get_elapsed(start)

            output_image = postprocess(
                raw_input_img_prev, output, img_width, img_height
            )
            delay_postprocess, start = get_elapsed(start)

            elapsed, frame_start = get_elapsed(frame_start)
            fps = 1000 / elapsed
            fps_log.append(fps)
            moving_avg = sum(fps_log[-100:]) / 100
            print(
                f"VO: {delay_video_open:6.2f} | PeP: {delay_preprocess:6.2f} | FP: {delay_front_process:6.2f} | RT: {delay_front_to_wait:6.2f} | DS: {delay_wait_to_data_sending:6.2f} | OR: {delay_output_receiveing:6.2f} | DQ: {delay_dequant:6.2f} | BP:{delay_back_process:6.2f} | PoP:{delay_postprocess:6.2f} | UI: {delay_print_ui:6.2f}"
                + f"  |||  FPS: {elapsed:8.2f} ms | {(fps):7.2f} fps | 100-avg.: {moving_avg:7.2f} fps"
            )
            cv2.putText(
                raw_input_img_prev,
                f"FPS: {fps:5.2f}, MA(100): {moving_avg:5.2f} fps",
                (30, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 0),
                lineType=cv2.LINE_AA,
                thickness=4,
            )
            cv2.putText(
                raw_input_img_prev,
                f"FPS: {fps:5.2f}, MA(100): {moving_avg:5.2f} fps",
                (30, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                lineType=cv2.LINE_AA,
                thickness=2,
            )

            cpu_utilization = psutil.cpu_percent(interval=0.0)
            cv2.putText(
                raw_input_img_prev,
                f"CPU: {cpu_utilization:5.1f}%",
                (30, 80),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                lineType=cv2.LINE_AA,
                thickness=2,
            )


            if view_img:
                cv2.imshow(Path(input_video).stem, output_image)
            if save_img:
                video_writer.write(output_image)

            delay_print_ui, start = get_elapsed(start)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        assert driver.Wait() == 0
        video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        assert driver.Close() == 0
        print(e)
        print("Error occured. Closed successfully.")
        exit()

    assert driver.Close() == 0
    print("Driver ended")


if __name__ == "__main__":
    main()
