"""Run YOLO / model-zoo inference on Mobilint MLA100 (ARIES) NPU.

Companion to main.py (which only handles ResNet). This script supports
YOLO object detection, instance segmentation, pose estimation, and
classification models exported by mblt-model-zoo as .mxq.
"""

import argparse
from pathlib import Path

from mblt_model_zoo import vision

REPO_RC = Path(__file__).parent / "mblt-model-zoo" / "tests" / "vision" / "rc"

TASK_DETECTION = "detection"
TASK_SEGMENTATION = "segmentation"
TASK_POSE = "pose"
TASK_CLASSIFICATION = "classification"

MODELS = {
    # object detection
    "yolov3":     (TASK_DETECTION, "YOLOv3"),
    "yolov3-tiny":(TASK_DETECTION, "YOLOv3_tiny"),
    "yolov5n":    (TASK_DETECTION, "YOLOv5n"),
    "yolov5s":    (TASK_DETECTION, "YOLOv5s"),
    "yolov5m":    (TASK_DETECTION, "YOLOv5m"),
    "yolov5l":    (TASK_DETECTION, "YOLOv5l"),
    "yolov5x":    (TASK_DETECTION, "YOLOv5x"),
    "yolov7":     (TASK_DETECTION, "YOLOv7"),
    "yolov7x":    (TASK_DETECTION, "YOLOv7x"),
    "yolov8n":    (TASK_DETECTION, "YOLOv8n"),
    "yolov8s":    (TASK_DETECTION, "YOLOv8s"),
    "yolov8m":    (TASK_DETECTION, "YOLOv8m"),
    "yolov8l":    (TASK_DETECTION, "YOLOv8l"),
    "yolov8x":    (TASK_DETECTION, "YOLOv8x"),
    "yolov9t":    (TASK_DETECTION, "YOLOv9t"),
    "yolov9s":    (TASK_DETECTION, "YOLOv9s"),
    "yolov9m":    (TASK_DETECTION, "YOLOv9m"),
    "yolov9c":    (TASK_DETECTION, "YOLOv9c"),
    "yolov9e":    (TASK_DETECTION, "YOLOv9e"),
    "yolov10n":   (TASK_DETECTION, "YOLOv10n"),
    "yolov10s":   (TASK_DETECTION, "YOLOv10s"),
    "yolov10m":   (TASK_DETECTION, "YOLOv10m"),
    "yolov10l":   (TASK_DETECTION, "YOLOv10l"),
    "yolov10x":   (TASK_DETECTION, "YOLOv10x"),
    "yolo11n":    (TASK_DETECTION, "YOLO11n"),
    "yolo11s":    (TASK_DETECTION, "YOLO11s"),
    "yolo11m":    (TASK_DETECTION, "YOLO11m"),
    "yolo11l":    (TASK_DETECTION, "YOLO11l"),
    "yolo11x":    (TASK_DETECTION, "YOLO11x"),
    "yolo12n":    (TASK_DETECTION, "YOLO12n"),
    "yolo12s":    (TASK_DETECTION, "YOLO12s"),
    "yolo12m":    (TASK_DETECTION, "YOLO12m"),
    "yolo12l":    (TASK_DETECTION, "YOLO12l"),
    "yolo12x":    (TASK_DETECTION, "YOLO12x"),
    "yolo26n":    (TASK_DETECTION, "YOLO26n"),
    "yolo26s":    (TASK_DETECTION, "YOLO26s"),
    "yolo26m":    (TASK_DETECTION, "YOLO26m"),
    "yolo26l":    (TASK_DETECTION, "YOLO26l"),
    "yolo26x":    (TASK_DETECTION, "YOLO26x"),

    # instance segmentation
    "yolov5n-seg":(TASK_SEGMENTATION, "YOLOv5nSeg"),
    "yolov5s-seg":(TASK_SEGMENTATION, "YOLOv5sSeg"),
    "yolov5m-seg":(TASK_SEGMENTATION, "YOLOv5mSeg"),
    "yolov5l-seg":(TASK_SEGMENTATION, "YOLOv5lSeg"),
    "yolov5x-seg":(TASK_SEGMENTATION, "YOLOv5xSeg"),
    "yolov8n-seg":(TASK_SEGMENTATION, "YOLOv8nSeg"),
    "yolov8s-seg":(TASK_SEGMENTATION, "YOLOv8sSeg"),
    "yolov8m-seg":(TASK_SEGMENTATION, "YOLOv8mSeg"),
    "yolov8l-seg":(TASK_SEGMENTATION, "YOLOv8lSeg"),
    "yolov8x-seg":(TASK_SEGMENTATION, "YOLOv8xSeg"),
    "yolov9c-seg":(TASK_SEGMENTATION, "YOLOv9cSeg"),
    "yolov9e-seg":(TASK_SEGMENTATION, "YOLOv9eSeg"),
    "yolo11n-seg":(TASK_SEGMENTATION, "YOLO11nSeg"),
    "yolo11s-seg":(TASK_SEGMENTATION, "YOLO11sSeg"),
    "yolo11m-seg":(TASK_SEGMENTATION, "YOLO11mSeg"),
    "yolo11l-seg":(TASK_SEGMENTATION, "YOLO11lSeg"),
    "yolo11x-seg":(TASK_SEGMENTATION, "YOLO11xSeg"),
    "yolo12n-seg":(TASK_SEGMENTATION, "YOLO12nSeg"),
    "yolo12s-seg":(TASK_SEGMENTATION, "YOLO12sSeg"),
    "yolo12m-seg":(TASK_SEGMENTATION, "YOLO12mSeg"),
    "yolo12l-seg":(TASK_SEGMENTATION, "YOLO12lSeg"),
    "yolo12x-seg":(TASK_SEGMENTATION, "YOLO12xSeg"),
    "yolo26n-seg":(TASK_SEGMENTATION, "YOLO26nSeg"),
    "yolo26s-seg":(TASK_SEGMENTATION, "YOLO26sSeg"),
    "yolo26m-seg":(TASK_SEGMENTATION, "YOLO26mSeg"),
    "yolo26l-seg":(TASK_SEGMENTATION, "YOLO26lSeg"),
    "yolo26x-seg":(TASK_SEGMENTATION, "YOLO26xSeg"),

    # pose estimation
    "yolov8n-pose":(TASK_POSE, "YOLOv8nPose"),
    "yolov8s-pose":(TASK_POSE, "YOLOv8sPose"),
    "yolov8m-pose":(TASK_POSE, "YOLOv8mPose"),
    "yolov8l-pose":(TASK_POSE, "YOLOv8lPose"),
    "yolov8x-pose":(TASK_POSE, "YOLOv8xPose"),
    "yolo11n-pose":(TASK_POSE, "YOLO11nPose"),
    "yolo11s-pose":(TASK_POSE, "YOLO11sPose"),
    "yolo11m-pose":(TASK_POSE, "YOLO11mPose"),
    "yolo11l-pose":(TASK_POSE, "YOLO11lPose"),
    "yolo11x-pose":(TASK_POSE, "YOLO11xPose"),
    "yolo26n-pose":(TASK_POSE, "YOLO26nPose"),
    "yolo26s-pose":(TASK_POSE, "YOLO26sPose"),
    "yolo26m-pose":(TASK_POSE, "YOLO26mPose"),
    "yolo26l-pose":(TASK_POSE, "YOLO26lPose"),
    "yolo26x-pose":(TASK_POSE, "YOLO26xPose"),

    # classification (YOLO cls heads)
    "yolov5n-cls":(TASK_CLASSIFICATION, "YOLOv5nCls"),
    "yolov5s-cls":(TASK_CLASSIFICATION, "YOLOv5sCls"),
    "yolov5m-cls":(TASK_CLASSIFICATION, "YOLOv5mCls"),
    "yolov8n-cls":(TASK_CLASSIFICATION, "YOLOv8nCls"),
    "yolov8s-cls":(TASK_CLASSIFICATION, "YOLOv8sCls"),
    "yolov8m-cls":(TASK_CLASSIFICATION, "YOLOv8mCls"),
    "yolo11n-cls":(TASK_CLASSIFICATION, "YOLO11nCls"),
    "yolo11s-cls":(TASK_CLASSIFICATION, "YOLO11sCls"),
    "yolo11m-cls":(TASK_CLASSIFICATION, "YOLO11mCls"),
    "yolo26n-cls":(TASK_CLASSIFICATION, "YOLO26nCls"),
    "yolo26s-cls":(TASK_CLASSIFICATION, "YOLO26sCls"),
    "yolo26m-cls":(TASK_CLASSIFICATION, "YOLO26mCls"),
}

DEFAULT_IMAGE_BY_TASK = {
    TASK_DETECTION:      REPO_RC / "cr7.jpg",
    TASK_SEGMENTATION:   REPO_RC / "cr7.jpg",
    TASK_POSE:           REPO_RC / "cr7.jpg",
    TASK_CLASSIFICATION: REPO_RC / "volcano.jpg",
}


def resolve_model_class(class_name: str):
    cls = getattr(vision, class_name, None)
    if cls is None:
        raise ValueError(
            f"Model class '{class_name}' not found in mblt_model_zoo.vision. "
            "Your installed model-zoo build may not include it."
        )
    return cls


def main():
    parser = argparse.ArgumentParser(
        description="Run a YOLO / model-zoo model on Mobilint MLA100 (ARIES)",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS.keys()),
        metavar="MODEL",
        help="Model key (see --list for full set). Examples: yolo11m, yolov8s-seg, yolo11x-pose, yolo11m-cls",
    )
    parser.add_argument("--image", type=str, default=None, help="Input image path (defaults vary by task).")
    parser.add_argument(
        "--infer-mode",
        choices=["single", "multi", "global4", "global8"],
        default="global8",
    )
    parser.add_argument("--product", type=str, default="aries")
    parser.add_argument("--mxq-path", type=str, default=None, help="Optional local .mxq path.")
    parser.add_argument("--model-type", type=str, default="DEFAULT")
    parser.add_argument("--save-path", type=str, default=None, help="Where to save the annotated image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Detection/seg/pose confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="Detection/seg/pose IoU (NMS) threshold.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to print for classification.")
    args = parser.parse_args()

    task, class_name = MODELS[args.model]
    image_path = args.image or str(DEFAULT_IMAGE_BY_TASK[task])

    print(f"[init] loading {args.model} ({class_name}, task={task}) on product={args.product}, "
          f"infer_mode={args.infer_mode}")

    ModelCls = resolve_model_class(class_name)
    model = ModelCls(
        local_path=args.mxq_path,
        model_type=args.model_type,
        infer_mode=args.infer_mode,
        product=args.product,
    )

    try:
        print(f"[infer] image: {image_path}")
        input_img = model.preprocess(image_path)
        output = model(input_img)

        if task == TASK_CLASSIFICATION:
            result = model.postprocess(output)
            print(f"[result] top-{args.topk} predictions:")
            result.plot(source_path=image_path, save_path=args.save_path, topk=args.topk)
        else:
            result = model.postprocess(
                output,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
            )
            print(f"[result] {task} (conf>={args.conf_thres}, iou>={args.iou_thres})")
            result.plot(source_path=image_path, save_path=args.save_path)

        if args.save_path:
            print(f"[saved] {args.save_path}")
    finally:
        model.dispose()


if __name__ == "__main__":
    main()
