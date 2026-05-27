"""ResNet inference example on Mobilint MLA100 (ARIES) NPU.

Uses the mblt-model-zoo package to load a pre-quantized ResNet model
(.mxq), runs it on the NPU, and prints top-5 ImageNet predictions.
"""

import argparse
from pathlib import Path

from mblt_model_zoo.vision import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

MODELS = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
}

DEFAULT_IMAGE = (
    Path(__file__).parent
    / "mblt-model-zoo"
    / "tests"
    / "vision"
    / "rc"
    / "volcano.jpg"
)


def main():
    parser = argparse.ArgumentParser(description="Run ResNet on Mobilint MLA100 (ARIES)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="resnet50")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument(
        "--infer-mode",
        choices=["single", "multi", "global4", "global8"],
        default="global8",
        help="ARIES inference execution mode",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the annotated result image",
    )
    args = parser.parse_args()

    print(f"[init] loading {args.model} for product=aries, infer_mode={args.infer_mode}")
    model = MODELS[args.model](
        infer_mode=args.infer_mode,
        product="aries",
    )

    try:
        print(f"[infer] image: {args.image}")
        input_img = model.preprocess(args.image)
        output = model(input_img)
        result = model.postprocess(output)

        print("[result] top-5 predictions:")
        result.plot(source_path=args.image, save_path=args.save_path, topk=5)
        if args.save_path:
            print(f"[saved] {args.save_path}")
    finally:
        model.dispose()


if __name__ == "__main__":
    main()
