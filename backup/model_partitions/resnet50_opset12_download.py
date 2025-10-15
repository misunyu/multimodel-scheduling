import os
import torch
import torchvision.models as models

def export_resnet50_onnx(output_path="resnet50-0676ba61_opset12_fp32.onnx"):
    # 1. Load pretrained ResNet50 (IMAGENET1K_V1, hash 0676ba61)
    print("Downloading torchvision pretrained ResNet50 (0676ba61)...")
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.eval()

    # 2. Create dummy input (batch=1, 3x224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Export to ONNX (opset 12)
    print(f"Exporting model to ONNX file: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX export complete!")
    print(f"Saved ONNX absolute path: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    export_resnet50_onnx()
