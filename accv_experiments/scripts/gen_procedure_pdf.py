# -*- coding: utf-8 -*-
"""Generate a PDF describing the fine-tuning + Mobilint MLA100 (ARIES) INT8 compile
procedure with the exact commands used, in order. Factual: values from the actual run."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, HRFlowable)
OUT="accv_experiments/results/mla100_training_compile_procedure.pdf"
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=15,spaceBefore=10,spaceAfter=6,textColor=colors.HexColor("#1a3c6e"))
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=11.5,spaceBefore=9,spaceAfter=4,textColor=colors.HexColor("#244b7a"))
BODY=ParagraphStyle("BODY",parent=ss["BodyText"],fontSize=9.3,leading=13,spaceAfter=4)
NOTE=ParagraphStyle("NOTE",parent=BODY,fontSize=8.7,textColor=colors.HexColor("#444444"),leftIndent=6)
CODE=ParagraphStyle("CODE",parent=ss["Code"],fontSize=7.8,leading=10,backColor=colors.HexColor("#f3f4f6"),
                    borderColor=colors.HexColor("#d0d4da"),borderWidth=0.5,borderPadding=5,leftIndent=2,spaceBefore=2,spaceAfter=6)
TITLE=ParagraphStyle("TITLE",parent=ss["Title"],fontSize=18,textColor=colors.HexColor("#13294b"),spaceAfter=2)
SUB=ParagraphStyle("SUB",parent=BODY,fontSize=9.5,textColor=colors.HexColor("#555555"),spaceAfter=2)

def code(s): return Preformatted(s.strip("\n"), CODE)
E=[]
def P(t,st=BODY): E.append(Paragraph(t,st))
def SP(h=4): E.append(Spacer(1,h))

P("Fine-Tuning and Mobilint MLA100 (ARIES) INT8 Compilation — Procedure & Commands", TITLE)
P("YOLOv11s on Argoverse-HD &nbsp;|&nbsp; RTX 5090 (FP32 training) + Mobilint MLA100 / ARIES (INT8 compile)", SUB)
P("Self-contained record of the EXP-FT-LOCAL pipeline. All commands are the ones actually executed; "
  "model/artifact hashes are SHA-256 prefixes of the produced files.", SUB)
E.append(HRFlowable(width="100%",thickness=0.8,color=colors.HexColor("#244b7a"),spaceBefore=4,spaceAfter=8))

P("0. Scope and key design decision", H1)
P("Goal: take the COCO-pretrained YOLOv11s, <b>fine-tune</b> it on Argoverse-HD, and produce an "
  "<b>INT8 model for the Mobilint MLA100 (ARIES) NPU</b>, then compare the INT8 quantization "
  "structure of the COCO model vs. the fine-tuned model.", )
P("<b>Why a local paired compile (not the vendor blob).</b> The paper's main NPU model is a "
  "vendor-published <i>.mxq</i> blob (HuggingFace <font face='Courier'>mobilint/YOLO11s</font>). A "
  "verification step showed the local compiler (qbcompiler 1.1.2) does <b>not</b> reproduce that "
  "vendor blob to within tolerance (per-size sAP &Delta; up to 0.008 &gt; 0.003). Therefore this "
  "procedure compiles <b>both</b> the COCO and the fine-tuned model with the <b>same local recipe</b> "
  "so the comparison is internally controlled (recipe held fixed; only the weights differ). These "
  "INT8 numbers are an internal COCO-vs-FT comparison, not directly comparable to the vendor-blob "
  "Table&nbsp;1.", NOTE)

P("1. Environments (two virtualenvs)", H1)
P("Training/inference and compilation use <b>separate</b> Python environments because the Mobilint "
  "compiler's native module requires a different PyTorch C++ ABI.", )
P("<b>(a) Training / inference venv</b> (<font face='Courier'>.venv</font>): torch 2.12.0+cu130, "
  "onnxruntime-gpu 1.20.1, ultralytics 8.4.56.", )
P("<b>(b) Compiler venv</b> (<font face='Courier'>.venv_qbc</font>): qbcompiler 1.1.2 (+aries2). Its "
  "native MMC module is built with the <b>C++11 ABI</b> and needs a cxx11-ABI PyTorch with CUDA "
  "(Blackwell sm_120) support &mdash; torch 2.12 (ABI mismatch) and torch 2.4 (no cxx11) both fail; "
  "<b>torch 2.8.0+cu128 (cxx11=True)</b> is the working combination.", NOTE)
P("Build the compiler venv (final working sequence):", H2)
code(r"""
python3.10 -m venv .venv_qbc
. .venv_qbc/bin/activate
# cxx11-ABI CUDA torch (Blackwell sm_120) required by qbcompiler's native MMC
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install onnxruntime-gpu==1.20.1
# Mobilint compiler wheel (no-deps so torch is not pulled/upgraded)
pip install --no-deps /home/msyu/PycharmProjects/MobilintTest/qbcompiler-1.1.2+aries2-py3-none-any.whl
# remaining compiler deps
pip install pydantic pydantic-settings msgpack onnx opencv-python-headless \
            pycocotools pyyaml typeguard tqdm transformers "tensorflow-cpu>=2.9"
# sanity: native compiler module must import (ABI gate)
python -c "import qbcompiler.mmc; from qbcompiler import Compiler, mxq_compile; print('MMC OK')"
""")
P("Verified: <font face='Courier'>torch 2.8.0+cu128 (cxx11=True)</font>, "
  "<font face='Courier'>onnxruntime 1.20.1</font>, <font face='Courier'>qbcompiler 1.1.2</font>; "
  "<font face='Courier'>qbcompiler.mmc</font> imports and CUDA runs on the RTX 5090.", NOTE)

P("2. Step 1 &mdash; Dataset preparation (Argoverse-HD &rarr; YOLO labels)", H1)
P("Argoverse-HD train annotations (COCO-style, 8 categories) are converted to Ultralytics YOLO "
  "labels. Labels are written in the <b>COCO-80 class space</b> (AHD id &rarr; COCO id mapping "
  "<font face='Courier'>[0,1,2,3,5,7,9,11]</font>) so the fine-tuned model keeps the 80-class head "
  "and the existing evaluation harness (COCO&rarr;AHD mapping) is reused unchanged. Images are "
  "symlinked; a dataset YAML is written.", )
code(r"""
# (training venv)
. .venv/bin/activate
python accv_experiments/scripts/ft_prep_dataset.py
#  -> builds accv_experiments/data/ahd_yolo/{images,labels}/{train,val}
#     train: 39,384 images / 771,774 boxes   val: 15,062 images / 253,941 boxes
#     writes ahd_coco80.yaml  (nc: 80, names = COCO-80, train/val image dirs)
""")
P("Resulting dataset YAML (key fields):", H2)
code(r"""
path: .../accv_experiments/data/ahd_yolo
train: images/train
val:   images/val
nc: 80
names: [person, bicycle, car, motorcycle, airplane, bus, train, truck, ... ]  # standard COCO-80
""")

P("3. Step 2 &mdash; Fine-tuning (RTX 5090, FP32 PyTorch)", H1)
P("Ultralytics 8.4.56 defaults, fixed knobs only: 50 epochs, imgsz 640 (= inference resolution), "
  "seed 0, deterministic. Initial weights = the public COCO-pretrained "
  "<font face='Courier'>yolo11s.pt</font>. Run detached (survives disconnection).", )
code(r"""
# (training venv) — exact training command
yolo detect train \
     model=yolo11s.pt \
     data=accv_experiments/data/ahd_yolo/ahd_coco80.yaml \
     epochs=50 imgsz=640 seed=0 deterministic=True \
     project=accv_experiments/results/ft_runs name=ft_yolo11s exist_ok=True

# how it was launched (detached, session-independent):
setsid bash accv_experiments/scripts/ft_train_driver.sh < /dev/null \
       > accv_experiments/results/ft_runs/driver.out 2>&1 &
""")
P("Optimizer was Ultralytics auto (resolved to MuSGD, lr0=0.01). Outcome: 50 epochs completed; the "
  "<b>best checkpoint = epoch 1</b> (Ultralytics' default best-by-val-mAP; mAP50-95 peaked 0.236 at "
  "epoch 1 and declined afterwards). Best weights kept as-is (no cherry-picking): "
  "<font face='Courier'>ft_runs/ft_yolo11s/weights/best.pt</font>.", NOTE)

P("4. Step 3 &mdash; Export to ONNX (input to the compiler)", H1)
P("The compiler consumes ONNX. Both the COCO-pretrained and the fine-tuned weights are exported at "
  "640&times;640, opset 13, static shapes, FP32.", )
code(r"""
# (training venv)
python - <<'PY'
from ultralytics import YOLO
# COCO-pretrained
YOLO("yolo11s.pt").export(format="onnx", imgsz=640, opset=13,
                          simplify=False, dynamic=False, half=False, device="cpu")
#   -> yolo11s.onnx
# fine-tuned
YOLO("accv_experiments/results/ft_runs/ft_yolo11s/weights/best.pt").export(
     format="onnx", imgsz=640, opset=13, simplify=False, dynamic=False, half=False, device="cpu")
#   -> accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx
PY
""")

P("5. Step 4 &mdash; Calibration set for INT8 quantization", H1)
P("INT8 quantization needs representative calibration inputs. A single fixed set of <b>200 "
  "Argoverse-HD train frames</b> (uniform sampling) is used for <b>both</b> compiles, so the "
  "calibration is identical across COCO and FT (only the weights differ).", )
code(r"""
# 200 uniformly-sampled train frames -> accv_experiments/results/qbc_calib200_train/
#   (raw JPGs; the yolo_640 preset letterboxes/normalizes them at compile time)
# file list saved at: accv_experiments/results/qbc_calib200_train.list
""")

P("6. Step 5 &mdash; Compile to MLA100 / ARIES INT8 (.mxq) with qbcompiler", H1)
P("Compilation uses the high-level <font face='Courier'>mxq_compile()</font> API of qbcompiler "
  "(product = ARIES / MLA100). Recipe held fixed across models:", )
P("&bull; <b>config_preset = \"yolo_640\"</b>: YOLO 640&times;640 letterbox preprocessing (pad 114), "
  "uint8 input (the NPU does normalization), 3 image channels.<br/>"
  "&bull; <b>inference_scheme = \"global8\"</b>: all-8-core mode, used for the single-stream "
  "(Table-1-style) evaluation. A <b>\"single\"</b>-mode build (1 core/instance, up to 8 concurrent) "
  "is also produced for the N=4 multi-stream test.<br/>"
  "&bull; <b>backend = \"onnx\"</b>, <b>device = \"gpu\"</b> (calibration runs on the host GPU).", )
code(r"""
# (compiler venv)
. .venv_qbc/bin/activate
python - <<'PY'
from qbcompiler import mxq_compile
CALIB = "accv_experiments/results/qbc_calib200_train"

# (1) COCO-pretrained -> INT8, global8
mxq_compile(model="yolo11s.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_coco_traincalib_global8.mxq",
            config_preset="yolo_640", inference_scheme="global8",
            device="gpu", backend="onnx")

# (2) fine-tuned -> INT8, global8   (same recipe/calibration, different weights)
mxq_compile(model="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_ft_traincalib_global8.mxq",
            config_preset="yolo_640", inference_scheme="global8",
            device="gpu", backend="onnx")

# (3) fine-tuned -> INT8, single mode (for N=4 concurrent NPU streams)
mxq_compile(model="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_ft_traincalib_single.mxq",
            config_preset="yolo_640", inference_scheme="single",
            device="gpu", backend="onnx")
PY
""")
P("Each compile took ~30&ndash;33&nbsp;s and produced a ~10.7&nbsp;MB <font face='Courier'>.mxq</font> "
  "(parse &rarr; quantize/calibrate &rarr; ARIES GlobalCluster build &rarr; export &amp; verify).", NOTE)

P("7. Step 6 &mdash; Post-compile verification", H1)
P("&bull; <b>Determinism / bit-identity:</b> each .mxq run on 20 frames twice &rarr; box/score diff = 0.<br/>"
  "&bull; <b>Operating point:</b> single-stream evaluation at threads=4, frame-skip &asymp; 0.<br/>"
  "&bull; <b>FP32 gate before compiling:</b> fine-tuned vs COCO offline mAP (small +0.045, medium "
  "+0.069) confirmed fine-tuning raised accuracy before INT8 conversion.", )

P("8. Artifacts (SHA-256 prefix)", H1)
tdata=[["File","role","sha256[:16]"],
 ["yolo11s.pt","COCO-pretrained init (Ultralytics v8.2.100)","85a76fe86dd8afe3"],
 ["ft_yolo11s/weights/best.pt","fine-tuned FP32 (epoch 1 best)","4c4b342a6d09de67"],
 ["yolo11s.onnx","COCO ONNX (640, opset13)","5ffc807bb21ba5f5"],
 ["best.onnx","fine-tuned ONNX (640, opset13)","8e189e0252604e8f"],
 ["qbc_coco_traincalib_global8.mxq","COCO INT8 (global8)","f987aca54e64c363"],
 ["qbc_ft_traincalib_global8.mxq","fine-tuned INT8 (global8)","ea091a9dc2485a79"],
 ["qbc_ft_traincalib_single.mxq","fine-tuned INT8 (single)","c0520594743655e4"]]
t=Table(tdata,colWidths=[150,205,95])
t.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("FONTSIZE",(0,0),(-1,-1),7.4),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
 ("FONTNAME",(0,1),(0,-1),"Courier"),("FONTNAME",(2,1),(2,-1),"Courier"),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f4f6")]),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(t)
SP(8)
P("Scripts: ft_prep_dataset.py, ft_train_driver.sh, ft_dual_compile.py, ft_stretch_vlm.py "
  "(all under accv_experiments/scripts/). Detector COCO-pretrained weights: Ultralytics "
  "GitHub assets release. NPU compiler: qbcompiler 1.1.2 (+aries2), Mobilint ARIES / MLA100.", NOTE)

SimpleDocTemplate(OUT,pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,topMargin=15*mm,bottomMargin=14*mm,
                  title="MLA100 Training & Compile Procedure").build(E)
print("written",OUT)
