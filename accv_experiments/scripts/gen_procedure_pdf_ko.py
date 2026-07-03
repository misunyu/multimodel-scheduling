# -*- coding: utf-8 -*-
"""한국어판: fine-tuning + Mobilint MLA100(ARIES) INT8 컴파일 절차/커맨드 PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                                Table, TableStyle, HRFlowable)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
# reportlab 내장 Korean CID 폰트 (Adobe-Korea1, 폰트파일 불필요)
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
pdfmetrics.registerFontFamily("HYGothic-Medium",normal="HYGothic-Medium",bold="HYGothic-Medium",
                              italic="HYGothic-Medium",boldItalic="HYGothic-Medium")
OUT="accv_experiments/results/mla100_training_compile_procedure_ko.pdf"
ss=getSampleStyleSheet()
def st(n,**k): return ParagraphStyle(n,fontName="HYGothic-Medium",**k)
TITLE=st("T",fontSize=17,leading=21,textColor=colors.HexColor("#13294b"),spaceAfter=2)
SUB=st("S",fontSize=9.3,leading=13,textColor=colors.HexColor("#555555"),spaceAfter=2)
H1=st("H1",fontSize=14,leading=18,textColor=colors.HexColor("#1a3c6e"),spaceBefore=10,spaceAfter=5)
H2=st("H2",fontSize=11,leading=15,textColor=colors.HexColor("#244b7a"),spaceBefore=8,spaceAfter=3)
BODY=st("B",fontSize=9.3,leading=14,spaceAfter=4)
NOTE=st("N",fontSize=8.7,leading=12.5,textColor=colors.HexColor("#444444"),leftIndent=6,spaceAfter=4)
CODE=ParagraphStyle("C",parent=ss["Code"],fontSize=7.8,leading=10,backColor=colors.HexColor("#f3f4f6"),
        borderColor=colors.HexColor("#d0d4da"),borderWidth=0.5,borderPadding=5,leftIndent=2,spaceBefore=2,spaceAfter=6)
def code(s): return Preformatted(s.strip("\n"),CODE)
def diagram():
    LG=colors.HexColor("#eef2f8"); MD=colors.HexColor("#244b7a"); GY=colors.HexColor("#555555")
    d=Drawing(505,212)
    d.add(String(250,202,"horizontal -> = fine-tuning effect    |    vertical (down) = quantization loss",
                 fontName="Helvetica-Oblique",fontSize=7.3,fillColor=GY,textAnchor="middle"))
    def box(x,y,t1,t2):
        d.add(Rect(x,y,180,36,rx=4,ry=4,fillColor=LG,strokeColor=MD,strokeWidth=0.8))
        d.add(String(x+90,y+21,t1,fontName="Helvetica-Bold",fontSize=8.6,textAnchor="middle"))
        d.add(String(x+90,y+8,t2,fontName="Helvetica",fontSize=6.8,fillColor=GY,textAnchor="middle"))
    box(30,150,"yolo11s.pt","COCO-pretrained,  FP32")
    box(290,150,"best.pt","fine-tuned (Argoverse-HD),  FP32")
    box(30,30,"coco INT8 (.mxq)","qbc_coco_traincalib_global8")
    box(290,30,"ft INT8 (.mxq)","qbc_ft_traincalib_global8")
    def ar_right(x1,x2,y,lab):
        d.add(Line(x1,y,x2-7,y,strokeColor=MD,strokeWidth=1.3))
        d.add(Polygon([x2,y,x2-8,y+3.5,x2-8,y-3.5],fillColor=MD,strokeColor=MD))
        d.add(String((x1+x2)/2,y+5,lab,fontName="Helvetica-Bold",fontSize=7.2,fillColor=MD,textAnchor="middle"))
    def ar_down(x,y1,y2,lab):
        d.add(Line(x,y1,x,y2+7,strokeColor=MD,strokeWidth=1.3))
        d.add(Polygon([x,y2,x-3.5,y2+8,x+3.5,y2+8],fillColor=MD,strokeColor=MD))
        d.add(String(x+6,(y1+y2)/2,lab,fontName="Helvetica-Bold",fontSize=7.2,fillColor=MD,textAnchor="start"))
    ar_right(210,290,168,"(1) fine-tuning (GPU)")
    ar_down(120,150,66,"(2) INT8 compile")
    ar_down(380,150,66,"(2) INT8 compile")
    return d
E=[]
def P(t,s=BODY): E.append(Paragraph(t,s))
def SP(h=4): E.append(Spacer(1,h))
C='<font face="Courier">'; _C='</font>'  # 인라인 코드

P("파인튜닝 및 Mobilint MLA100(ARIES) INT8 컴파일 — 절차 및 커맨드",TITLE)
P("YOLOv11s · Argoverse-HD &nbsp;|&nbsp; RTX 5090(FP32 학습) + Mobilint MLA100 / ARIES(INT8 컴파일)",SUB)
P("EXP-FT-LOCAL 파이프라인의 자기완결형 기록. 모든 커맨드는 실제 실행한 것이며, 해시는 산출 파일의 "
  "SHA-256 앞자리입니다.",SUB)
E.append(HRFlowable(width="100%",thickness=0.8,color=colors.HexColor("#244b7a"),spaceBefore=4,spaceAfter=8))

P("0. 목적과 핵심 설계 결정",H1)
P("목표: COCO 사전학습 YOLOv11s를 Argoverse-HD로 <b>파인튜닝</b>하고, "
  "<b>Mobilint MLA100(ARIES) NPU용 INT8 모델</b>로 컴파일한 뒤, COCO 모델과 파인튜닝 모델의 "
  "INT8 양자화 구조를 비교한다.")
P("0.1 이 컴파일의 목적 — paired comparison (fine-tuning 효과 분리)",H2)
P("이 절차의 INT8 컴파일은 <b>fine-tuning 효과를 비교하기 위한 paired compilation</b>이다. "
  "<b>동일 dataset과 동일 recipe를 고정</b>하고 <b>모델 가중치만</b>(COCO 사전학습 ↔ fine-tuned) "
  "바꿔 한 쌍으로 컴파일한다. 다른 조건을 모두 묶어두므로, 두 INT8 모델의 차이는 곧 fine-tuning "
  "효과로 해석할 수 있다(통계의 paired comparison과 같은 논리).")
P("&bull; <b>고정(동일)</b>: calibration data(train 200장, 같은 리스트), 평가 data(val 24 logs, "
  "같은 protocol), compile recipe(qbcompiler, yolo_640 preset, global8, ONNX 640/opset13).<br/>"
  "&bull; <b>변화(비교 축)</b>: 모델 가중치만 — COCO-pretrained vs fine-tuned.",NOTE)
P("여기서 dataset은 비교 대상이 아니라 <b>통제 변수(전제)</b>다. fine-tuning 학습 자체는 "
  "Argoverse-HD train으로 수행하고(COCO 모델은 학습하지 않아 baseline 역할), 두 모델의 calibration·"
  "평가는 같은 data로 맞춘다. 정확히는 <b>“동일 calibration·평가 data 위에서, Argoverse-HD train으로 "
  "fine-tuning한 효과가 INT8 quantization 구조에 어떻게 나타나는지 비교하는” 컴파일</b>이다.",NOTE)
P("확인된 결과: fine-tuning으로 small-object 정확도를 올린 뒤에도 INT8의 small-biased quantization "
  "구조가 유지됨(small 상대 손실 ≫ large) — 그 구조가 low-accuracy regime의 artifact가 아님을 "
  "보이는 것이 이 paired compilation의 목적이다.",NOTE)
P("0.2 비교 구조 — fine-tuning(학습) vs INT8 컴파일(양자화)은 별개 단계",H2)
P("두 단계는 전혀 다르다. <b>① fine-tuning</b>은 GPU에서 모델 가중치(weight) 값을 학습으로 바꾸는 "
  "과정이며 타입은 계속 <b>FP32</b>다(NPU와 무관). <b>② INT8 컴파일</b>은 학습이 끝난 모델을 NPU(ARIES)가 "
  "실행할 <b>INT8</b> 형식으로 <b>quantization</b>·패키징하는 과정으로, 가중치를 학습하는 게 아니라 형식을 "
  "바꾸는 것이다. 즉 fine-tuning은 NPU 컴파일을 의미하지 않는다.")
E.append(Spacer(1,4)); E.append(diagram()); E.append(Spacer(1,6))
P("그러므로 “다운로드한 float 모델”과 “컴파일한 int8 모델”은 <b>가중치를 비교하는 두 모델이 아니라 한 "
  "모델의 두 형식</b>(FP32 / INT8)이다. 측정은 아래 2×2 구조로 이뤄진다.",BODY)
tdm=[["","FP32 (GPU)","INT8 (NPU, 컴파일 후)"],
     ["COCO (원본, 학습 안 함)","yolo11s.pt","qbc_coco_traincalib_global8.mxq"],
     ["fine-tuned (Argoverse-HD)","best.pt","qbc_ft_traincalib_global8.mxq"]]
tm=Table(tdm,colWidths=[150,95,205])
tm.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("BACKGROUND",(0,1),(0,-1),colors.HexColor("#eef2f8")),
 ("FONTNAME",(0,0),(-1,-1),"HYGothic-Medium"),("FONTSIZE",(0,0),(-1,-1),7.2),
 ("FONTNAME",(1,1),(2,-1),"Courier"),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(Spacer(1,2)); E.append(tm); SP(4)
P("&bull; <b>세로(↓, FP32→INT8)</b> = 그 모델의 <b>quantization 손실</b> (가중치 동일, 형식만 다름).<br/>"
  "&bull; <b>가로(→, COCO→fine-tuned)</b> = <b>fine-tuning 효과</b> (가중치가 다름).<br/>"
  "paired comparison은 두 행의 quantization 손실 구조(세로 차이)를 나란히 비교한다.",NOTE)
P("<b>왜 vendor blob이 아니라 로컬 쌍대 컴파일인가.</b> 논문 본 실험의 NPU 모델은 vendor가 배포한 "
  f"{C}.mxq{_C} blob(HuggingFace {C}mobilint/YOLO11s{_C})이다. 검증 결과 로컬 컴파일러(qbcompiler "
  "1.1.2)는 이 vendor blob을 허용오차 내로 재현하지 못했다(per-size sAP 차이 최대 0.008 &gt; 0.003). "
  "따라서 본 절차는 COCO와 파인튜닝 모델을 <b>동일한 로컬 레시피</b>로 모두 컴파일해, 레시피를 고정하고 "
  "가중치만 다르게 한 내부 통제 비교를 수행한다. 이 INT8 수치는 내부 COCO-vs-FT 비교용이며 vendor blob "
  "기반 Table 1과 직접 비교하지 않는다.",NOTE)

P("1. 실행 환경 (가상환경 2개)",H1)
P("학습/추론과 컴파일은 <b>분리된</b> 파이썬 환경을 쓴다. Mobilint 컴파일러의 네이티브 모듈이 다른 "
  "PyTorch C++ ABI를 요구하기 때문이다.")
P(f"<b>(a) 학습/추론 venv</b>({C}.venv{_C}): torch 2.12.0+cu130, onnxruntime-gpu 1.20.1, "
  "ultralytics 8.4.56.")
P(f"<b>(b) 컴파일러 venv</b>({C}.venv_qbc{_C}): qbcompiler 1.1.2(+aries2). 네이티브 MMC 모듈이 "
  "<b>C++11 ABI</b>로 빌드돼 있어, cxx11-ABI이면서 CUDA(Blackwell sm_120)를 지원하는 PyTorch가 필요하다. "
  "torch 2.12(ABI 불일치)·torch 2.4(cxx11 아님) 모두 실패하고, <b>torch 2.8.0+cu128(cxx11=True)</b>이 "
  "작동 조합이다.",NOTE)
P("컴파일러 venv 구축 (최종 작동 순서):",H2)
code(r"""
python3.10 -m venv .venv_qbc
. .venv_qbc/bin/activate
# qbcompiler 네이티브 MMC가 요구하는 cxx11-ABI CUDA torch (Blackwell sm_120)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install onnxruntime-gpu==1.20.1
# Mobilint 컴파일러 휠 (--no-deps: torch 미변경)
pip install --no-deps /home/msyu/PycharmProjects/MobilintTest/qbcompiler-1.1.2+aries2-py3-none-any.whl
# 나머지 의존성
pip install pydantic pydantic-settings msgpack onnx opencv-python-headless \
            pycocotools pyyaml typeguard tqdm transformers "tensorflow-cpu>=2.9"
# sanity: 네이티브 컴파일 모듈 import (ABI 게이트)
python -c "import qbcompiler.mmc; from qbcompiler import Compiler, mxq_compile; print('MMC OK')"
""")
P(f"확인: {C}torch 2.8.0+cu128 (cxx11=True){_C}, {C}onnxruntime 1.20.1{_C}, {C}qbcompiler 1.1.2{_C}; "
  f"{C}qbcompiler.mmc{_C} import 성공, RTX 5090에서 CUDA 정상.",NOTE)

P("2. 단계 1 — 데이터셋 준비 (Argoverse-HD → YOLO 라벨)",H1)
P("Argoverse-HD train 어노테이션(COCO 형식, 8개 카테고리)을 Ultralytics YOLO 라벨로 변환한다. 라벨은 "
  f"<b>COCO-80 클래스 공간</b>으로 기록한다(AHD id → COCO id 매핑 {C}[0,1,2,3,5,7,9,11]{_C}). 이렇게 하면 "
  "파인튜닝 모델이 80-class head를 유지해 기존 평가 harness(COCO→AHD 매핑)를 그대로 재사용할 수 있다. "
  "이미지는 symlink, 데이터셋 YAML을 생성한다.")
code(r"""
# (학습 venv)
. .venv/bin/activate
python accv_experiments/scripts/ft_prep_dataset.py
#  -> accv_experiments/data/ahd_yolo/{images,labels}/{train,val} 생성
#     train: 39,384 images / 771,774 boxes,  val: 15,062 images / 253,941 boxes
#     ahd_coco80.yaml 작성 (nc: 80, names = COCO-80, train/val 이미지 경로)
""")
P("생성된 데이터셋 YAML(핵심 필드):",H2)
code(r"""
path: .../accv_experiments/data/ahd_yolo
train: images/train
val:   images/val
nc: 80
names: [person, bicycle, car, motorcycle, airplane, bus, train, truck, ... ]  # 표준 COCO-80
""")

P("3. 단계 2 — 파인튜닝 (RTX 5090, FP32 PyTorch)",H1)
P("Ultralytics 8.4.56 기본값, 고정 항목만 지정: 50 epoch, imgsz 640(=추론 해상도), seed 0, "
  f"deterministic. 초기 가중치 = 공개 COCO 사전학습 {C}yolo11s.pt{_C}. 세션과 무관하게 분리 실행.")
code(r"""
# (학습 venv) — 실제 학습 커맨드
yolo detect train \
     model=yolo11s.pt \
     data=accv_experiments/data/ahd_yolo/ahd_coco80.yaml \
     epochs=50 imgsz=640 seed=0 deterministic=True \
     project=accv_experiments/results/ft_runs name=ft_yolo11s exist_ok=True

# 분리 실행 방법(세션 독립):
setsid bash accv_experiments/scripts/ft_train_driver.sh < /dev/null \
       > accv_experiments/results/ft_runs/driver.out 2>&1 &
""")
P("옵티마이저는 Ultralytics auto(MuSGD, lr0=0.01로 결정). 결과: 50 epoch 완료, <b>best checkpoint = "
  "epoch 1</b>(Ultralytics 기본 기준 = val mAP 최고; mAP50-95가 epoch 1에서 0.236으로 정점 후 하락). "
  f"cherry-pick 없이 best 그대로 사용: {C}ft_runs/ft_yolo11s/weights/best.pt{_C}.",NOTE)

P("4. 단계 3 — ONNX export (컴파일러 입력)",H1)
P("컴파일러는 ONNX를 입력으로 받는다. COCO 사전학습·파인튜닝 가중치 모두 640×640, opset 13, 정적 shape, "
  "FP32로 export.")
code(r"""
# (학습 venv)
python - <<'PY'
from ultralytics import YOLO
# COCO 사전학습
YOLO("yolo11s.pt").export(format="onnx", imgsz=640, opset=13,
                          simplify=False, dynamic=False, half=False, device="cpu")
#   -> yolo11s.onnx
# 파인튜닝
YOLO("accv_experiments/results/ft_runs/ft_yolo11s/weights/best.pt").export(
     format="onnx", imgsz=640, opset=13, simplify=False, dynamic=False, half=False, device="cpu")
#   -> accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx
PY
""")

P("5. 단계 4 — INT8 양자화용 calibration 세트",H1)
P("INT8 양자화에는 대표 입력이 필요하다. <b>Argoverse-HD train 프레임 200장</b>(균등 샘플)을 고정해 "
  "<b>두 컴파일에 공통</b>으로 사용한다 — calibration을 COCO·FT 간 동일하게 두고 가중치만 다르게 한다.")
code(r"""
# train 균등 샘플 200장 -> accv_experiments/results/qbc_calib200_train/
#   (raw JPG; yolo_640 preset이 컴파일 시 letterbox/normalize 수행)
# 파일 리스트: accv_experiments/results/qbc_calib200_train.list
""")

P("6. 단계 5 — MLA100 / ARIES INT8(.mxq) 컴파일 (qbcompiler)",H1)
P(f"컴파일은 qbcompiler의 고수준 {C}mxq_compile(){_C} API 사용(product = ARIES / MLA100). 레시피는 "
  "모델 간 고정:")
P("&bull; <b>config_preset = \"yolo_640\"</b>: YOLO 640×640 letterbox 전처리(pad 114), uint8 입력"
  "(정규화는 NPU가 수행), 3채널.<br/>"
  "&bull; <b>inference_scheme = \"global8\"</b>: 8코어 전체 모드, 단일 스트림(Table 1 방식) 평가용. "
  "N=4 멀티스트림 테스트용으로 <b>\"single\"</b> 모드(인스턴스당 1코어, 최대 8동시) 빌드도 생성.<br/>"
  "&bull; <b>backend = \"onnx\"</b>, <b>device = \"gpu\"</b>(calibration은 호스트 GPU에서 수행).")
code(r"""
# (컴파일러 venv)
. .venv_qbc/bin/activate
python - <<'PY'
from qbcompiler import mxq_compile
CALIB = "accv_experiments/results/qbc_calib200_train"

# (1) COCO 사전학습 -> INT8, global8
mxq_compile(model="yolo11s.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_coco_traincalib_global8.mxq",
            config_preset="yolo_640", inference_scheme="global8",
            device="gpu", backend="onnx")

# (2) 파인튜닝 -> INT8, global8  (동일 레시피/calibration, 가중치만 다름)
mxq_compile(model="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_ft_traincalib_global8.mxq",
            config_preset="yolo_640", inference_scheme="global8",
            device="gpu", backend="onnx")

# (3) 파인튜닝 -> INT8, single 모드 (N=4 동시 NPU 스트림용)
mxq_compile(model="accv_experiments/results/ft_runs/ft_yolo11s/weights/best.onnx",
            calib_data_path=CALIB,
            save_path="accv_experiments/results/qbc_ft_traincalib_single.mxq",
            config_preset="yolo_640", inference_scheme="single",
            device="gpu", backend="onnx")
PY
""")
P("각 컴파일은 약 30~33초 소요, ~10.7MB .mxq 산출(parse → 양자화/calibration → ARIES GlobalCluster "
  "빌드 → export & 검증).",NOTE)

P("7. 단계 6 — 컴파일 후 검증",H1)
P("&bull; <b>결정성/bit-identity:</b> 각 .mxq를 20프레임 2회 실행 → box/score diff = 0.<br/>"
  "&bull; <b>operating point:</b> 단일 스트림 평가 threads=4, frame-skip ≈ 0.<br/>"
  "&bull; <b>컴파일 전 FP32 게이트:</b> 파인튜닝 vs COCO offline mAP(small +0.045, medium +0.069)로 "
  "INT8 변환 전 정확도 상승을 확인.")

P("8. 정확도 — fine-tuning 전/후 비교 (per-size sAP)",H1)
P("동일 24개 Argoverse-HD val log, 단일 스트림, threads=4(frame skip≈0), 3회 반복 평균. COCO(원본, "
  "학습 안 함)와 fine-tuned 모델을 FP32(GPU)·INT8(NPU) 두 형식에서 측정한 streaming sAP.",BODY)
P("sAP는 streaming(전달 시각 기준) 정확도, mAP는 offline(표준 COCO, 전달 지연 무관) 정확도다. 아래에 둘 다 제시한다.",NOTE)
acc=[["모델 / 형식","all","small","medium","large"],
     ["COCO  — FP32 (GPU)","0.1957","0.0159","0.1839","0.4768"],
     ["COCO  — INT8 (NPU)","0.1874","0.0090","0.1535","0.4796"],
     ["fine-tuned — FP32 (GPU)","0.2185","0.0485","0.2373","0.4559"],
     ["fine-tuned — INT8 (NPU)","0.1920","0.0316","0.1976","0.4362"]]
ta=Table(acc,colWidths=[175,70,70,70,70])
ta.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("FONTNAME",(0,0),(-1,-1),"HYGothic-Medium"),("FONTSIZE",(0,0),(-1,-1),7.6),
 ("FONTNAME",(1,1),(-1,-1),"Courier"),
 ("BACKGROUND",(0,3),(-1,4),colors.HexColor("#eef6ee")),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(Spacer(1,2)); E.append(ta); SP(4)
P("offline mAP @0.5:0.95 (위 sAP와 동일 harness·24 val log·3반복, per-log 평균):",H2)
mp=[["모델 / 형식","offline mAP@0.5:0.95"],
    ["COCO  — FP32 (GPU)","0.2438"],
    ["COCO  — INT8 (NPU)","0.2311"],
    ["fine-tuned — FP32 (GPU)","0.2700"],
    ["fine-tuned — INT8 (NPU)","0.2339"]]
tmp=Table(mp,colWidths=[210,150])
tmp.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("FONTNAME",(0,0),(-1,-1),"HYGothic-Medium"),("FONTSIZE",(0,0),(-1,-1),7.6),
 ("FONTNAME",(1,1),(-1,-1),"Courier"),
 ("BACKGROUND",(0,3),(-1,4),colors.HexColor("#eef6ee")),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(Spacer(1,2)); E.append(tmp); SP(2)
P("(per-size offline mAP는 4셀 전부 저장돼 있지 않아 overall만 제시. FP32 per-size mAP는 아래 게이트 표 참고.)",NOTE)
P("참고 — 컴파일 전 FP32 게이트 (val 전수 <b>pooled</b> offline mAP, per-size; COCO vs fine-tuned). 위 per-log 집계와 방식이 달라 절대값이 다르다:",H2)
acc2=[["size","COCO (FP32)","fine-tuned (FP32)","Δ"],
      ["all","0.1844","0.2131","+0.0287"],
      ["small","0.0125","0.0576","+0.0451"],
      ["medium","0.1734","0.2421","+0.0688"],
      ["large","0.5029","0.4419","-0.0611"]]
tb=Table(acc2,colWidths=[90,120,135,80])
tb.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("FONTNAME",(0,0),(-1,-1),"HYGothic-Medium"),("FONTSIZE",(0,0),(-1,-1),7.6),
 ("FONTNAME",(1,1),(-1,-1),"Courier"),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(Spacer(1,2)); E.append(tb); SP(4)
P("해석: fine-tuning으로 <b>small·medium 정확도가 크게 상승</b>(small은 ~3–4배), <b>large는 소폭 하락</b>. "
  "INT8(NPU)도 같은 경향을 따른다. 핵심은 fine-tuning으로 정확도를 올린 뒤에도 INT8의 small-biased "
  "quantization 구조(세로 차이: small 손실 ≫ large)가 유지된다는 점이다.",NOTE)
P("9. 산출물 (SHA-256 앞자리)",H1)
tdata=[["파일","역할","sha256[:16]"],
 ["yolo11s.pt","COCO 사전학습 초기가중치(Ultralytics v8.2.100)","85a76fe86dd8afe3"],
 ["ft_yolo11s/weights/best.pt","파인튜닝 FP32(epoch 1 best)","4c4b342a6d09de67"],
 ["yolo11s.onnx","COCO ONNX(640, opset13)","5ffc807bb21ba5f5"],
 ["best.onnx","파인튜닝 ONNX(640, opset13)","8e189e0252604e8f"],
 ["qbc_coco_traincalib_global8.mxq","COCO INT8(global8)","f987aca54e64c363"],
 ["qbc_ft_traincalib_global8.mxq","파인튜닝 INT8(global8)","ea091a9dc2485a79"],
 ["qbc_ft_traincalib_single.mxq","파인튜닝 INT8(single)","c0520594743655e4"]]
t=Table(tdata,colWidths=[150,205,95])
t.setStyle(TableStyle([
 ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#244b7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
 ("FONTNAME",(0,0),(-1,-1),"HYGothic-Medium"),("FONTSIZE",(0,0),(-1,-1),7.2),("FONTNAME",(0,0),(-1,0),"HYGothic-Medium"),
 ("FONTNAME",(2,1),(2,-1),"Courier"),
 ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c2c8d0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
 ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f4f6")]),
 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
E.append(t); SP(8)
P("스크립트: ft_prep_dataset.py, ft_train_driver.sh, ft_dual_compile.py, ft_stretch_vlm.py "
  "(모두 accv_experiments/scripts/). 검출기 COCO 사전학습 가중치: Ultralytics GitHub assets 릴리스. "
  "NPU 컴파일러: qbcompiler 1.1.2(+aries2), Mobilint ARIES / MLA100.",NOTE)

SimpleDocTemplate(OUT,pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,topMargin=15*mm,bottomMargin=14*mm,
                  title="MLA100 학습·컴파일 절차").build(E)
print("written",OUT)
