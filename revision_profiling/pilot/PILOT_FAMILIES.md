# Pilot family 사전 확인 (Phase 1b §1.5)

2026-10-13, 실행 전 기록. 재구성 규칙: TinyLLaMA instance 제거 → N=0 제거 → 기존 family와 같으면 제거(`code/gen_schedules.py: reconstructed_families`).

재구성 결과: **22 families**, N≥3 12개, 후보 Σ2^N = 504, run 2,016(4 rates) + 반복 960. Phase 1의 "TinyLLaMA 제외" 시나리오와 일치한다 [확인].

| N | family 수 | family (모델, 이름순) |
|---|---|---|
| 1 | 8 | gpt2 / mnasnet / resnet50 / resnext50 / shufflenet-v2-12 / squeezenet1.0-12 / vgg19 / yolov4 |
| 2 | 2 | gpt2+mnasnet / gpt2+yolov4 |
| 3 | 6 | gpt2+mnasnet+resnet50 / gpt2+resnext50+vgg19 / gpt2+resnext50+yolov4 / mnasnet+resnet50+resnext50 / mnasnet+resnext50+yolov4 / resnet50+vgg19+yolov4 |
| 4 | 1 | gpt2+mnasnet+resnet50+resnext50 |
| 5 | 1 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12 |
| 6 | 2 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12 / mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 |
| 7 | 2 | gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19 / mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19+yolov4 |
| 8 | **0** | — |

## 조건별 확인

| 조건 | 존재 | 선정 (이름순 첫 번째) |
|---|---|---|
| (a) N=4 | 예 (1개) | `gpt2+mnasnet+resnet50+resnext50` |
| (b) N=8 | **아니오** | 없음. 기존 N=8 family 2개는 둘 다 TinyLLaMA를 포함했고, 제외 후 N=7이 되었다 |
| (c) LM 없는 N=6 | 예 (1개) | `mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19` |

(b)가 없으므로 지시문에 따라 대체하지 않고 사용자 확인을 받는다. 영향 범위: §2 nice 비교(N=8, 12 runs)와 §5.2 arrival의 큰-N 절반(54 runs).

## 사용자 확인 (2026-10-13)

(b)의 대체 대상으로 **N=7 `gpt2+mnasnet+resnet50+resnext50+shufflenet-v2-12+squeezenet1.0-12+vgg19`** 를 쓴다(사용자 선택). §2 nice 비교와 §5.2의 큰-N 검증에 적용한다.

"반반" 배치는 이름순 instance의 앞 `N // 2`개를 GPU에, 나머지를 CPU에 둔다: N=4는 `GGCC`(gpt2·mnasnet GPU), N=7은 `GGGCCCC`(gpt2·mnasnet·resnet50 GPU).
