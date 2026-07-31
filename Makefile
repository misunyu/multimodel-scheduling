# 그림 재생성과 검사의 단일 진입점.
#
# 새로 만든 것은 없다 — 이미 있던 스크립트를 잇는다. 진입점이 하나여야 하는 이유는
# `figure_regeneration_report.md`가 "생성했다"고 기록한 스크립트 5종이 정작 `fig/`에 없었던 일
# 때문이다(runs/README.md "세 번의 증발" ②). 무엇을 돌리면 그림이 다시 나오는지가 한 곳에 있어야 한다.
#
#   make figures   fig/plot_*_v11.py 전부 실행 → figures/*.pdf + 사이드카
#   make check     그림↔확정표(check_figures) + 논문↔확정표(check_paper)
#   make paper     paper/build.sh (4패스)
#   make lock      새 main.tex 를 받은 뒤 paper.lock 재동기화
#
# `figures` 는 저장소 루트의 `figures/` 심볼릭 링크를 거쳐 `docs/figures/` 를 가리킨다.
# 논문의 \graphicspath 가 ../figures/ 이므로 링크가 그 경로를 성립시킨다 — 복사본이 아니다.

# matplotlib 은 PDF 에 /CreationDate 를 찍는다. 그대로 두면 **내용이 같아도** 재생성할 때마다
# sha256 이 바뀌어 paper.lock 이 매번 거짓으로 깨진다(실측 확인). SOURCE_DATE_EPOCH 를 고정하면
# 바이트까지 동일해진다. 값 자체는 의미가 없고 고정돼 있다는 것만이 의미다.
#
# 주의: 현재 커밋된 PDF 들은 이 고정 이전에 생성됐다. 이 변경 이후 `make figures` 를 처음 돌리면
# 그림 해시가 한 번 바뀌므로 `make lock` 으로 재동기화해야 한다. 그 다음부터는 안정적이다.
SOURCE_DATE_EPOCH ?= 1767225600
export SOURCE_DATE_EPOCH

PLOTS := $(sort $(wildcard fig/plot_*_v11.py))

.PHONY: help figures check check-full paper lock

help:
	@echo "make figures   그림·사이드카 재생성 ($(words $(PLOTS)) scripts)"
	@echo "make check     그림↔확정표, 논문↔확정표 대조"
	@echo "make check-full  check + 미추적 수치 줄 목록"
	@echo "make paper     배포본 4패스 빌드"
	@echo "make lock      새 main.tex 수령 후 paper.lock 재동기화"

figures:
	@for p in $(PLOTS); do echo "--> $$p"; python3 $$p || exit 1; done

check:
	python3 fig/check_figures.py --figdir fig
	@echo
	python3 scripts/gen_paper_lock.py
	@echo
	python3 scripts/check_paper.py

check-full:
	python3 fig/check_figures.py --figdir fig
	@echo
	python3 scripts/check_paper.py --list-untracked

paper:
	bash paper/build.sh

lock:
	python3 scripts/gen_paper_lock.py --write
