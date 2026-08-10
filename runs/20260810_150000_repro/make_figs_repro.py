"""작업 19-r2 [3]: 논문 그림 2종을 이 run 디렉터리로 재생성.

make_figures.py 는 수정하지 않는다. 모듈을 import 한 뒤 출력 디렉터리(FIGDIR)만
이 디렉터리로 바꿔 호출한다 → analysis/figures/ 는 건드리지 않는다.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/msyu/PycharmProjects/multimodel-scheduling-mobilint")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "xgboost_model" / "full_collection_540" / "scripts"))

import make_figures as mf  # noqa: E402

mf.FIGDIR = HERE / "figures"
mf.FIGDIR.mkdir(parents=True, exist_ok=True)

t = {}
t0 = time.time(); info1 = mf.fig_f1("declared_normalized"); t["f1"] = round(time.time() - t0, 1)
t0 = time.time(); info2 = mf.fig_f2(); t["f2"] = round(time.time() - t0, 1)
(HERE / "figs_info.json").write_text(json.dumps(
    {"f1": {str(k): v for k, v in (info1 or {}).items()},
     "f2": {str(k): v for k, v in (info2 or {}).items()},
     "seconds": t, "outdir": str(mf.FIGDIR)}, indent=2, ensure_ascii=False, default=str))
print("wrote to", mf.FIGDIR, t)
