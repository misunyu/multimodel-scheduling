"""QoS body numbers (Sec 4.3): protected All-GPU worst 0.033 vs All-NPU 0.083, 2.5x.
Source exp_qos_results.csv (CUDA-priority foreground); All-NPU 0.083 from tab:main."""
from __future__ import annotations
import sys
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import RES, RD

PKG = RD / "qos"
ALLNPU_VLM_WORST = 0.083  # tab:main L3VLM All-NPU worst


def main():
    d = pd.read_csv(RES / "exp_qos_results.csv")
    prot = d[(d.mechanism == "prio") & (d.cotenant.str.contains("qwen", case=False))]
    # worst-stream per repeat then mean
    worst = prot.groupby("repeat")["sap"].min().mean()
    PKG.mkdir(exist_ok=True)
    df = pd.DataFrame([{"config": "protected All-GPU (CUDA prio, VLM)", "worst_sap": round(worst, 4),
                        "source": "exp_qos_results.csv"},
                       {"config": "All-NPU (VLM)", "worst_sap": ALLNPU_VLM_WORST, "source": "tab:main"}])
    df.to_csv(PKG / "qos_worst.csv", index=False)
    ratio = ALLNPU_VLM_WORST / worst
    (PKG / "qos_numbers.txt").write_text(
        f"protected All-GPU worst-stream sAP (VLM) = {worst:.4f} -> 0.033\n"
        f"All-NPU worst-stream sAP (VLM) = {ALLNPU_VLM_WORST:.3f}\n"
        f"ratio = {ratio:.2f}x -> 2.5x\n")
    print(f"qos: protected All-GPU worst={worst:.4f} (->0.033), All-NPU {ALLNPU_VLM_WORST}, "
          f"ratio {ratio:.2f}x (paper 2.5x)")
    return True


if __name__ == "__main__":
    main()
