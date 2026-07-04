"""Cell-level cross-check: every package's printed value must appear in main_vision.tex.
Paper is source of truth. Prints PASS/FAIL per package; lists missing tokens.
KNOWN_RECONCILE tokens are documented near-misses (not hard failures)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import paper_text

# per package: tokens that MUST appear literally in the paper (printed values)
EXPECT = {
    "table1_single_stream": ["8.3", "10.0", "0.196", "0.186", "-5.0", "-48.4", "-19.7",
                             "0.477", "0.478", "+0.001", "p<0.01", "p<0.001", "n.s."],
    "table2_decomp": ["-0.008", "-0.036", "+0.001", "-0.001", "-0.030", "-0.098",
                      "-48.4", "-19.7", "-8.0", "-20.5"],
    "table3_main": ["0.098", "0.137", "0.103", "0.083", "0.126", "0.062", "0.100", "0.042",
                    "0.088", "0.063", "0.099", "0.016", "0.047", "0.120", "24", "83", "76", "100"],
    "fig2_sweep": ["(24.3,0.0979)", "(67.3,0.0713)", "(24.3,0.1026)", "(0,0.0836)"],
    "table4_persize": ["0.042", "0.078", "0.096", "0.120", "0.017", "0.032", "0.040", "0.046"],
    "table_policy_comparison": ["0.098 (+0.005)", "0.071 (+0.012)", "0.084 (+0.000)",
                                "0.042 (+0.020)", "0.016 (+0.068)", "0.103", "0.063"],
    "table_metric_sensitivity": ["0.107", "0.126", "0.087", "0.115", "0.072", "0.091", "0.066", "0.084"],
    "table_detector_generality": ["-36\\%", "-1.2\\%", "-48\\%", "+0.1\\%", "-45\\%", "-4.6\\%",
                                  "-44\\%", "-0.3\\%", "-8.1\\%", "0.080/0.011", "0.083/0.016",
                                  "0.082/0.016", "0.060/0.009", "\\approx45\\%", "\\approx36\\%"],
    "qos": ["0.033", "0.083", "2.5"],
    "fig_qvsl": ["c2_q_vs_l_bar.pdf"],
    "appendix_gpu_util": ["68\\%", "3012", "7606"],       # commented tab:gpu-util
    "appendix_util_vs_dm": ["a3_util_vs_dm.pdf"],           # commented fig:util-vs-dm
}
KNOWN_RECONCILE = {
    "table2_decomp": {"-16.1": "package uses full-precision rel% (-16.059); paper prints -16.0 "
                               "(4dp-rounded numerator). 0.1pp display artifact."},
}


def main():
    txt = paper_text()
    all_pass = True
    for pkg, toks in EXPECT.items():
        missing = [t for t in toks if t not in txt]
        recon = KNOWN_RECONCILE.get(pkg, {})
        status = "PASS" if not missing else "FAIL"
        if missing:
            all_pass = False
        print(f"[{status}] {pkg}" + ("" if not missing else f"  missing: {missing}"))
        for tok, why in recon.items():
            print(f"    RECONCILE {tok}: {why}")
    print("\nOVERALL:", "ALL PASS" if all_pass else "some FAIL (see above)")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
