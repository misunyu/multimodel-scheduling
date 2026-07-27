"""Pre-MLA100 two-target predictor, kept for paper-figure reproduction only.

The working set these figures were produced on (resnext50, vgg19, yolov4, ...) is not
in the MLA100 predictor's coverage, so those scripts cannot be repointed at the new
bundle -- they would fail the unprofiled-model check. New code must use
deploy_predictor_logic.DeployPredictor (three targets, MLA100 artifacts) instead.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from xgboost_model.deploy_selector_xgb_suite_legacy import (
    featurize_from_combo,
    load_models,
    _load_yaml_or_json,
    _iter_combos_from_schedule,
)

class DeployPredictor:
    def __init__(self, log_callback=None):
        # Loud guard (v22 task 95): this is the pre-MLA100 predictor trained on the
        # OLD working set (resnext50, vgg19, yolov4, ...). It exists ONLY to reproduce
        # already-published legacy figures. New rankings MUST use
        # deploy_predictor_logic.DeployPredictor (three targets, MLA100 bundle). Set
        # FSRR_ALLOW_LEGACY_PREDICTOR=1 to acknowledge and silence this warning.
        import os as _os, sys as _sys
        if _os.environ.get("FSRR_ALLOW_LEGACY_PREDICTOR") != "1":
            print(
                "[LEGACY PREDICTOR] deploy_predictor_logic_legacy is pre-MLA100 "
                "(old vocabulary). Do NOT use it for new rankings -- use "
                "deploy_predictor_logic.DeployPredictor. See docs/ranking_regeneration_report.md. "
                "Set FSRR_ALLOW_LEGACY_PREDICTOR=1 if this is an intentional legacy "
                "reproduction.", file=_sys.stderr)
        self.log_callback = log_callback

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def _infer_model_info(self, p: Path) -> Tuple[Path, Optional[str]]:
        if p.is_dir():
            # Try to infer mode from directory name or content
            for m in ["rank", "score", "double", "two_target"]:
                if m in p.name:
                    return p, m
            
            y1_files = sorted(p.glob("*_y1.json"))
            for y1 in y1_files:
                prefix = y1.with_suffix("")  # remove .json
                if prefix.name.endswith("_y1"):
                    prefix = prefix.with_name(prefix.name[:-3])
                y2 = p / f"{prefix.name}_y2.json"
                if y2.exists():
                    return p / prefix.name, "double"
            raise FileNotFoundError(f"No valid model prefix found in: {p}")
        else:
            name = p.name
            inferred_mode = None
            for m in ["rank", "score", "double", "two_target"]:
                if m in name:
                    inferred_mode = m
                    break
            
            if name.endswith("_y1.json"):
                prefix = name[:-len("_y1.json")]
            elif name.endswith("_y2.json"):
                prefix = name[:-len("_y2.json")]
            elif name.endswith("_score.json"):
                prefix = name[:-len("_score.json")]
                inferred_mode = "score"
            elif name.endswith("_rank.json"):
                prefix = name[:-len("_rank.json")]
                inferred_mode = "rank"
            else:
                prefix = p.stem
            
            return p.parent / prefix, inferred_mode

    def predict_best_combination(self, schedule_yaml_path: Optional[str] = None, model_input_path: str = "", alpha: float = 0.3, schedule_data: Optional[dict] = None):
        """Predict best combination using XGBoost models."""
        if schedule_data is not None:
            schedule_doc = schedule_data
            sched_name = "memory_dict"
        elif schedule_yaml_path is not None:
            sched_path = Path(schedule_yaml_path)
            if not sched_path.exists():
                raise FileNotFoundError(f"Schedule YAML not found: {schedule_yaml_path}")
            schedule_doc = _load_yaml_or_json(sched_path)
            sched_name = sched_path.name
        else:
            raise ValueError("Either schedule_yaml_path or schedule_data must be provided.")
        
        model_prefix, inferred_mode = self._infer_model_info(Path(model_input_path))

        b1, b2, feats, mode, model_alpha = load_models(inferred_mode, alpha, prefix=model_prefix)
        self.log(f"[Predict] Using mode: {mode}")
        if mode not in ["score", "double", "rank", "two_target"]:
            raise ValueError(f"Unsupported model mode: {mode}.")

        # Use model's alpha if available, otherwise use the passed alpha
        effective_alpha = model_alpha if model_alpha is not None else alpha

        combos = _iter_combos_from_schedule(schedule_doc)
        if not combos:
            raise ValueError("No combinations found in schedule.")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(combo_blob)
            # Reindex to match features used during training
            X = X.reindex(columns=feats, fill_value=0.0)

            # Predict using underlying models
            if mode == "rank":
                y_pred = b1.predict(X)
                fps = 0.0; drop = 0.0; score = float(y_pred[0])
            elif mode == "score":
                y_pred = b1.predict(X)
                fps = 0.0; drop = 0.0; score = float(y_pred[0])
            else:
                y1_pred = b1.predict(X)
                y2_pred = b2.predict(X)
                fps = float(y1_pred[0]); drop = float(y2_pred[0])
                score = fps - float(effective_alpha) * drop
            
            self.log(f"[Predict] Combination: {name} -> FPS: {fps:.2f}, Drop: {drop:.2f}, Score: {score:.2f}")
            rows.append({
                "source": sched_name,
                "combination": str(name),
                "pred_total_throughput_fps": fps,
                "pred_drop_rate_fps": drop,
                "pred_score": score,
            })

        df = pd.DataFrame(rows).sort_values(["pred_score"], ascending=[False]).reset_index(drop=True)
        # Save predictions.csv (overwrite)
        try:
            out_csv = os.path.join(os.path.dirname(__file__), 'predictions.csv')
            df.to_csv(out_csv, index=False)
            self.log(f"[Predict] Saved predictions to {out_csv}")
        except Exception as e:
            self.log(f"[Warn] Failed to save predictions.csv: {e}")
        
        best_combo = str(df.iloc[0]["combination"]) if len(df) > 0 else None
        return best_combo, df
