import os
import pandas as pd
from pathlib import Path
from typing import Optional

from xgboost_model.deploy_selector_xgb_suite import (
    load_static_profiles,
    featurize_from_combo,
    predict_targets,
    combo_has_generative,
    score_combo,
    _load_yaml_or_json,
    _iter_combos_from_schedule,
)

# Runtime scoring weights. alpha matches deploy_selector_xgb_suite.DEFAULT_ALPHA.
#
# beta deliberately diverges from the suite's DEFAULT_BETA (1.0): the FSRR/BoundGuard
# paper defines S = y1 + 0.5*y3 - 0.3*y2, so this repo scores at beta=0.5 while the
# mobilint repo scores at 1.0. beta is a policy weight, not a learned parameter, so
# the models are unaffected -- only the ranking of mixed (LM-bearing) working sets is.
# Vision-only sets are identical under both, since the y3 term is dropped entirely.
DEFAULT_ALPHA: float = 0.3
DEFAULT_BETA: float = 0.5


class DeployPredictor:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def _infer_model_prefix(self, p: Path) -> Path:
        """Resolve a directory, a bare prefix, or one target JSON to the shared prefix."""
        # Bare prefix (e.g. .../deploy_cpu_npu) whose _y1.json exists.
        if not p.is_dir() and Path(str(p) + "_y1.json").exists():
            return p
        if p.is_dir():
            for y1 in sorted(p.glob("*_y1.json")):
                prefix = y1.name[:-len("_y1.json")]
                if (p / f"{prefix}_y2.json").exists():
                    return p / prefix
            raise FileNotFoundError(
                f"No valid model prefix with _y1.json and _y2.json found in: {p}")
        name = p.name
        if name.endswith("_y1.json"):
            prefix = name[:-len("_y1.json")]
        elif name.endswith("_y2.json"):
            prefix = name[:-len("_y2.json")]
        elif name.endswith("_y3.json"):
            prefix = name[:-len("_y3.json")]
        else:
            raise ValueError(f"Model file must end with _y1.json, _y2.json or _y3.json: {p}")
        y1 = p.parent / f"{prefix}_y1.json"
        y2 = p.parent / f"{prefix}_y2.json"
        if not y1.exists() or not y2.exists():
            raise FileNotFoundError(
                f"Missing counterpart JSON next to {p}. Expected both {y1.name} and {y2.name}.")
        return p.parent / prefix

    def _resolve_static_json(self) -> Path:
        """Locate the static profile JSON the featurizer reads its per-device costs from."""
        root = Path(__file__).resolve().parent
        candidates = [
            root / "xgboost_model" / "performance_data" / "sample_profiling_data" / "sample_profiling_data.json",
            root / "sample_profiling_data.json",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            raise FileNotFoundError(
                f"Static profiling JSON not found in: {[str(c) for c in candidates]}")
        return p

    def _validate_feature_vector(self, X: pd.DataFrame, model_prefix: Path) -> None:
        """Fail loudly when the featurized row does not match the trained columns.

        `_align_features` zero-fills columns it cannot find and drops ones it does not
        expect, both silently -- a featurizer/predictor mismatch would then produce a
        confident number from a partly-zero vector. Check it instead of trusting it.
        """
        import json
        fpath = Path(str(model_prefix) + "_features.json")
        if not fpath.exists():
            raise FileNotFoundError(
                f"{fpath.name} is missing, so the feature column order cannot be verified. "
                f"XGBoost would consume the columns positionally and silently mispredict.")
        trained = json.loads(fpath.read_text(encoding="utf-8"))
        produced = list(X.columns)
        missing = [c for c in trained if c not in produced]
        extra = [c for c in produced if c not in trained]
        if missing or extra:
            raise ValueError(
                f"Feature mismatch against {fpath.name} (trained={len(trained)}, "
                f"produced={len(produced)}). Missing (would be zero-filled): "
                f"{', '.join(missing) or 'none'}. Unexpected (would be dropped): "
                f"{', '.join(extra) or 'none'}. Retrain the predictor or update the featurizer.")

    def predict_best_combination(self, schedule_yaml_path: Optional[str] = None,
                                 model_input_path: str = "",
                                 alpha: float = DEFAULT_ALPHA,
                                 schedule_data: Optional[dict] = None,
                                 beta: float = DEFAULT_BETA):
        """Rank a schedule's combinations with the three-target XGBoost predictor.

        model_input_path may be a directory, a bare prefix (.../deploy_cpu_npu), or one
        of the target JSONs; the counterparts are resolved from it.

        Returns (best_combination_name, df) where df carries, per combination:
          pred_norm_throughput (y1), pred_deadline_miss_rate (y2), pred_norm_tokens (y3),
          has_generative, pred_score.

        All three targets are normalized to [0,1] WITHIN a working set, so scores rank
        placements inside one set and are meaningless across sets.
        """
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

        model_prefix = self._infer_model_prefix(Path(model_input_path))
        static_json_path = self._resolve_static_json()
        S = load_static_profiles(static_json_path)
        self.log(f"[Predict] Model prefix: {model_prefix.name}, static profiles: {static_json_path.name}")

        combos = _iter_combos_from_schedule(schedule_doc)
        if not combos:
            raise ValueError("No combinations found in schedule.")

        # Every model must have a static profile: `_device_static` returns NaN for an
        # unprofiled model and `featurize_from_combo` fills NaN with 0.0, so a missing
        # profile would otherwise sail through as a confident prediction over an
        # all-zero feature row instead of failing.
        unprofiled = sorted({v.get("model") for _, blob in combos
                             for v in (blob or {}).values()
                             if isinstance(v, dict) and v.get("model") not in S})
        if unprofiled:
            raise ValueError(
                f"No static profile for: {', '.join(unprofiled)}. These models are not in "
                f"{Path(static_json_path).name}, so their features would be all zeros and "
                f"the prediction would be meaningless. Profile them first.")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(S, combo_blob)
            self._validate_feature_vector(X, model_prefix)
            # A vision-only set generates no tokens, so the y3 model is neither loaded
            # nor scored -- otherwise it predicts a nonzero token rate for a set with no
            # LM in it and hands every such combination beta*y3 of free score.
            gen = combo_has_generative(combo_blob)
            y1_pred, y2_pred, y3_pred = predict_targets(model_prefix, X, with_y3=gen)
            y1 = float(y1_pred[0]); y2 = float(y2_pred[0]); y3 = float(y3_pred[0])
            score = score_combo(y1, y2, y3, gen, alpha, beta)

            self.log(f"[Predict] {name} -> T_norm: {y1:.3f}, miss: {y2:.3f}, "
                     f"tok_norm: {y3:.3f}{'' if gen else ' (n/a)'}, S: {score:.3f}")
            rows.append({
                "source": sched_name,
                "combination": str(name),
                "pred_norm_throughput": y1,
                "pred_deadline_miss_rate": y2,
                "pred_norm_tokens": y3,
                "has_generative": gen,
                "pred_score": score,
            })

        df = pd.DataFrame(rows).sort_values(["pred_score"], ascending=[False]).reset_index(drop=True)
        try:
            out_csv = os.path.join(os.path.dirname(__file__), 'predictions.csv')
            df.to_csv(out_csv, index=False)
            self.log(f"[Predict] Saved predictions to {out_csv}")
        except Exception as e:
            self.log(f"[Warn] Failed to save predictions.csv: {e}")

        best_combo = str(df.iloc[0]["combination"]) if len(df) > 0 else None
        return best_combo, df
