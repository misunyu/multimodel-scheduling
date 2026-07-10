#!/usr/bin/env python3
"""
Best Deploy Finder & Executor
This script launches the Best Deploy Finder GUI application, loading
best_deploy_finder_executor.ui and showing the models folder structure
in the model_tree_view. Top-level folders under the root are checkable.

Usage:
    python best_deploy_finder_executor.py [--models-root MODELS_DIR]
"""

import os
import sys
import argparse
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QLabel, QDoubleSpinBox, QWidget, QHBoxLayout, QGridLayout


def discover_file_backed_models(models_root: str):
    """Vision models deployable from the models/ folder.

    `models/onnx/<name>.onnx`     -> runnable on CPU
    `models/mobilint/<name>.mxq`  -> the compiled form, runnable on the Mobilint NPU

    A model is deployable when both exist. Returns (deployable, cpu_only, npu_only)
    as sorted lists of bare names (no extension).
    """
    def _names(d, ext):
        if not os.path.isdir(d):
            return set()
        return {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(ext)}

    cpu = _names(os.path.join(models_root, 'onnx'), '.onnx')
    npu = _names(os.path.join(models_root, 'mobilint'), '.mxq')
    return sorted(cpu & npu), sorted(cpu - npu), sorted(npu - cpu)


def discover_hf_backed_models(static_json_path):
    """Generative models (LLM / VLM) deployable from HuggingFace checkpoints.

    These have no local .onnx/.mxq — the Mobilint runtime loads them from the Hub —
    so availability is gated on having a static profile entry (i.e. they were
    profiled and therefore have the features the predictor needs).
    """
    try:
        import json
        import model_registry as reg
        profiled = {r.get('model') for r in
                    json.loads(Path(static_json_path).read_text()).get('total_data', [])}
        return sorted(m for m in reg.model_names()
                      if reg.kind_of(m) in ('llm', 'vlm') and m in profiled)
    except Exception:
        return []


class ModelChecklistModel(QStandardItemModel):
    """Checkable list of model names (no extension) shown in model_tree_view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Model"])

    def populate(self, names):
        self.removeRows(0, self.rowCount())
        for name in names:
            item = QStandardItem(name)
            item.setCheckable(True)
            item.setCheckState(Qt.Unchecked)
            item.setEditable(False)
            self.appendRow(item)

    def checked_model_names(self):
        return [self.item(r).text() for r in range(self.rowCount())
                if self.item(r).checkState() == Qt.Checked]


class BestDeployFinderApp(QMainWindow):
    def __init__(self, models_root=None):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'best_deploy_finder_executor.ui'), self)

        # Default models root to ./models
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'models')

        # The tree view lists deployable model NAMES (no extension) with checkboxes.
        self.model_list = ModelChecklistModel(self)
        self.model_tree_view.setModel(self.model_list)
        self.model_tree_view.setRootIsDecorated(False)
        self.model_tree_view.setHeaderHidden(False)
        self._reload_model_list()

        # Wire up browse buttons if present
        if hasattr(self, 'deploy_model_browse_button'):
            self.deploy_model_browse_button.clicked.connect(self.select_models_folder)
        if hasattr(self, 'prediction_model_browse_button'):
            self.prediction_model_browse_button.clicked.connect(self.select_prediction_model)
        if hasattr(self, 'device_conf_browse_button'):
            self.device_conf_browse_button.clicked.connect(self.select_device_config)
        # Wire up custom buttons
        if hasattr(self, 'input_rate_button'):
            self.input_rate_button.clicked.connect(self.on_input_rate_clicked)
        if hasattr(self, 'predict_best_button'):
            self.predict_best_button.clicked.connect(self.on_predict_best_clicked)
        if hasattr(self, 'load_execute_best_button'):
            self.load_execute_best_button.clicked.connect(self.on_load_execute_best_clicked)

        # Initialize line edits if present
        if hasattr(self, 'deployment_model_input'):
            self.deployment_model_input.setText(self.models_root)
        # Default the placement predictor to the CPU + Mobilint-NPU model. Set it
        # explicitly (not just when empty) so the .ui's directory default does not
        # cause an ambiguous prefix among the several trained models.
        _cpu_npu_prefix = os.path.join(os.path.dirname(__file__), 'xgboost_model', 'artifacts', 'deploy_cpu_npu')
        if hasattr(self, 'prediction_model_input') and os.path.exists(_cpu_npu_prefix + '_y1.json'):
            self.prediction_model_input.setText(_cpu_npu_prefix)
        if hasattr(self, 'device_config_input') and not self.device_config_input.text():
            self.device_config_input.setText(
                os.path.join(os.path.dirname(__file__), 'target_device_cpu_npu.yaml'))

        # Initialize log window if present
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.setReadOnly(True)

        # State: input FPS mapping per model
        self.input_fps_by_model = {}

        # Default outputs
        self.generated_schedule_path = os.path.join(os.path.dirname(__file__), 'model_schedules.yaml')

    def _log(self, message: str):
        if hasattr(self, 'log_text_edit') and self.log_text_edit is not None:
            # QPlainTextEdit supports appendPlainText, not append
            self.log_text_edit.appendPlainText(message)
        else:
            print(message)

    def _reload_model_list(self):
        """Rescan and show every deployable model name in the tree view.

        Vision models come from models/onnx + models/mobilint; generative models
        (LLM/VLM) come from the registry and load from HuggingFace.
        """
        file_backed, cpu_only, npu_only = discover_file_backed_models(self.models_root)
        try:
            hf_backed = discover_hf_backed_models(self._resolve_static_json())
        except Exception:
            hf_backed = []
        deployable = sorted(set(file_backed) | set(hf_backed))
        self.model_list.populate(deployable)
        self._log(f"[Models] vision (onnx + mxq): {', '.join(file_backed) or 'none'}")
        if hf_backed:
            self._log(f"[Models] generative (HuggingFace): {', '.join(hf_backed)}")
        if cpu_only:
            self._log(f"[Models] Skipped (no .mxq for NPU): {', '.join(cpu_only)}")
        if npu_only:
            self._log(f"[Models] Skipped (no .onnx for CPU): {', '.join(npu_only)}")

    def _get_selected_model_names(self):
        """Model names checked in the tree view."""
        models = self.model_list.checked_model_names()
        if models:
            self._log(f"[Info] Selected models: {', '.join(models)}")
        return models

    def on_input_rate_clicked(self):
        models = self._get_selected_model_names()
        if not models:
            self._log("[Warning] No models selected. Please select folders in the model tree.")
            return
        # Load the dialog UI
        dialog_ui_path = os.path.join(os.path.dirname(__file__), 'input_rate_dialog.ui')
        dlg = QDialog(self)
        uic.loadUi(dialog_ui_path, dlg)

        # Find the container layout inside the scroll area widget
        container_widget = dlg.findChild(QWidget, 'scrollAreaWidgetContents')
        container_layout = container_widget.layout() if container_widget else None
        if container_layout is None:
            self._log('[Error] Failed to locate rates_container layout in dialog UI.')
            return

        # Clear any placeholder items
        while container_layout.count():
            item = container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        # Add rows to a grid layout: labels aligned to the longest name width
        spin_boxes = {}
        # Determine pixel width of the longest model name for alignment
        fm = container_widget.fontMetrics()
        adv = getattr(fm, 'horizontalAdvance', None)
        def _w(t: str) -> int:
            return adv(t) if callable(adv) else fm.width(t)
        max_label_px = max((_w(m) for m in models), default=0)
        pad_px = 16

        # Ensure we have a QGridLayout
        if not isinstance(container_layout, QGridLayout):
            grid = QGridLayout(container_widget)
            grid.setContentsMargins(0, 0, 0, 0)
            container_widget.setLayout(grid)
            container_layout = grid
        try:
            container_layout.setColumnStretch(0, 0)
            container_layout.setColumnStretch(1, 1)
        except Exception:
            pass

        for row, model in enumerate(models):
            label = QLabel(model, container_widget)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setMinimumWidth(max_label_px + pad_px)

            spin = QDoubleSpinBox(container_widget)
            spin.setDecimals(1)
            spin.setMinimum(0.1)
            spin.setMaximum(1000.0)
            spin.setSingleStep(0.1)
            # Pre-fill from existing mapping or default 30.0
            spin.setValue(float(self.input_fps_by_model.get(model, 10.0)))

            container_layout.addWidget(label, row, 0)
            container_layout.addWidget(spin, row, 1)
            spin_boxes[model] = spin

        # Attach spin boxes dict for retrieval on accept
        dlg._spin_boxes_by_model = spin_boxes

        # Resize dialog to fit its contents tightly (height varies with model count)
        try:
            dlg.adjustSize()
        except Exception:
            pass

        if dlg.exec_() == QDialog.Accepted:
            for model, spin in spin_boxes.items():
                self.input_fps_by_model[model] = float(spin.value())
            # Log results
            pairs = ", ".join([f"{m}: {v:.1f}" for m, v in sorted(self.input_fps_by_model.items())])
            self._log(f"[Info] Updated input rates: {pairs}")

    def select_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Models Folder', self.models_root)
        if folder:
            self.models_root = folder
            if hasattr(self, 'deployment_model_input'):
                self.deployment_model_input.setText(folder)
            self._reload_model_list()

    def select_prediction_model(self):
        # Expect a folder that contains <prefix>_y1.json and <prefix>_y2.json
        path = QFileDialog.getExistingDirectory(self, 'Select Prediction Model Folder', os.getcwd())
        if path and hasattr(self, 'prediction_model_input'):
            self.prediction_model_input.setText(path)

    def select_device_config(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Device Configuration', os.getcwd(), 'YAML Files (*.yaml *.yml);;All Files (*)')
        if path and hasattr(self, 'device_config_input'):
            self.device_config_input.setText(path)

    def get_checked_top_level_dirs(self):
        """Backwards-compatible alias: the tree now holds model names, not folders."""
        return self.model_list.checked_model_names()

    def log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        text = f'[{ts}] {msg}'
        if hasattr(self, 'log_text_edit') and self.log_text_edit is not None:
            self.log_text_edit.appendPlainText(text)
        print(text)

    def _resolve_static_json(self):
        """Locate the static profile JSON (feature source), preferring the xgboost_model path."""
        root = Path(__file__).resolve().parent
        candidates = [
            root / "xgboost_model" / "performance_data" / "sample_profiling_data" / "sample_profiling_data.json",
            root / "sample_profiling_data.json",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            raise FileNotFoundError(f"Static profiling JSON not found in: {[str(c) for c in candidates]}")
        return p

    def build_schedule_from_selection(self, model_names, out_path: str) -> str:
        """Generate CPU + Mobilint-NPU placement candidates for the selected models.

        Replaces the legacy CPU + Neubla-NPU flow: the platform is fixed to
        CPU + NPU (both devices shareable, matching the trained `deploy_cpu_npu`
        predictor). Every checked model is placed on cpu or npu; per-view infps
        defaults to the model's baseline rate (overridable via the input-rate
        dialog). Writes the schedule YAML plus a `<out>.meta.json` sidecar.
        """
        import json
        import yaml
        import model_registry as reg
        import generate_schedules as gs

        models = [m for m in (model_names or []) if m in reg.MODELS]
        unknown = [m for m in (model_names or []) if m not in reg.MODELS]
        if unknown:
            self.log(f"[Warn] Not in the model registry, ignored: {', '.join(unknown)}")
        if not models:
            raise ValueError("No models selected. Check at least one model in the list.")
        if len(models) > 4:
            self.log(f"[Warn] {len(models)} models selected; the executor renders at most 4 views.")

        self.log(f"[Info] Platform: CPU + Mobilint NPU (both shareable). Models: {', '.join(models)}")

        # Baseline rates for default infps (1x). Overridable per model via the dialog.
        static_json = self._resolve_static_json()
        baseline = {}
        try:
            baseline = {r['model']: r.get('baseline_rate')
                        for r in json.loads(Path(static_json).read_text()).get('total_data', [])}
        except Exception as e:
            self.log(f"[Warn] Could not load baseline rates: {e}")

        platform_devices = ["cpu", "npu"]
        workload_id = next((m for m in models if reg.get(m).get("task") == "detection"), models[0])
        schedules, meta = {}, {}
        for idx, placement in enumerate(gs.enumerate_placements(models, platform_devices), start=1):
            name = f"combination_{idx}"
            entry = {}
            for j, (model, dev) in enumerate(placement.items()):
                infps = None
                v = (getattr(self, 'input_fps_by_model', {}) or {}).get(model)
                if v is not None:
                    try:
                        infps = float(v)
                    except Exception:
                        infps = None
                if infps is None:
                    infps = float(baseline.get(model) or 1.0)
                entry[f"{model}_{dev}"] = {
                    "model": model, "execution": dev,
                    "display": f"view{j + 1}", "infps": round(infps, 3),
                }
            schedules[name] = entry
            meta[name] = {"rate_factor": 1.0, "workload": workload_id}

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("# model_schedules.yaml - CPU + Mobilint NPU placement candidates\n\n")
                f.write(yaml.dump(schedules, default_flow_style=False, sort_keys=False))
            Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            raise RuntimeError(f"Failed to write schedule YAML '{out_path}': {e}")
        self.log(f"[Info] Wrote {len(schedules)} CPU/NPU combinations to {out_path}")
        return out_path

    def generate_all_combinations(self) -> str:
        """Enumerate CPU/NPU placements of the models checked in the tree view.

        The platform is fixed to CPU + Mobilint NPU, so no device config is needed.
        Per-model input rates come from the input-rate dialog, else the baseline rate.
        Returns the generated schedule YAML path.
        """
        models = self._get_selected_model_names()
        if not models:
            raise ValueError("No models selected. Check at least one model in the list.")
        return self.build_schedule_from_selection(models, self.generated_schedule_path)

    def predict_best_combination(self, schedule_yaml_path: str, model_input_path: str, alpha: float = 0.3, beta: float = 0.5):
        """Predict best combination using three-target XGBoost JSON models.
        - model_input_path can be either:
          - A directory containing files: <prefix>_y1.json, <prefix>_y2.json (and _y3.json)
          - One of the JSON files (we'll infer the prefix and counterparts)
        Returns (best_combination_name, df) where df contains columns:
          [source, combination, pred_total_throughput_fps, pred_drop_rate_fps,
           pred_tokens_per_s, pred_score].
        Score = fps + beta*tokens_per_s - alpha*drop_rate.
        """
        import pandas as pd
        from pathlib import Path
        from xgboost_model.deploy_selector_xgb_suite import (
            load_static_profiles,
            featurize_from_combo,
            predict_targets,
            _load_yaml_or_json,
            _iter_combos_from_schedule,
        )

        def _infer_model_prefix(p: Path) -> Path:
            # Bare prefix (e.g. .../deploy_cpu_npu) whose _y1.json exists.
            if not p.is_dir() and Path(str(p) + "_y1.json").exists():
                return p
            if p.is_dir():
                y1_files = sorted(p.glob("*_y1.json"))
                for y1 in y1_files:
                    prefix = y1.with_suffix("")  # remove .json
                    if prefix.name.endswith("_y1"):
                        prefix = prefix.with_name(prefix.name[:-3])
                    y2 = p / f"{prefix.name}_y2.json"
                    if y2.exists():
                        return p / prefix.name
                raise FileNotFoundError(f"No valid model prefix with _y1.json and _y2.json found in: {p}")
            else:
                name = p.name
                if name.endswith("_y1.json"):
                    prefix = name[:-len("_y1.json")]
                elif name.endswith("_y2.json"):
                    prefix = name[:-len("_y2.json")]
                else:
                    raise ValueError(f"Model file must end with _y1.json or _y2.json: {p}")
                y1 = p.parent / f"{prefix}_y1.json"
                y2 = p.parent / f"{prefix}_y2.json"
                if not y1.exists() or not y2.exists():
                    raise FileNotFoundError(f"Missing counterpart JSON next to {p}. Expected both {y1.name} and {y2.name}.")
                return p.parent / prefix

        sched_path = Path(schedule_yaml_path)
        if not sched_path.exists():
            raise FileNotFoundError(f"Schedule YAML not found: {schedule_yaml_path}")
        model_prefix = _infer_model_prefix(Path(model_input_path))

        static_json_path = self._resolve_static_json()

        S = load_static_profiles(static_json_path)
        schedule_doc = _load_yaml_or_json(sched_path)
        combos = _iter_combos_from_schedule(schedule_doc)
        if not combos:
            raise ValueError("No combinations found in schedule YAML.")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(S, combo_blob)
            y1_pred, y2_pred, y3_pred = predict_targets(model_prefix, X)
            # y1 = norm throughput, y2 = deadline miss rate, y3 = norm tokens
            fps = float(y1_pred[0]); miss = float(y2_pred[0]); tok = float(y3_pred[0])
            score = fps + float(beta) * tok - float(alpha) * miss
            rows.append({
                "source": sched_path.name,
                "combination": str(name),
                "pred_norm_throughput": fps,
                "pred_deadline_miss_rate": miss,
                "pred_norm_tokens": tok,
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

    def _build_cpu_only_schedule(self, model_names, out_path: str) -> str:
        """Build a schedule with a single combination where all selected models run on CPU
        using the per-model input rates previously set by the user (input_rate_dialog).
        Returns the output YAML path.
        """
        import yaml
        models = [m for m in (model_names or []) if m]
        if not models:
            raise ValueError("No models selected. Please check at least one model in the list.")
        # Limit to 4 views for viewer layout consistency (mirrors other code paths)
        if len(models) > 4:
            self.log(f"[Warn] More than 4 models selected. Using only the first 4.")
            models = models[:4]
        schedules = {
            "combination_1": {}
        }
        for j, model in enumerate(models):
            infps = None
            if isinstance(getattr(self, 'input_fps_by_model', None), dict):
                v = self.input_fps_by_model.get(model)
                try:
                    if v is not None:
                        infps = float(v)
                except Exception:
                    infps = None
            # Reasonable defaults if not provided
            if infps is None:
                lname = model.lower()
                if "resnet50" in lname:
                    infps = 2.0
                elif "yolov3" in lname:
                    infps = 30.0
                else:
                    infps = 10.0
            entry = {
                "model": model,
                "execution": "cpu",
                "display": f"view{j+1}",
                "infps": float(infps),
            }
            schedules["combination_1"][f"{model}_cpu"] = entry
        # Write YAML
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("# model_schedules.yaml\n")
            f.write("# Auto-generated (CPU-only)\n\n")
            f.write(yaml.dump(schedules, default_flow_style=False))
        return out_path

    def _launch_executor_subprocess(self, schedule_path: str, combo_name: str = None, duration: int = None):
        """Launch schedule_executor_main.py in a separate process to avoid nested QApps.
        If combo_name is provided, run executor-only mode for that single combination.
        """
        import subprocess
        py = sys.executable or 'python'
        exec_path = os.path.join(os.path.dirname(__file__), 'schedule_executor_main.py')
        args = [py, exec_path, '--schedule', schedule_path]
        if duration is not None:
            try:
                d = int(duration)
                args += ['--duration', str(max(1, d))]
            except Exception:
                pass
        if combo_name:
            args += ['--schedule-name', combo_name]
        self.log(f"[Exec] Launching executor: {' '.join(args)}")
        try:
            subprocess.Popen(args)
        except Exception as e:
            self.log(f"[Error] Failed to launch executor: {e}")

    def on_load_execute_best_clicked(self):
        """Load best predicted deployment and start execution.
        If no predictions exist, run all selected models on CPU at user-specified input rates.
        """
        # 1) Try to read best from predictions.csv
        predictions_csv = os.path.join(os.path.dirname(__file__), 'predictions.csv')
        schedule_path = self.generated_schedule_path
        best_combo = None
        if os.path.exists(predictions_csv):
            try:
                import csv
                with open(predictions_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    first = next(reader, None)
                    if first and 'combination' in first and first['combination']:
                        best_combo = first['combination']
                        self.log(f"[Load] Using best combination from predictions.csv: {best_combo}")
            except Exception as e:
                self.log(f"[Warn] Failed to parse predictions.csv: {e}")
        # 2) If no predictions, create a CPU-only schedule for the selected models
        if not best_combo:
            selected = self._get_selected_model_names()
            if not selected:
                self.log("[Error] No models selected. Please check at least one model in the list.")
                return
            try:
                schedule_path = self._build_cpu_only_schedule(selected, schedule_path)
                best_combo = 'combination_1'
                self.log(f"[Build] Created CPU-only schedule: {schedule_path}")
            except Exception as e:
                self.log(f"[Error] Failed to build CPU-only schedule: {e}")
                return
        # 3) Ensure schedule exists
        if not os.path.exists(schedule_path):
            self.log(f"[Error] Schedule file not found: {schedule_path}")
            return
        # Optional: pick duration from UI if available later; for now, default to 60
        duration = 60
        # 4) Launch executor in a subprocess with selected combo
        self._launch_executor_subprocess(schedule_path, combo_name=best_combo, duration=duration)

    def on_predict_best_clicked(self):
        """Handler invoked when predict_best_button is clicked."""
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        pred_model = self.prediction_model_input.text() if hasattr(self, 'prediction_model_input') else ''
        device_conf = self.device_config_input.text() if hasattr(self, 'device_config_input') else ''

        # Models checked in the tree view (names, no extension)
        selected = self._get_selected_model_names()

        # Log inputs
        self.log(f"[Predict] models_root={models_root}")
        self.log(f"[Predict] prediction_model={pred_model}")
        self.log(f"[Predict] selected_models={selected}")

        # Validate. The platform is fixed to CPU + Mobilint NPU, so no device
        # config is required. `pred_model` may be a prefix, a directory, or a _y*.json.
        try:
            if not selected:
                raise ValueError("No models selected. Please check at least one model in the list.")
            if not pred_model or not (os.path.exists(pred_model) or os.path.exists(pred_model + '_y1.json')):
                raise FileNotFoundError(f"Prediction model not found: {pred_model}")
        except Exception as e:
            self.log(f"[Error] {e}")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            return

        # Step 1: Generate schedule YAML (using generate_all_combinations)
        try:
            schedule_path = self.generate_all_combinations()
            self.log(f"[Step1] Generated schedule YAML: {schedule_path}")
        except Exception as e:
            self.log(f"[Error][Step1] {e}")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            return

        # Step 2: Run prediction using XGBoost model
        try:
            best_combo, df = self.predict_best_combination(schedule_path, pred_model)
            if not best_combo:
                raise RuntimeError("Prediction produced no result.")
            # Extract combination number (digits at end)
            import re
            m = re.search(r"(\d+)$", best_combo)
            combo_number = m.group(1) if m else best_combo
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText(str(combo_number))
            # Log top predictions summary
            self.log(f"[Step2] Top-1 combination: {best_combo}")
            try:
                topn = min(5, len(df))
                self.log("[Step2] Top predictions:")
                for i in range(topn):
                    self.log(f"  {i+1}. {df.iloc[i]['combination']} -> {float(df.iloc[i]['pred_score']):.4f}")
            except Exception:
                pass
        except Exception as e:
            self.log(f"[Error][Step2] {e}")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Best Deploy Finder GUI')
    parser.add_argument('--models-root', type=str, help='Path to the models root folder (default: ./models)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = QApplication(sys.argv)
    window = BestDeployFinderApp(models_root=args.models_root)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()