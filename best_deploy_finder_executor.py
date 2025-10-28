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
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QDialog, QLabel, QSpinBox, QWidget, QHBoxLayout, QGridLayout
from schedule_generator.file_manager import FileManager


class CheckableFileSystemModel(QFileSystemModel):
    """QFileSystemModel where only immediate children of the root are checkable (directories)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_index = None
        self._check_states = {}  # path -> Qt.CheckState

    def set_root_index(self, index):
        self._root_index = index

    def is_top_level_child(self, index):
        if not index.isValid() or self._root_index is None:
            return False
        return self.parent(index) == self._root_index

    def flags(self, index):
        base = super().flags(index)
        if index.column() == 0 and self.isDir(index) and self.is_top_level_child(index):
            return base | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return base

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.CheckStateRole and index.column() == 0 and self.isDir(index) and self.is_top_level_child(index):
            path = self.filePath(index)
            return self._check_states.get(path, Qt.Unchecked)
        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == 0 and self.isDir(index) and self.is_top_level_child(index):
            path = self.filePath(index)
            self._check_states[path] = Qt.Checked if value == Qt.Checked else Qt.Unchecked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        return super().setData(index, value, role)

    def get_checked_top_level_dirs(self):
        """Return absolute paths of checked immediate child directories under the current root."""
        return [path for path, state in self._check_states.items() if state == Qt.Checked]


class BestDeployFinderApp(QMainWindow):
    def __init__(self, models_root=None):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'best_deploy_finder_executor.ui'), self)

        # Default models root to ./models
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'models')

        # Setup file system model and tree view
        self.fs_model = CheckableFileSystemModel(self)
        self.fs_model.setRootPath(self.models_root)
        root_index = self.fs_model.index(self.models_root)
        self.fs_model.set_root_index(root_index)

        # Hook the model to the tree view defined in the UI
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setRootIndex(root_index)
        # Show only name column
        for col in range(1, self.fs_model.columnCount()):
            self.model_tree_view.setColumnHidden(col, True)
        # Expand one level for visibility
        self.model_tree_view.expand(root_index)

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

    def _get_selected_model_names(self):
        # Prefer checked top-level directories (checkbox state) as the source of truth
        if hasattr(self.fs_model, 'get_checked_top_level_dirs'):
            checked_dirs = self.fs_model.get_checked_top_level_dirs()
            if checked_dirs:
                # Map to folder basenames (model names)
                models = [os.path.basename(p) for p in checked_dirs if os.path.isdir(p)]
                models = sorted([m for m in models if m])
                if models:
                    self._log(f"[Info] Using checked models: {', '.join(models)}")
                    return models
        # Fallback to highlighted selection for compatibility
        fm = FileManager(log_callback=self._log)
        return fm.get_models_from_selection(self.model_tree_view, self.fs_model, self.models_root)

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

            spin = QSpinBox(container_widget)
            spin.setMinimum(0)
            spin.setMaximum(1000)
            spin.setSingleStep(1)
            # Pre-fill from existing mapping or default 10
            try:
                preset = int(self.input_fps_by_model.get(model, 10))
            except Exception:
                preset = 10
            spin.setValue(preset)

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
                try:
                    self.input_fps_by_model[model] = int(spin.value())
                except Exception:
                    self.input_fps_by_model[model] = 0
            # Log results
            pairs = ", ".join([f"{m}: {int(v)}" for m, v in sorted(self.input_fps_by_model.items())])
            self._log(f"[Info] Updated input rates: {pairs}")

    def select_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Models Folder', self.models_root)
        if folder:
            self.models_root = folder
            if hasattr(self, 'deployment_model_input'):
                self.deployment_model_input.setText(folder)
            root_index = self.fs_model.setRootPath(self.models_root)
            if isinstance(root_index, bool):
                root_index = self.fs_model.index(self.models_root)
            self.fs_model.set_root_index(root_index)
            self.model_tree_view.setRootIndex(root_index)
            self.model_tree_view.expand(root_index)

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
        """Return a list of absolute paths for checked top-level directories under models_root."""
        checked = []
        root_index = self.fs_model.index(self.models_root)
        rows = self.fs_model.rowCount(root_index)
        for r in range(rows):
            idx = self.fs_model.index(r, 0, root_index)
            path = self.fs_model.filePath(idx)
            state = self.fs_model.data(idx, Qt.CheckStateRole)
            if state == Qt.Checked:
                checked.append(path)
        return checked

    def log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        text = f'[{ts}] {msg}'
        if hasattr(self, 'log_text_edit') and self.log_text_edit is not None:
            self.log_text_edit.appendPlainText(text)
        print(text)

    def build_schedule_from_selection(self, models_root: str, checked_dirs, device_conf_path: str, out_path: str) -> str:
        """Generate a schedule YAML (model_schedules.yaml) from selected top-level model folders and device config.
        Returns the output path. Mirrors the approach from backup.schedule_generator_app.generate_all_combinations.
        """
        import yaml
        # Derive model names from checked directories (top-level under models_root)
        models = [os.path.basename(d) for d in checked_dirs if os.path.isdir(d)]
        models = [m for m in models if m]
        if not models:
            raise ValueError("No models selected. Please check at least one top-level model folder.")
        if len(models) > 4:
            self.log(f"[Warn] More than 4 models selected. Using only the first 4.")
            models = models[:4]

        # Load device info
        try:
            import yaml as _yaml
            with open(device_conf_path, 'r') as f:
                device_config = _yaml.safe_load(f) or {}
            cpu_count = device_config.get("devices", {}).get("cpu", {}).get("count", 1)
            npu_cfg = device_config.get("devices", {}).get("npu", {})
            npu_count = npu_cfg.get("count", 0)
            npu_ids = list(npu_cfg.get("ids", []))
            self.log(f"[Info] Device config: CPU={cpu_count}, NPU={npu_count}, NPU IDs={npu_ids}")
        except Exception as e:
            raise RuntimeError(f"Failed to load device config '{device_conf_path}': {e}")

        # Generate all combinations (CPU multi-assign allowed; NPU unique per model)
        combinations = []

        def rec(idx, assign, available_npus: set):
            if idx >= len(models):
                combinations.append(assign.copy())
                return
            model = models[idx]
            # Option 1: CPU
            assign[model] = "cpu"
            rec(idx + 1, assign, available_npus)
            # Option 2: each available NPU
            for nid in list(available_npus):
                assign[model] = f"npu{nid}"
                new_avail = set(available_npus)
                new_avail.remove(nid)
                rec(idx + 1, assign, new_avail)

        rec(0, {}, set(npu_ids))

        # Build schedules dict
        schedules = {}
        for i, combo in enumerate(combinations):
            combo_name = f"combination_{i+1}"
            schedules[combo_name] = {}
            for j, (model, device) in enumerate(combo.items()):
                model_id = f"{model}_{device}"
                # Determine default infps
                infps = None
                # Prefer explicit input rates if provided by the dialog
                if isinstance(getattr(self, 'input_fps_by_model', None), dict):
                    v = self.input_fps_by_model.get(model)
                    try:
                        if v is not None:
                            infps = int(v)
                    except Exception:
                        infps = None
                # Fallback heuristics if not provided
                if infps is None:
                    lname = model.lower()
                    if "resnet50" in lname:
                        infps = 2
                    elif "yolov3" in lname:
                        infps = 30
                    else:
                        infps = 10
                entry = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}",
                }
                if infps is not None:
                    entry["infps"] = int(infps)
                schedules[combo_name][model_id] = entry
        # Write YAML
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("# model_schedules.yaml\n")
                f.write("# Auto-generated\n\n")
                f.write(yaml.dump(schedules, default_flow_style=False))
        except Exception as e:
            raise RuntimeError(f"Failed to write schedule YAML '{out_path}': {e}")
        return out_path

    def generate_all_combinations(self) -> str:
        """Generate all possible model-to-device combinations into model_schedules.yaml using:
        - Checked top-level model folders in the tree.
        - Device config path from self.device_config_input.
        - Per-model input rates from input_rate_dialog (self.input_fps_by_model).
        Returns the output YAML path.
        """
        # Resolve selections
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        checked_dirs = self.get_checked_top_level_dirs()
        if not checked_dirs:
            raise ValueError("No model folders selected. Please check model folders in the tree.")
        device_conf = self.device_config_input.text() if hasattr(self, 'device_config_input') else ''
        if not device_conf or not os.path.exists(device_conf):
            raise FileNotFoundError(f"Device config not found: {device_conf}")
        out_path = self.generated_schedule_path
        # Delegate to existing builder (kept for compatibility)
        return self.build_schedule_from_selection(models_root, checked_dirs, device_conf, out_path)

    def predict_best_combination(self, schedule_yaml_path: str, model_input_path: str, alpha: float = 0.2):
        """Predict best combination using two-target XGBoost JSON models.
        - model_input_path can be either:
          - A directory containing two files: <prefix>_y1.json and <prefix>_y2.json
          - One of the two JSON files (we'll infer the prefix and the counterpart)
        Returns (best_combination_name, df) where df contains columns:
          [source, combination, pred_total_throughput_fps, pred_drop_rate_fps, pred_score].
        """
        import pandas as pd
        from pathlib import Path
        from xgboost_model.deploy_selector_xgb_suite import (
            load_static_profiles,
            featurize_from_combo,
            predict_two_targets,
            _load_yaml_or_json,
            _iter_combos_from_schedule,
        )

        def _infer_model_prefix(p: Path) -> Path:
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

        # Use project root sample_profiling_data.json as required
        static_json_path = Path(__file__).resolve().parent / "sample_profiling_data.json"
        if not static_json_path.exists():
            raise FileNotFoundError(f"Static profiling JSON not found: {static_json_path}")

        S = load_static_profiles(static_json_path)
        schedule_doc = _load_yaml_or_json(sched_path)
        combos = _iter_combos_from_schedule(schedule_doc)
        if not combos:
            raise ValueError("No combinations found in schedule YAML.")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(S, combo_blob)
            y1_pred, y2_pred = predict_two_targets(model_prefix, X)
            fps = float(y1_pred[0]); drop = float(y2_pred[0])
            score = fps - float(alpha) * drop
            rows.append({
                "source": sched_path.name,
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

    def _build_cpu_only_schedule(self, checked_dirs, out_path: str) -> str:
        """Build a schedule with a single combination where all selected models run on CPU
        using the per-model input rates previously set by the user (input_rate_dialog).
        Returns the output YAML path.
        """
        import yaml
        # Derive model names
        models = [os.path.basename(d) for d in checked_dirs if os.path.isdir(d)]
        models = [m for m in models if m]
        if not models:
            raise ValueError("No models selected. Please check model folders in the tree.")
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
                        infps = int(v)
                except Exception:
                    infps = None
            # Reasonable defaults if not provided
            if infps is None:
                lname = model.lower()
                if "resnet50" in lname:
                    infps = 2
                elif "yolov3" in lname:
                    infps = 30
                else:
                    infps = 10
            entry = {
                "model": model,
                "execution": "cpu",
                "display": f"view{j+1}",
                "infps": int(infps),
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
            checked_dirs = self.get_checked_top_level_dirs()
            if not checked_dirs:
                self.log("[Error] No model folders selected. Please check model folders in the tree.")
                return
            try:
                schedule_path = self._build_cpu_only_schedule(checked_dirs, schedule_path)
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

        # Collect selected (checked) top-level model folders
        checked_dirs = self.get_checked_top_level_dirs()

        # Log inputs
        self.log(f"[Predict] models_root={models_root}")
        self.log(f"[Predict] prediction_model={pred_model}")
        self.log(f"[Predict] device_config={device_conf}")
        self.log(f"[Predict] checked_top_level_dirs={checked_dirs}")

        # Validate
        try:
            if not checked_dirs:
                raise ValueError("No model folders selected. Please check model folders in the tree.")
            if not device_conf or not os.path.exists(device_conf):
                raise FileNotFoundError(f"Device config not found: {device_conf}")
            if not pred_model or not os.path.exists(pred_model):
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