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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QDialog, QLabel, QSpinBox, QWidget, QHBoxLayout, QGridLayout, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem
import yaml
from pathlib import Path
from schedule_generator.file_manager import FileManager


class ChangeDeployDialog(QDialog):
    """Modeless dialog to show and select generated combinations from model_schedules.yaml."""

    def __init__(self, parent, schedule_path):
        super().__init__(parent)
        self.setWindowTitle("Change Deployment")
        self.setModal(False)  # Modeless
        self.schedule_path = schedule_path
        self.parent_app = parent

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.execute_button = QPushButton("Execute Selected")
        self.execute_button.clicked.connect(self.on_execute_clicked)
        layout.addWidget(self.execute_button)

        self.load_combinations()

    def load_combinations(self):
        self.list_widget.clear()
        if not os.path.exists(self.schedule_path):
            self.list_widget.addItem("No schedule file found.")
            self.execute_button.setEnabled(False)
            return

        try:
            with open(self.schedule_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                self.list_widget.addItem("Schedule file is empty.")
                self.execute_button.setEnabled(False)
                return

            combos = [key for key in data.keys() if key.startswith('combination_')]
            # Sort naturally if possible
            try:
                combos.sort(key=lambda x: int(x.split('_')[1]))
            except:
                combos.sort()

            for combo in combos:
                self.list_widget.addItem(combo)

            if not combos:
                self.list_widget.addItem("No combinations found in file.")
                self.execute_button.setEnabled(False)
        except Exception as e:
            self.list_widget.addItem(f"Error loading YAML: {e}")
            self.execute_button.setEnabled(False)

    def on_execute_clicked(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            combo_name = selected_item.text()
            if combo_name.startswith('combination_'):
                self.parent_app.log(f"[Action] Manually selected {combo_name} for execution.")
                self.parent_app._kill_existing_executor()
                self.parent_app._launch_executor_subprocess(self.schedule_path, combo_name=combo_name, duration=60)
                # self.close() # Keep it open as it's modeless, or close if user prefers. Requirement didn't specify.


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
        is_top = self.is_top_level_child(index)
        is_dir = self.isDir(index)
        is_onnx = self.filePath(index).lower().endswith('.onnx')

        if index.column() == 0 and is_top and (is_dir or is_onnx):
            return base | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return base

    def data(self, index, role=Qt.DisplayRole):
        is_top = self.is_top_level_child(index)
        is_dir = self.isDir(index)
        is_onnx = self.filePath(index).lower().endswith('.onnx')

        if role == Qt.CheckStateRole and index.column() == 0 and is_top and (is_dir or is_onnx):
            path = self.filePath(index)
            return self._check_states.get(path, Qt.Unchecked)
        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        is_top = self.is_top_level_child(index)
        is_dir = self.isDir(index)
        is_onnx = self.filePath(index).lower().endswith('.onnx')

        if role == Qt.CheckStateRole and index.column() == 0 and is_top and (is_dir or is_onnx):
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
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'deploy_models')

        # Default prediction model settings
        self.default_gpu_pred_model_dir = os.path.join(os.path.dirname(__file__), 'xgboost_model', 'artifacts', 'gpu')
        self.default_gpu_pred_model_prefix = 'xgb_model_x3_double'
        self.default_npu_pred_model_dir = os.path.join(os.path.dirname(__file__), 'xgboost_model', 'artifacts', 'npu')
        self.default_npu_pred_model_prefix = 'xgb_model_npu_double'

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
        if hasattr(self, 'change_deploy_button'):
            self.change_deploy_button.clicked.connect(self.on_change_deploy_clicked)

        if hasattr(self, 'device_type_combo'):
            self.device_type_combo.currentTextChanged.connect(self.on_device_type_changed)

        # Initialize line edits if present
        if hasattr(self, 'deployment_model_input'):
            self.deployment_model_input.setText(self.models_root)

        # Initial prediction model setup
        self.on_device_type_changed()

        # Initialize log window if present
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.setReadOnly(True)

        # State: input FPS mapping per model
        self.input_fps_by_model = {}

        # Prediction/execution state tracking
        self._prev_selected_models = set()      # set[str]
        self._prev_score = None                 # float | None
        self._prev_best_combo = None            # str | None
        self._current_selected_models = set()   # set[str]
        self._current_best_combo = None         # str | None
        self._current_score = None              # float | None
        self._executor_proc = None              # subprocess.Popen | None
        self._running_combo_name = None         # str | None - currently executing combination name

        # Status bar (labels) gating (legacy signature fields retained for logs/backward-compat; not primary gating now)
        self._last_displayed_best_sig = None    # tuple | None — stable signature of last displayed best combo
        self._last_displayed_best_name = None   # str | None — last displayed combo name (for logs only)

        # Default outputs
        self.generated_schedule_path = os.path.join(os.path.dirname(__file__), 'model_schedules.yaml')

    def _log(self, message: str):
        if hasattr(self, 'log_text_edit') and self.log_text_edit is not None:
            # QPlainTextEdit supports appendPlainText, not append
            self.log_text_edit.appendPlainText(message)
        else:
            print(message)

    def _get_selected_model_names(self):
        # Prefer checked top-level directories or .onnx files as the source of truth
        if hasattr(self.fs_model, 'get_checked_top_level_dirs'):
            checked = self.get_checked_top_level_dirs()
            if checked:
                # Map to folder basenames or .onnx file stems (model names)
                models = []
                for p in checked:
                    if os.path.isdir(p):
                        models.append(os.path.basename(p))
                    elif p.lower().endswith('.onnx'):
                        models.append(os.path.splitext(os.path.basename(p))[0])
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

    def on_device_type_changed(self):
        if not hasattr(self, 'device_type_combo'):
            return

        device_type = self.device_type_combo.currentText().lower() # 'gpu' or 'npus'
        if device_type == 'npus':
            base_dir = self.default_npu_pred_model_dir
            prefix = self.default_npu_pred_model_prefix
        else:
            base_dir = self.default_gpu_pred_model_dir
            prefix = self.default_gpu_pred_model_prefix

        if hasattr(self, 'prediction_model_input'):
            # Check if default files exist
            y1_path = os.path.join(base_dir, f"{prefix}_y1.json")
            y2_path = os.path.join(base_dir, f"{prefix}_y2.json")
            if os.path.exists(y1_path) and os.path.exists(y2_path):
                self.prediction_model_input.setText(base_dir)

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
        device_type = self.device_type_combo.currentText().lower() if hasattr(self, 'device_type_combo') else 'gpu'
        default_dir = self.default_npu_pred_model_dir if device_type == 'npus' else self.default_gpu_pred_model_dir

        start_dir = default_dir if os.path.exists(default_dir) else os.getcwd()
        path = QFileDialog.getExistingDirectory(self, 'Select Prediction Model Folder', start_dir)
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

    def build_schedule_from_selection(self, models_root: str, checked_paths, device_conf_path: str, out_path: str) -> str:
        """Generate a schedule YAML (model_schedules.yaml) from selected top-level model folders/.onnx files and device config.
        Always builds CPU/GPU combinations.
        """
        return self._build_schedule_cpu_gpu(models_root, checked_paths, device_conf_path, out_path)

    def _build_schedule_cpu_gpu(self, models_root: str, checked_paths, device_conf_path: str, out_path: str) -> str:
        """CPU/GPU mode: each model -> cpu or gpu (2^N combinations)."""
        import yaml
        models = []
        for p in checked_paths:
            if os.path.isdir(p):
                models.append(os.path.basename(p))
            elif p.lower().endswith('.onnx'):
                models.append(os.path.splitext(os.path.basename(p))[0])
        models = [m for m in models if m]

        if not models:
            raise ValueError("No models selected (folder or .onnx file).")
        if len(models) > 4:
            self.log(f"[Warn] More than 4 models selected. Using only the first 4.")
            models = models[:4]

        # Load device info
        try:
            with open(device_conf_path, 'r') as f:
                device_config = yaml.safe_load(f) or {}
            cpu_cfg = device_config.get("devices", {}).get("cpu", {})
            cpu_count = cpu_cfg.get("count", 1)
            gpu_cfg = device_config.get("devices", {}).get("gpu", {})
            gpu_count = gpu_cfg.get("count", 1)  # Default 1
            gpu_ids = list(gpu_cfg.get("ids", [0] if gpu_count > 0 else []))
            self.log(f"[Predict] Mode: CPU/GPU, Config: {device_conf_path}")
            self.log(f"[Predict] Detected: CPU count={cpu_count}, GPU count={gpu_count}, GPU IDs={gpu_ids}")
        except Exception as e:
            raise RuntimeError(f"Failed to load device config '{device_conf_path}': {e}")

        if gpu_count <= 0:
            self.log("[Predict] GPU is disabled by device config (count <= 0). Generating CPU-only schedules.")

        # Generate 2^N combinations
        combinations = []
        def rec(idx, assign):
            if idx >= len(models):
                combinations.append(assign.copy())
                return
            model = models[idx]
            # Option 1: CPU
            assign[model] = "cpu"
            rec(idx + 1, assign)
            # Option 2: GPU (only if enabled)
            if gpu_count > 0:
                assign[model] = "gpu"
                rec(idx + 1, assign)

        rec(0, {})
        self.log(f"[Predict] Generated {len(combinations)} combinations (CPU/GPU only)")
        return self._write_schedule_yaml(models, combinations, out_path)

    def _build_schedule_cpu_npus(self, models_root: str, checked_paths, device_conf_path: str, out_path: str) -> str:
        """Deprecated: NPUs mode is no longer used for schedule creation."""
        self.log("[Warn] _build_schedule_cpu_npus is deprecated. Using _build_schedule_cpu_gpu instead.")
        return self._build_schedule_cpu_gpu(models_root, checked_paths, device_conf_path, out_path)

    def _write_schedule_yaml(self, models, combinations, out_path: str) -> str:
        import yaml
        schedules = {}
        for i, combo in enumerate(combinations):
            combo_name = f"combination_{i+1}"
            schedules[combo_name] = {}
            for j, (model, device) in enumerate(combo.items()):
                model_id = f"{model}_{device}"
                infps = None
                if isinstance(getattr(self, 'input_fps_by_model', None), dict):
                    v = self.input_fps_by_model.get(model)
                    try:
                        if v is not None:
                            infps = int(v)
                    except Exception:
                        infps = None
                if infps is None:
                    lname = model.lower()
                    if "resnet50" in lname: infps = 2
                    elif "yolov3" in lname: infps = 30
                    else: infps = 10
                entry = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}",
                }
                if infps is not None:
                    entry["infps"] = int(infps)
                schedules[combo_name][model_id] = entry
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("# model_schedules.yaml\n# Auto-generated\n\n")
                f.write(yaml.dump(schedules, default_flow_style=False))
        except Exception as e:
            raise RuntimeError(f"Failed to write schedule YAML '{out_path}': {e}")
        return out_path

    def generate_all_combinations(self) -> str:
        """Generate all possible model-to-device combinations into model_schedules.yaml using:
        - Checked top-level model folders/.onnx files in the tree.
        - Device config path from self.device_config_input.
        - Per-model input rates from input_rate_dialog (self.input_fps_by_model).
        Returns the output YAML path.
        """
        # Resolve selections
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        checked_paths = self.get_checked_top_level_dirs()
        if not checked_paths:
            raise ValueError("No models selected (folder or .onnx file). Please check model items in the tree.")
        device_conf = self.device_config_input.text() if hasattr(self, 'device_config_input') else ''
        if not device_conf or not os.path.exists(device_conf):
            raise FileNotFoundError(f"Device config not found: {device_conf}")
        out_path = self.generated_schedule_path
        # Delegate to existing builder (kept for compatibility)
        return self.build_schedule_from_selection(models_root, checked_paths, device_conf, out_path)

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

        schedule_doc = _load_yaml_or_json(sched_path)
        combos = _iter_combos_from_schedule(schedule_doc)
        if not combos:
            raise ValueError("No combinations found in schedule YAML.")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(combo_blob)
            y1_pred, y2_pred = predict_two_targets(model_prefix, X)
            fps = float(y1_pred[0]); drop = float(y2_pred[0])
            score = fps - float(alpha) * drop
            self.log(f"[Predict] Combination: {name} -> FPS: {fps:.2f}, Drop: {drop:.2f}, Score: {score:.2f}")
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

    def _build_cpu_only_schedule(self, checked_paths, out_path: str) -> str:
        """Build a schedule with a single combination where all selected models run on CPU
        using the per-model input rates previously set by the user (input_rate_dialog).
        Returns the output YAML path.
        """
        import yaml
        # Derive model names
        models = []
        for p in checked_paths:
            if os.path.isdir(p):
                models.append(os.path.basename(p))
            elif p.lower().endswith('.onnx'):
                models.append(os.path.splitext(os.path.basename(p))[0])
        models = [m for m in models if m]

        if not models:
            raise ValueError("No models selected. Please check model folders/.onnx files in the tree.")
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

    def _compute_combo_signature(self, schedule_path: str, combo_name: str):
        """Compute a stable signature for a combination from the schedule YAML so we can
        detect real changes regardless of combo naming or ordering.
        Signature contains a sorted tuple of entries (model, execution, infps, display).
        Returns None on error.
        """
        try:
            import yaml as _yaml
            with open(schedule_path, 'r', encoding='utf-8') as f:
                doc = _yaml.safe_load(f) or {}
            combo = doc.get(str(combo_name)) if isinstance(doc, dict) else None
            if not isinstance(combo, dict):
                return None
            items = []
            for _key, entry in combo.items():
                if not isinstance(entry, dict):
                    continue
                model = entry.get('model')
                execu = entry.get('execution')
                infps = entry.get('infps', None)
                display = entry.get('display', None)
                try:
                    if infps is not None:
                        infps = int(infps)
                except Exception:
                    pass
                items.append((str(model), str(execu), infps, str(display) if display is not None else None))
            sig = tuple(sorted(items))
            return sig
        except Exception as e:
            try:
                self.log(f"[Warn] Failed to compute signature for {combo_name}: {e}")
            except Exception:
                pass
            return None

    def _kill_existing_executor(self):
        """Terminate previously launched executor subprocess if it's still running.
        Try graceful group signal first (mimics pressing Stop), then escalate.
        """
        import os as _os
        import signal as _signal
        import platform as _platform
        import subprocess as _subprocess
        proc = getattr(self, '_executor_proc', None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self.log("[Exec] Terminating previous executor window (gracefully)...")
                try:
                    if _platform.system() == 'Windows':
                        # Try CTRL_BREAK to the process group if possible
                        try:
                            proc.send_signal(_signal.CTRL_BREAK_EVENT)
                        except Exception:
                            proc.terminate()
                    else:
                        # POSIX: signal the whole process group that we created
                        try:
                            _os.killpg(proc.pid, _signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                except Exception:
                    pass
                # Wait a bit for clean shutdown
                try:
                    proc.wait(timeout=6)
                except Exception:
                    # Escalate to force-kill the group first, then the process
                    self.log("[Exec] Forcing kill of previous executor window...")
                    try:
                        if _platform.system() != 'Windows':
                            try:
                                _os.killpg(proc.pid, _signal.SIGKILL)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        proc.kill()
                    except Exception:
                        pass
        finally:
            self._executor_proc = None
            self._running_combo_name = None

    def _launch_executor_subprocess(self, schedule_path: str, combo_name: str = None, duration: int = None):
        """Launch schedule_executor_main.py in a separate process to avoid nested QApps.
        If combo_name is provided, run executor-only mode for that single combination.
        Returns the subprocess handle and stores it as self._executor_proc.
        """
        import subprocess
        import platform
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
            popen_kwargs = {}
            if platform.system() == 'Windows':
                try:
                    popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                except Exception:
                    pass
            else:
                # POSIX: start a new session so we can signal the whole process group
                popen_kwargs['start_new_session'] = True
            proc = subprocess.Popen(args, **popen_kwargs)
            self._executor_proc = proc
            # Track which combination is running (if provided)
            try:
                self._running_combo_name = str(combo_name) if combo_name else None
            except Exception:
                self._running_combo_name = combo_name
            return proc
        except Exception as e:
            self.log(f"[Error] Failed to launch executor: {e}")
            return None

    def on_change_deploy_clicked(self):
        """Show Change Deployment Dialog (modeless)."""
        if not hasattr(self, '_change_deploy_dialog') or self._change_deploy_dialog is None:
            self._change_deploy_dialog = ChangeDeployDialog(self, self.generated_schedule_path)
        else:
            self._change_deploy_dialog.load_combinations()

        self._change_deploy_dialog.show()
        self._change_deploy_dialog.raise_()
        self._change_deploy_dialog.activateWindow()

    def on_stop_execution_clicked(self):
        """Handler for the Stop Execution button.
        Mimics pressing Stop in the info window by terminating the running executor subprocess.
        """
        try:
            self.log("[Action] Stop requested. Terminating running executor...")
        except Exception:
            pass
        self._kill_existing_executor()
        try:
            self.log("[Action] Execution stopped.")
        except Exception:
            pass

    def on_load_execute_best_clicked(self):
        """Load best predicted deployment and start execution.
        If no predictions exist, run all selected models on CPU at user-specified input rates.
        Updated behavior per requirement:
        1) When pressed, if a current prediction exists for the current selection,
           - If the newly selected model set differs from the previous set: stop previous and run the new best.
           - If the newly selected model set is the same as the previous set: compare scores.
             If the new score is >= 15% higher than the previous score, stop previous and run new best;
             otherwise, keep the current executor running. Always log the score difference.
        2) If no current prediction context exists, fall back to predictions.csv or CPU-only schedule.
        """
        try:
            has_current = bool(getattr(self, '_current_best_combo', None))
            cur_models = getattr(self, '_current_selected_models', set()) or set()
            prev_models = getattr(self, '_prev_selected_models', set()) or set()
            cur_score = getattr(self, '_current_score', None)
            prev_score = getattr(self, '_prev_score', None)
            schedule_path = self.generated_schedule_path

            if has_current and cur_models:
                if not os.path.exists(schedule_path):
                    self.log(f"[Warn] Expected schedule not found for current prediction: {schedule_path}. Falling back.")
                else:
                    # Case A: selection changed -> always restart with new best
                    if cur_models != prev_models:
                        self.log("[Load] Current selection differs from previous. Running newly predicted best combination.")
                        self._kill_existing_executor()
                        duration = 60
                        self._launch_executor_subprocess(schedule_path, combo_name=self._current_best_combo, duration=duration)
                        return
                    # Case B: selection same -> if same best combination is already running, keep running; else compare scores
                    else:
                        # If currently running the same combination, keep running regardless of score difference
                        try:
                            proc = getattr(self, '_executor_proc', None)
                            running_name = getattr(self, '_running_combo_name', None)
                            is_alive = (proc is not None and proc.poll() is None)
                            if is_alive and running_name and str(running_name) == str(self._current_best_combo):
                                self.log(f"[Decision] Same selection and same combination '{running_name}' is already running. Keeping current executor.")
                                return
                        except Exception:
                            pass
                        # Log score difference if available
                        try:
                            if (cur_score is not None) and (prev_score is not None):
                                diff = float(cur_score) - float(prev_score)
                                prev = float(prev_score)
                                curr = float(cur_score)
                                diff = curr - prev
                                pct = (diff / prev * 100.0) if prev != 0.0 else float('inf')
                                self.log(f"[Decision] Same selection. prev_score={prev:.6f}, new_score={curr:.6f}, diff={diff:.6f} ({pct:.2f}%)")
                                should_restart = (prev == 0.0 and curr > 0.0) or (prev != 0.0 and curr >= 1.15 * prev)
                                if should_restart:
                                    self.log("[Decision] New score is >= 15% higher (or previous was 0 and new > 0). Restarting with new best combination.")
                                    self._kill_existing_executor()
                                    duration = 60
                                    self._launch_executor_subprocess(schedule_path, combo_name=self._current_best_combo, duration=duration)
                                    return
                                else:
                                    self.log("[Decision] Improvement < 15%. Keeping current executor running.")
                                    return
                            else:
                                self.log("[Decision] Same selection but insufficient score history to compare. Keeping current executor running.")
                                return
                        except Exception:
                            # On any logging/format issue, do not disrupt execution choice; keep current running.
                            self.log("[Decision] Error computing score difference. Keeping current executor running.")
                            return
        except Exception:
            # Continue to fallback path below on any unexpected issue
            pass

        # Fallback to legacy behavior using predictions.csv or CPU-only schedule
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
        # If no predictions.csv best, create a CPU-only schedule for the selected models
        if not best_combo:
            checked_paths = self.get_checked_top_level_dirs()
            if not checked_paths:
                self.log("[Error] No models selected (folder or .onnx file). Please check model items in the tree.")
                return
            try:
                schedule_path = self._build_cpu_only_schedule(checked_paths, schedule_path)
                best_combo = 'combination_1'
                self.log(f"[Build] Created CPU-only schedule: {schedule_path}")
            except Exception as e:
                self.log(f"[Error] Failed to build CPU-only schedule: {e}")
                return
        # Ensure schedule exists
        if not os.path.exists(schedule_path):
            self.log(f"[Error] Schedule file not found: {schedule_path}")
            return
        # Terminate existing executor before launching a new one
        self._kill_existing_executor()
        # Optional: pick duration from UI if available later; for now, default to 60
        duration = 60
        # Launch executor in a subprocess with selected combo
        self._launch_executor_subprocess(schedule_path, combo_name=best_combo, duration=duration)

    def on_predict_best_clicked(self):
        """Handler invoked when predict_best_button is clicked.
        Also manages internal state for previous/current selections and scores.
        """
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        pred_model = self.prediction_model_input.text() if hasattr(self, 'prediction_model_input') else ''
        device_conf = self.device_config_input.text() if hasattr(self, 'device_config_input') else ''

        # Fallback for prediction model if path is empty or does not exist
        if not pred_model or not os.path.exists(pred_model):
            # Try default path based on current device type selection
            device_type = self.device_type_combo.currentText().lower() if hasattr(self, 'device_type_combo') else 'gpu'
            if device_type == 'npus':
                base_dir = self.default_npu_pred_model_dir
                prefix = self.default_npu_pred_model_prefix
            else:
                base_dir = self.default_gpu_pred_model_dir
                prefix = self.default_gpu_pred_model_prefix

            if os.path.exists(base_dir):
                y1_path = os.path.join(base_dir, f"{prefix}_y1.json")
                y2_path = os.path.join(base_dir, f"{prefix}_y2.json")
                if os.path.exists(y1_path) and os.path.exists(y2_path):
                    pred_model = base_dir
                    self.log(f"[Predict] Using default fallback prediction model ({device_type.upper()}): {pred_model}")
                    if hasattr(self, 'prediction_model_input'):
                        self.prediction_model_input.setText(pred_model)

        # Final check of the used pred_model path
        self.log(f"[Predict] final model_input_path: {pred_model}")

        # Device Config path used for both generation and logging
        self.log(f"[Predict] Device Type: {self.device_type_combo.currentText() if hasattr(self, 'device_type_combo') else 'GPU'}")
        self.log(f"[Predict] Device Config Path: {device_conf}")

        # Collect selected (checked) top-level model folders/.onnx files
        checked_paths = self.get_checked_top_level_dirs()

        # Shift current->previous state before computing new prediction
        try:
            self._prev_selected_models = set(getattr(self, '_current_selected_models', set()) or set())
            self._prev_score = getattr(self, '_current_score', None)
            self._prev_best_combo = getattr(self, '_current_best_combo', None)
        except Exception:
            self._prev_selected_models = set()
            self._prev_score = None
            self._prev_best_combo = None

        # Log inputs
        self.log(f"[Predict] models_root={models_root}")
        self.log(f"[Predict] checked_top_level_items={checked_paths}")

        # Validate
        try:
            if not checked_paths:
                raise ValueError("No models selected (folder or .onnx file). Please check model items in the tree.")
            if not device_conf or not os.path.exists(device_conf):
                raise FileNotFoundError(f"Device config not found: {device_conf}")
            if not pred_model or not os.path.exists(pred_model):
                raise FileNotFoundError(f"Prediction model not found: {pred_model}")
        except Exception as e:
            self.log(f"[Error] {e}")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            for _name in ('throughput_value', 'drop_value', 'score_value'):
                if hasattr(self, _name):
                    try:
                        getattr(self, _name).setText('-')
                    except Exception:
                        pass
            return

        # Step 1: Generate schedule YAML (using generate_all_combinations)
        try:
            schedule_path = self.generate_all_combinations()
            self.log(f"[Step1] Generated schedule YAML: {schedule_path}")
        except Exception as e:
            self.log(f"[Error][Step1] {e}")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            for _name in ('throughput_value', 'drop_value', 'score_value'):
                if hasattr(self, _name):
                    try:
                        getattr(self, _name).setText('-')
                    except Exception:
                        pass
            return

        # Step 2: Run prediction using XGBoost model
        try:
            best_combo, df = self.predict_best_combination(schedule_path, pred_model)
            if not best_combo:
                raise RuntimeError("Prediction produced no result.")
            # Decide whether to update status bar according to the new rule:
            # Do NOT update if (selection unchanged) AND ((new_score <= 0.85 * prev_score) OR (new_best_name == prev_best_name)).
            try:
                new_score = float(df.iloc[0]['pred_score']) if len(df) > 0 else None
            except Exception:
                new_score = None
            # same_selection 판정: 체크된 경로들의 basename/stem 목록이 이전과 같은지 확인
            current_model_names = []
            for p in checked_paths:
                if os.path.isdir(p):
                    current_model_names.append(os.path.basename(p))
                elif p.lower().endswith('.onnx'):
                    current_model_names.append(os.path.splitext(os.path.basename(p))[0])

            same_selection = (set(sorted(current_model_names)) == (getattr(self, '_prev_selected_models', set()) or set()))
            prev_score = getattr(self, '_prev_score', None)
            prev_best = getattr(self, '_prev_best_combo', None)
            new_best = str(best_combo)

            skip_update = False
            reason_msgs = []
            if same_selection:
                if (prev_score is not None) and (new_score is not None) and (new_score <= 0.85 * float(prev_score)):
                    skip_update = True
                    reason_msgs.append(f"new_score={new_score:.6f} <= 0.85 * prev_score={float(prev_score):.6f}")
                if prev_best is not None and new_best == str(prev_best):
                    skip_update = True
                    reason_msgs.append(f"new_best_name == prev_best_name == '{new_best}'")

            if skip_update:
                self.log(f"[UI] Status bar NOT updated (same selection AND { ' OR '.join(reason_msgs) }).")
            else:
                # Update labels
                import re
                m = re.search(r"(\d+)$", best_combo)
                combo_number = m.group(1) if m else best_combo
                if hasattr(self, 'label_best_deploy_value'):
                    self.label_best_deploy_value.setText(str(combo_number))
                # Update the predicted metrics to the status labels
                try:
                    top_fps = float(df.iloc[0]['pred_total_throughput_fps'])
                    top_drop = float(df.iloc[0]['pred_drop_rate_fps'])
                    top_score = float(df.iloc[0]['pred_score'])
                    if hasattr(self, 'throughput_value'):
                        try:
                            self.throughput_value.setText(f"{top_fps:.3f}")
                        except Exception:
                            self.throughput_value.setText(str(top_fps))
                    if hasattr(self, 'drop_value'):
                        try:
                            self.drop_value.setText(f"{top_drop:.3f}")
                        except Exception:
                            self.drop_value.setText(str(top_drop))
                    if hasattr(self, 'score_value'):
                        try:
                            self.score_value.setText(f"{top_score:.3f}")
                        except Exception:
                            self.score_value.setText(str(top_score))
                except Exception:
                    pass
                # Also record legacy signature fields for possible future diagnostics (optional)
                try:
                    sig = self._compute_combo_signature(schedule_path, best_combo)
                    self._last_displayed_best_sig = sig
                except Exception:
                    pass
                try:
                    self._last_displayed_best_name = str(best_combo)
                except Exception:
                    self._last_displayed_best_name = best_combo
                self.log(f"[UI] Status bar updated for best combination: {best_combo}")
            # Log top predictions summary
            self.log(f"[Step2] Top-1 combination: {best_combo}")
            try:
                topn = min(5, len(df))
                self.log("[Step2] Top predictions:")
                for i in range(topn):
                    self.log(f"  {i+1}. {df.iloc[i]['combination']} -> {float(df.iloc[i]['pred_score']):.4f}")
            except Exception:
                pass
            # Update internal current-state tracking
            try:
                # derive current selected model names from checked_paths
                current_models = []
                for p in checked_paths:
                    if os.path.isdir(p):
                        current_models.append(os.path.basename(p))
                    elif p.lower().endswith('.onnx'):
                        current_models.append(os.path.splitext(os.path.basename(p))[0])
                self._current_selected_models = set(sorted(current_models))
                self._current_best_combo = str(best_combo)
                self._current_score = float(df.iloc[0]['pred_score']) if len(df) > 0 else None
                self.log(f"[State] Updated current selection: {sorted(list(self._current_selected_models))}")
                self.log(f"[State] Current best: {self._current_best_combo} (score={self._current_score})")
                if getattr(self, '_prev_selected_models', None) is not None:
                    self.log(f"[State] Previous selection: {sorted(list(self._prev_selected_models))} (score={self._prev_score})")
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