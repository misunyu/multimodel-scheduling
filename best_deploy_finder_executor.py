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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QDialog, QLabel, QSpinBox, QWidget, QHBoxLayout, QGridLayout, QAbstractItemView, QMessageBox
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
            # Enforce maximum of 4 checked models at top level
            if value == Qt.Checked:
                # Count currently checked top-level dirs
                checked_count = sum(1 for state in self._check_states.values() if state == Qt.Checked)
                if checked_count >= 4:
                    # Show warning dialog and revert the last attempted check
                    parent = self.parent() if isinstance(self.parent(), QWidget) else None
                    QMessageBox.warning(
                        parent,
                        '선택 제한',
                        '최대 4개의 모델만 선택할 수 있습니다.\n추가로 선택한 항목은 해제됩니다.'
                    )
                    # Ensure UI reflects the unchecked state
                    self._check_states[path] = Qt.Unchecked
                    self.dataChanged.emit(index, index, [Qt.CheckStateRole])
                    return False
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

        # Default models root to ./models_onnx
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'models_onnx')

        # Setup file system model and tree view
        self.fs_model = CheckableFileSystemModel(self)
        self.fs_model.setRootPath(self.models_root)
        root_index = self.fs_model.index(self.models_root)
        self.fs_model.set_root_index(root_index)

        # Hook the model to the tree view defined in the UI
        self.model_tree_view.setModel(self.fs_model)
        self.model_tree_view.setRootIndex(root_index)
        # Allow multi-selection in the tree view
        try:
            self.model_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
        except Exception:
            pass
        # Show only name column
        for col in range(1, self.fs_model.columnCount()):
            self.model_tree_view.setColumnHidden(col, True)
        # Expand one level for visibility
        self.model_tree_view.expand(root_index)

        # Selection limiting state
        self._suppress_selection_handler = False
        self._last_selected_top_keys = set()  # distinct top-level model keys currently allowed
        try:
            sel_model = self.model_tree_view.selectionModel()
            if sel_model is not None:
                sel_model.selectionChanged.connect(self._on_tree_selection_changed)
        except Exception:
            pass

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
        if hasattr(self, 'stop_execution'):
            self.stop_execution.clicked.connect(self.on_stop_execution_clicked)

        # Initialize line edits if present
        if hasattr(self, 'deployment_model_input'):
            self.deployment_model_input.setText(self.models_root)

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

    # ------------------------------
    # Selection limiting helpers (max 4)
    # ------------------------------
    def _path_to_top_key(self, path: str):
        """Map a filesystem path (file or dir) to a top-level 'model key' under models_root.
        Rules:
        - Top-level file *.onnx -> key = file stem
        - Top-level folder containing model.onnx -> key = folder name
        - Nested selections under such a folder count toward that folder's key
        Otherwise return None.
        """
        try:
            root = self.models_root
            if not path:
                return None
            # Normalize
            path = os.path.abspath(path)
            root = os.path.abspath(root)

            # If it's a file
            if os.path.isfile(path):
                dirp = os.path.dirname(path)
                base = os.path.basename(path)
                # Direct child .onnx
                if dirp == root and base.lower().endswith('.onnx'):
                    return os.path.splitext(base)[0]
                # model.onnx inside a direct child folder
                if base.lower() == 'model.onnx' and os.path.dirname(dirp) == root:
                    return os.path.basename(dirp)
                # Other files: map to their direct child folder if applicable
                if os.path.dirname(dirp) == root:
                    return os.path.basename(dirp)
                return None

            # If it's a directory
            if os.path.isdir(path):
                parent = os.path.dirname(path)
                if parent == root:
                    # Count only folders that are direct children; prefer those with model.onnx
                    if os.path.isfile(os.path.join(path, 'model.onnx')):
                        return os.path.basename(path)
                    # If no model.onnx, still treat as a bucket for selection counting
                    return os.path.basename(path)
                # If nested, attribute to its top-level parent folder under root
                while parent and parent != '/' and parent != root:
                    path, parent = parent, os.path.dirname(parent)
                if parent == root:
                    return os.path.basename(path)
            return None
        except Exception:
            return None

    def _collect_selected_top_keys(self):
        keys = []
        try:
            for idx in self.model_tree_view.selectedIndexes():
                if idx.column() != 0:
                    continue
                p = self.fs_model.filePath(idx)
                k = self._path_to_top_key(p)
                if k:
                    keys.append(k)
        except Exception:
            return set()
        return set(keys)

    def _deselect_key(self, key: str):
        try:
            sel_model = self.model_tree_view.selectionModel()
            if sel_model is None:
                return
            # Deselect all selected indexes that map to this key
            for idx in list(self.model_tree_view.selectedIndexes()):
                if idx.column() != 0:
                    continue
                p = self.fs_model.filePath(idx)
                if self._path_to_top_key(p) == key:
                    try:
                        sel_model.select(idx, sel_model.Deselect)
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_tree_selection_changed(self, selected, deselected):
        if self._suppress_selection_handler:
            return
        try:
            # Determine keys before applying enforcement
            current_keys = self._collect_selected_top_keys()
            if len(current_keys) <= 4:
                self._last_selected_top_keys = set(current_keys)
                return

            # Compute which keys were newly added
            added_keys = set()
            try:
                for idx in selected.indexes():
                    if idx.column() != 0:
                        continue
                    p = self.fs_model.filePath(idx)
                    k = self._path_to_top_key(p)
                    if k and k not in self._last_selected_top_keys:
                        added_keys.add(k)
            except Exception:
                added_keys = set()

            # Temporarily suppress recursive handling
            self._suppress_selection_handler = True
            try:
                warning_needed = False
                # First, try to drop newly added keys until we are within limit
                for k in list(added_keys):
                    if len(current_keys) <= 4:
                        break
                    self._deselect_key(k)
                    if k in current_keys:
                        current_keys.remove(k)
                    warning_needed = True

                # If still over limit (e.g., programmatic selection without 'selected' info), trim extras
                if len(current_keys) > 4:
                    # Preserve previously allowed keys as much as possible
                    keep = list(self._last_selected_top_keys)
                    # Fill up to 4 with any remaining current keys
                    for k in sorted(current_keys):
                        if len(keep) >= 4:
                            break
                        if k not in keep:
                            keep.append(k)
                    # Deselect all not in keep
                    for k in list(current_keys):
                        if k not in keep:
                            self._deselect_key(k)
                            warning_needed = True
                    current_keys = set(keep)

                if warning_needed:
                    try:
                        QMessageBox.warning(self, '선택 제한', '최대 4개의 모델만 선택할 수 있습니다.\n추가로 선택한 항목은 해제됩니다.')
                    except Exception:
                        pass
            finally:
                self._suppress_selection_handler = False

            self._last_selected_top_keys = set(current_keys)
        except Exception:
            # On any error, do not block user selection
            self._suppress_selection_handler = False

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

    def _enumerate_models_under_root(self):
        """Enumerate runnable models under self.models_root for the Input Rate dialog.
        Rules:
        - If there is a top-level file with .onnx extension, include it with model name = file stem.
        - If there is a top-level directory and it contains a file named 'model.onnx', include it with model name = folder name.
        - If both a file and a folder would yield the same model name, prefer the folder/model.onnx entry.
        Returns an ordered list of model names (sorted) and an internal mapping name->path kept on self for potential future use.
        """
        root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        try:
            entries = os.listdir(root)
        except Exception as e:
            self._log(f"[Error] Failed to list models root '{root}': {e}")
            return []

        name_to_path = {}
        # First, collect files (*.onnx)
        for name in entries:
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith('.onnx'):
                model_name = os.path.splitext(name)[0]
                if model_name and model_name not in name_to_path:
                    name_to_path[model_name] = path
        # Then, collect directories containing model.onnx (override if name clashes)
        for name in entries:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                candidate = os.path.join(path, 'model.onnx')
                if os.path.isfile(candidate):
                    model_name = name
                    # Prefer folder/model.onnx over top-level file with same name
                    name_to_path[model_name] = candidate

        # Store for potential later usage; dialog only needs names for now
        try:
            self._input_rate_model_paths = dict(name_to_path)
        except Exception:
            self._input_rate_model_paths = name_to_path

        model_names = sorted(name_to_path.keys())
        self._log(f"[Info] Detected {len(model_names)} models under root for Input Rate: {', '.join(model_names) if model_names else '(none)'}")
        return model_names

    def on_input_rate_clicked(self):
        # New requirement: target files under ./models_onnx and directories containing model.onnx.
        # Model naming: file .onnx -> file stem; directory/model.onnx -> directory name.
        models = self._enumerate_models_under_root()
        if not models:
            self._log("[Warning] 모델 루트 폴더에서 실행 대상 모델을 찾지 못했습니다. (.onnx 파일 또는 <folder>/model.onnx)")
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
            # Reset selection limiting state and reconnect handler
            try:
                self._last_selected_top_keys = set()
                sel_model = self.model_tree_view.selectionModel()
                if sel_model is not None:
                    # Avoid duplicate connections by disconnecting if already connected
                    try:
                        sel_model.selectionChanged.disconnect(self._on_tree_selection_changed)
                    except Exception:
                        pass
                    sel_model.selectionChanged.connect(self._on_tree_selection_changed)
            except Exception:
                pass

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
        # Note: We no longer cap the number of selected models to 4.
        # The display layout still supports up to 4 views; models beyond 4 will have display='none'.

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
            # Determine display mapping based on the 'use_display_view' checkbox.
            use_display_checked = False
            try:
                use_display_checked = bool(self.use_display_view.isChecked())
            except Exception:
                use_display_checked = False
            preferred = ["mnasnet", "resnet50", "resnext50", "yolov4"]
            models_in_combo = list(combo.keys())
            display_map = {m: "none" for m in models_in_combo}
            if use_display_checked:
                v = 1
                for name in preferred:
                    for m in models_in_combo:
                        if m.lower() == name and v <= 4:
                            display_map[m] = f"view{v}"
                            v += 1
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
                    "display": display_map.get(model, "none"),
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
        """Generate model schedules based on ONNX files directly under ./models_onnx and
        folders that contain a model.onnx (e.g., tiny-llama-chat-onnx/model.onnx).
        Selection rule:
        - Use the Input Rate dialog's values (self.input_fps_by_model). Models with infps > 0
          are considered selected. Model names are:
            * <file_stem> for top-level *.onnx files
            * <folder_name> for <folder>/model.onnx
        Output:
        - Save into the test_schedules folder.
        - Filename includes the number of selected models and each model's input rate.
        Returns the output YAML path.
        """
        import re
        import yaml

        # Resolve models root and device config
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        device_conf = self.device_config_input.text() if hasattr(self, 'device_config_input') else ''
        if not device_conf or not os.path.exists(device_conf):
            raise FileNotFoundError(f"Device config not found: {device_conf}")

        # Enumerate available models under root (top-level *.onnx and */model.onnx)
        available_models = self._enumerate_models_under_root()  # returns model names
        if not available_models:
            raise ValueError("No ONNX models found directly under the models root. (*.onnx or <folder>/model.onnx)")

        # Determine selected models with the following priority:
        # 1) Highlighted items in the tree view (files or folders). This allows selecting top-level *.onnx files.
        # 2) Checked top-level directories (legacy checkbox behavior).
        # 3) Input Rate dialog selections (infps > 0). If none, default all detected models to rate=10.
        ui_selected: list = []
        try:
            # Collect highlighted selections from the tree (column 0 only)
            selected_indices = [idx for idx in self.model_tree_view.selectedIndexes() if idx.column() == 0]
            sel_models = set()
            for idx in selected_indices:
                p = self.fs_model.filePath(idx)
                try:
                    if os.path.isfile(p):
                        # If a top-level .onnx file was selected, use its stem as model name
                        if p.lower().endswith('.onnx') and os.path.dirname(p) == self.models_root:
                            sel_models.add(os.path.splitext(os.path.basename(p))[0])
                        # If 'model.onnx' inside a top-level folder was selected, use the folder name
                        elif os.path.basename(p).lower() == 'model.onnx' and os.path.dirname(os.path.dirname(p)) == self.models_root:
                            sel_models.add(os.path.basename(os.path.dirname(p)))
                    elif os.path.isdir(p):
                        # If a top-level folder containing model.onnx was selected, use folder name
                        if os.path.dirname(p) == self.models_root and os.path.isfile(os.path.join(p, 'model.onnx')):
                            sel_models.add(os.path.basename(p))
                except Exception:
                    continue
            ui_selected = sorted([m for m in sel_models if m in available_models])
        except Exception:
            ui_selected = []

        checked_models: list = []
        try:
            checked_dirs = self.get_checked_top_level_dirs()
            checked_models = sorted([os.path.basename(p) for p in checked_dirs if os.path.isdir(p)])
            # Only keep those that are valid available models (folder must contain model.onnx)
            checked_models = [m for m in checked_models if m in available_models]
        except Exception:
            checked_models = []

        # Compose final selection list
        if ui_selected:
            selected_models = ui_selected
            try:
                self.log(f"[Info] Using highlighted selections: {', '.join(selected_models)}")
            except Exception:
                pass
        elif checked_models:
            selected_models = checked_models
            try:
                self.log(f"[Info] Using checked models: {', '.join(selected_models)}")
            except Exception:
                pass
        else:
            # Fall back to Input Rate dialog mapping
            selected_models = []
        
        # Build per-model rates for the chosen models
        rates = {}
        if selected_models:
            for m in selected_models:
                v = None
                if isinstance(getattr(self, 'input_fps_by_model', None), dict):
                    try:
                        v = int(self.input_fps_by_model.get(m, 0))
                    except Exception:
                        v = 0
                if not v or v <= 0:
                    v = 10
                rates[m] = int(v)
        else:
            # Determine selected models via input rates (> 0). If the dialog was never used
            # or no positive rates were set, default all detected models to rate=10.
            if isinstance(getattr(self, 'input_fps_by_model', None), dict):
                for m in available_models:
                    try:
                        v = int(self.input_fps_by_model.get(m, 0))
                    except Exception:
                        v = 0
                    if v > 0:
                        rates[m] = v
            if not rates:
                # Fallback: use default rate 10 for all models
                rates = {m: 10 for m in available_models}
                try:
                    self.log("[Info] Input Rate dialog not used or no positive rates provided. Using default infps=10 for all models.")
                except Exception:
                    pass
            selected_models = sorted(rates.keys())

        # Do not limit the number of models; viewer layout maps at most 4 models to views, others use display='none'.
        selected_models = sorted(selected_models)
        if len(selected_models) > 4:
            try:
                self.log(f"[Info] {len(selected_models)} models selected; display will show up to 4, others set to 'none'.")
            except Exception:
                pass
        # Persist the currently selected models for downstream logic (predict/load handlers)
        try:
            self._last_selected_models_from_rates = set(selected_models)
        except Exception:
            self._last_selected_models_from_rates = set(selected_models)

        # Load device info (optional): we only care if a GPU device section exists; we will use a single logical GPU.
        try:
            import yaml as _yaml
            with open(device_conf, 'r') as f:
                device_config = _yaml.safe_load(f) or {}
            # Try to read GPU availability for logging only
            gpu_cfg = device_config.get("devices", {}).get("gpu", {})
            gpu_count = gpu_cfg.get("count", 1)
            self.log(f"[Info] Device config loaded. Using single GPU (count reported={gpu_count})")
        except Exception as e:
            raise RuntimeError(f"Failed to load device config '{device_conf}': {e}")

        # Generate all combinations with CPU/GPU options only.
        # GPU has no concurrency limit (each model can choose GPU independently).
        combinations = []
        def rec(idx, assign):
            if idx >= len(selected_models):
                combinations.append(assign.copy())
                return
            model = selected_models[idx]
            # Option 1: CPU
            assign[model] = "cpu"
            rec(idx + 1, assign)
            # Option 2: GPU
            assign[model] = "gpu"
            rec(idx + 1, assign)
        rec(0, {})

        # Build schedules dict
        schedules = {}
        for i, combo in enumerate(combinations):
            combo_name = f"combination_{i+1}"
            schedules[combo_name] = {}
            # Determine display mapping based on the 'use_display_view' checkbox.
            use_display_checked = False
            try:
                use_display_checked = bool(self.use_display_view.isChecked())
            except Exception:
                use_display_checked = False
            preferred = ["mnasnet", "resnet50", "resnext50", "yolov4"]
            models_in_combo = list(combo.keys())
            display_map = {m: "none" for m in models_in_combo}
            if use_display_checked:
                v = 1
                for name in preferred:
                    for m in models_in_combo:
                        if m.lower() == name and v <= 4:
                            display_map[m] = f"view{v}"
                            v += 1
            for j, (model, device) in enumerate(combo.items()):
                infps = int(rates.get(model, 0)) if model in rates else 0
                if infps <= 0:
                    # Fallback defaults if somehow missing
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
                    "display": display_map.get(model, "none"),
                    "infps": int(infps),
                }
                schedules[combo_name][f"{model}_{device}"] = entry

        # Prepare output path under test_schedules with informative filename
        root_dir = os.path.dirname(__file__)
        out_dir = os.path.join(root_dir, 'test_schedules')
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        # Compose filename: m{N}_{abbrev-rate}_... .yaml (use only first two letters of model name; join with single underscore)
        def _sanitize(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]", "-", s)
        def _abbr(model: str) -> str:
            s = _sanitize(model)
            # keep only letters/digits for abbreviation base
            import re as _re
            alnum = ''.join(_re.findall(r"[A-Za-z0-9]", s))
            if len(alnum) >= 2:
                return alnum[:2].lower()
            return s[:2].lower() if len(s) >= 2 else s.lower()
        parts = [f"m{len(selected_models)}"] + [f"{_abbr(m)}-{int(rates[m])}" for m in selected_models]
        # Append display flag to filename when use_display_view is checked
        display_suffix = ""
        try:
            if bool(self.use_display_view.isChecked()):
                display_suffix = "_display"
        except Exception:
            display_suffix = ""
        filename = f"schedule_{'_'.join(parts)}{display_suffix}.yaml"
        out_path = os.path.join(out_dir, filename)

        # Write YAML
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("# model_schedules.yaml\n")
                f.write("# Auto-generated from top-level ONNX selection\n\n")
                f.write(yaml.dump(schedules, default_flow_style=False))
        except Exception as e:
            raise RuntimeError(f"Failed to write schedule YAML '{out_path}': {e}")

        # Update the default generated path for downstream steps
        try:
            self.generated_schedule_path = out_path
        except Exception:
            pass
        return out_path

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
        # No cap on model count; display mapping still limited to 4 views.
        schedules = {
            "combination_1": {}
        }
        # Determine display mapping according to use_display_view checkbox
        use_display_checked = False
        try:
            use_display_checked = bool(self.use_display_view.isChecked())
        except Exception:
            use_display_checked = False
        preferred = ["mnasnet", "resnet50", "resnext50", "yolov4"]
        display_map = {m: "none" for m in models}
        if use_display_checked:
            v = 1
            for name in preferred:
                for m in models:
                    if m.lower() == name and v <= 4:
                        display_map[m] = f"view{v}"
                        v += 1
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
                "display": display_map.get(model, "none"),
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

        # Collect selected (checked) top-level model folders
        checked_dirs = self.get_checked_top_level_dirs()

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
        self.log(f"[Predict] prediction_model={pred_model}")
        self.log(f"[Predict] device_config={device_conf}")
        self.log(f"[Predict] checked_top_level_dirs={checked_dirs}")

        # Validate (do not require checked folders; selection is driven by Input Rate dialog)
        try:
            if not device_conf or not os.path.exists(device_conf):
                raise FileNotFoundError(f"Device config not found: {device_conf}")
            if not pred_model or not os.path.exists(pred_model):
                raise FileNotFoundError(f"Prediction model not found: {pred_model}")
            if not checked_dirs:
                try:
                    sel_idx = [idx for idx in self.model_tree_view.selectedIndexes() if idx.column() == 0]
                except Exception:
                    sel_idx = []
                if sel_idx:
                    self.log("[Info] No checked folders; will use highlighted selections in the tree.")
                else:
                    self.log("[Info] No checked folders; will use Input Rate selections under models_onnx.")
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
            # Determine current selection set: prefer checked dirs; fallback to Input Rate selections
            try:
                from_checked = set(sorted([os.path.basename(p) for p in checked_dirs if os.path.isdir(p)]))
            except Exception:
                from_checked = set()
            rate_selected = getattr(self, '_last_selected_models_from_rates', set()) or set()
            current_sel_set = from_checked if from_checked else set(sorted(list(rate_selected)))
            same_selection = (current_sel_set == (getattr(self, '_prev_selected_models', set()) or set()))
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
                # derive current selected model names: prefer checked dirs; fallback to Input Rate selections
                from_checked = set(sorted([os.path.basename(p) for p in checked_dirs if os.path.isdir(p)]))
                rate_selected = getattr(self, '_last_selected_models_from_rates', set()) or set()
                current_models = from_checked if from_checked else set(sorted(list(rate_selected)))
                self._current_selected_models = current_models
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
    parser.add_argument('--models-root', type=str, help='Path to the models root folder (default: ./models_onnx)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = QApplication(sys.argv)
    window = BestDeployFinderApp(models_root=args.models_root)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()