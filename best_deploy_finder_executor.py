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
import yaml
from pathlib import Path
from typing import Tuple, Optional
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QLabel, QSpinBox, QWidget, QHBoxLayout, QGridLayout, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem, QLineEdit

from schedule_generator.file_manager import FileManager
from schedule_executor_main import ScheduleExecutor, InfoWindow
from unified_viewer import UnifiedViewer

# Local modules
from gui_utils import ChangeDeployDialog, CheckableFileSystemModel
from schedule_generator_logic import ScheduleGenerator
from deploy_predictor_logic import DeployPredictor






class BestDeployFinderApp(QMainWindow):
    def __init__(self, models_root=None):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'best_deploy_finder_executor.ui'), self)

        # Initialize business logic modules
        self.schedule_generator = ScheduleGenerator(log_callback=self.log)
        self.deploy_predictor = DeployPredictor(log_callback=self.log)

        # Default models root to ./models
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'models_onnx')

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
        if hasattr(self, 'stop_best_button'):
            self.stop_best_button.clicked.connect(self.on_stop_execution_clicked)
        if hasattr(self, 'default_input_rate_button'):
            self.default_input_rate_button.clicked.connect(self.select_default_input_file)

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
        self._input_rate_dialog = None

        # Prediction/execution state tracking
        self._viewer = None
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(1000)  # 1 second
        self._metrics_timer.timeout.connect(self._update_live_metrics)

        self._prev_selected_models = set()      # set[str]
        self._prev_score = None                 # float | None
        self._prev_best_combo = None            # str | None
        self._current_selected_models = set()   # set[str]
        self._current_best_combo = None         # str | None
        self._current_score = None              # float | None
        self._executor_proc = None              # subprocess.Popen | None (Legacy)
        self._executor = None                   # ScheduleExecutor | None
        self._running_combo_name = None         # str | None - currently executing combination name
        self._executor_info_window = None       # InfoWindow | None

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

    def _load_default_input_rates(self):
        """Load default input rates from lineEdit_default_input_file if specified."""
        if hasattr(self, 'lineEdit_default_input_file'):
            default_file = self.lineEdit_default_input_file.text().strip()
            if default_file and os.path.isfile(default_file):
                try:
                    fm = FileManager(log_callback=self._log)
                    sample_data = fm.load_sample_data(filename=default_file)
                    if sample_data and 'total_data' in sample_data:
                        count = 0
                        for item in sample_data['total_data']:
                            model_name = item.get('model', '')
                            # Normalize model name: it might be a path, take basename/stem
                            if '/' in model_name or '\\' in model_name:
                                model_name = os.path.basename(model_name)
                            if model_name.lower().endswith('.onnx'):
                                model_name = os.path.splitext(model_name)[0]
                            
                            # Determine FPS: prefer gpu_fps or npu1_fps, then cpu_fps, then derive from ms
                            fps = item.get('gpu_fps') or item.get('npu1_fps') or item.get('cpu_fps')
                            if not fps or fps <= 0:
                                ms = item.get('gpu_infer') or item.get('npu1_infer') or item.get('cpu_infer')
                                if ms and ms > 0:
                                    fps = 1000.0 / ms
                            
                            if fps and fps > 0:
                                # Update mapping only if NOT already present to preserve user modifications
                                if model_name not in self.input_fps_by_model:
                                    self.input_fps_by_model[model_name] = int(fps)
                                    count += 1
                        if count > 0:
                            self._log(f"[Info] Loaded {count} default input rates from {default_file}")
                except Exception as e:
                    self._log(f"[Error] Failed to load default input rates: {e}")

    def on_input_rate_clicked(self):
        if self._input_rate_dialog is not None and self._input_rate_dialog.isVisible():
            self._input_rate_dialog.raise_()
            self._input_rate_dialog.activateWindow()
            return

        models = self._get_selected_model_names()
        if not models:
            self._log("[Warning] No models selected. Please select folders in the model tree.")
            return

        # Load default input rates from file if specified
        self._load_default_input_rates()

        # Load the dialog UI
        dialog_ui_path = os.path.join(os.path.dirname(__file__), 'input_rate_dialog.ui')
        dlg = QDialog(self)
        uic.loadUi(dialog_ui_path, dlg)
        self._input_rate_dialog = dlg
        dlg.setModal(False) # Modeless

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
            spin.setMaximum(2147483647) # Max 32-bit int, allows 10 digits (up to 2,147,483,647)
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

        # Add multiplier UI below the list
        from PyQt5.QtWidgets import QLineEdit, QDoubleSpinBox
        multiplier_container = QWidget(dlg)
        multiplier_layout = QHBoxLayout(multiplier_container)
        multiplier_layout.setContentsMargins(0, 10, 0, 0)
        
        mult_label = QLabel("Multiplier:", multiplier_container)
        mult_spin = QDoubleSpinBox(multiplier_container)
        mult_spin.setRange(0.0, 1000.0)
        mult_spin.setValue(1.0)
        mult_spin.setSingleStep(0.1)
        mult_spin.setDecimals(2)
        
        mult_apply_btn = QPushButton("Multiply", multiplier_container)
        reset_btn = QPushButton("Reset", multiplier_container)
        
        multiplier_layout.addWidget(mult_label)
        multiplier_layout.addWidget(mult_spin)
        multiplier_layout.addWidget(mult_apply_btn)
        multiplier_layout.addWidget(reset_btn)
        multiplier_layout.addStretch()
        
        def apply_multiplier():
            factor = mult_spin.value()
            for model_name, spin in spin_boxes.items():
                new_val = int(spin.value() * factor)
                spin.setValue(min(new_val, 2147483647))
            self._log(f"[Info] Applied multiplier {factor} to all input rates.")

        def reset_to_defaults():
            # Force reload from file if specified
            if hasattr(self, 'lineEdit_default_input_file'):
                default_file = self.lineEdit_default_input_file.text().strip()
                if default_file and os.path.isfile(default_file):
                    try:
                        fm = FileManager(log_callback=self._log)
                        sample_data = fm.load_sample_data(filename=default_file)
                        if sample_data and 'total_data' in sample_data:
                            # Use a temporary mapping to store values from the file
                            file_fps = {}
                            for item in sample_data['total_data']:
                                model_name = item.get('model', '')
                                if '/' in model_name or '\\' in model_name:
                                    model_name = os.path.basename(model_name)
                                if model_name.lower().endswith('.onnx'):
                                    model_name = os.path.splitext(model_name)[0]
                                
                                fps = item.get('gpu_fps') or item.get('npu1_fps') or item.get('cpu_fps')
                                if not fps or fps <= 0:
                                    ms = item.get('gpu_infer') or item.get('npu1_infer') or item.get('cpu_infer')
                                    if ms and ms > 0:
                                        fps = 1000.0 / ms
                                
                                if fps and fps > 0:
                                    file_fps[model_name] = int(fps)
                            
                            # Update only the spin boxes for models present in the file
                            count = 0
                            for m_name, spin in spin_boxes.items():
                                if m_name in file_fps:
                                    spin.setValue(file_fps[m_name])
                                    count += 1
                                else:
                                    # If not in file, reset to default 10?
                                    spin.setValue(10)
                            self._log(f"[Info] Reset {count} input rates to defaults from {default_file}")
                    except Exception as e:
                        self._log(f"[Error] Failed to reset to default input rates: {e}")
                else:
                    # If no file, just reset to 10
                    for m_name, spin in spin_boxes.items():
                        spin.setValue(10)
                    self._log("[Info] No default input file found. Reset all input rates to 10.")

        mult_apply_btn.clicked.connect(apply_multiplier)
        reset_btn.clicked.connect(reset_to_defaults)
        
        # Insert multiplier UI before the button box
        button_box = dlg.findChild(QWidget, 'buttonBox')
        dlg.layout().insertWidget(dlg.layout().indexOf(button_box), multiplier_container)

        # Requirements: "좌측 하단에 Apply 버튼 만들어서 OK 버튼 누르지 않고 Apply만 눌어도 입력되어있는 input rate 들이 적용되게 해 줘. Apply 버튼은 OK 버튼과 다르게 창이 닫히지 않아야 돼."
        
        # We'll create a new layout for the bottom to place Apply on the left and ButtonBox on the right
        bottom_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        bottom_layout.addWidget(apply_btn)
        bottom_layout.addStretch()
        
        # Remove button_box from its current parent layout and add to our new bottom_layout
        dlg.layout().removeWidget(button_box)
        bottom_layout.addWidget(button_box)
        
        dlg.layout().addLayout(bottom_layout)
        
        def save_rates():
            for model, spin in spin_boxes.items():
                try:
                    self.input_fps_by_model[model] = int(spin.value())
                except Exception:
                    self.input_fps_by_model[model] = 0
            # Log results
            pairs = ", ".join([f"{m}: {int(v)}" for m, v in sorted(self.input_fps_by_model.items())])
            self._log(f"[Info] Updated input rates: {pairs}")

        # Connect Apply button
        apply_btn.clicked.connect(save_rates)
        
        # Connect OK button (accepted signal)
        def on_accepted():
            save_rates()
            dlg.accept() # Use dlg.accept() to close with Accepted status
            
        button_box.accepted.disconnect() # disconnect default accept()
        button_box.accepted.connect(on_accepted)
        button_box.rejected.connect(dlg.reject)

        # Attach spin boxes dict for retrieval on accept
        dlg._spin_boxes_by_model = spin_boxes

        # Resize dialog to fit its contents tightly (height varies with model count)
        try:
            dlg.setMinimumWidth(800)
            dlg.adjustSize()
        except Exception:
            pass

        dlg.show() # Show modeless

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

    def select_default_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Default Input File', os.getcwd(), 'All Files (*)')
        if path and hasattr(self, 'lineEdit_default_input_file'):
            self.lineEdit_default_input_file.setText(path)

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
        return self.schedule_generator.build_schedule_from_selection(
            checked_paths=checked_paths,
            device_conf_path=device_conf_path,
            out_path=out_path,
            input_fps_by_model=getattr(self, 'input_fps_by_model', None)
        )


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

    def predict_best_combination(self, schedule_yaml_path: str, model_input_path: str, alpha: float = 0.3):
        """Predict best combination using XGBoost models."""
        return self.deploy_predictor.predict_best_combination(
            schedule_yaml_path=schedule_yaml_path,
            model_input_path=model_input_path,
            alpha=alpha
        )

    def _build_cpu_only_schedule(self, checked_paths, out_path: str) -> str:
        """Build a schedule with a single combination where all selected models run on CPU."""
        return self.schedule_generator.build_cpu_only_schedule(
            checked_paths=checked_paths,
            out_path=out_path,
            input_fps_by_model=getattr(self, 'input_fps_by_model', None)
        )

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
        """Stop current execution by stopping the executor or viewer directly."""
        if hasattr(self, '_metrics_timer'):
            self._metrics_timer.stop()

        if hasattr(self, '_viewer') and self._viewer is not None:
            self.log("[Exec] Stopping current viewer...")
            try:
                self._viewer.stop_execution()
            except Exception as e:
                self.log(f"[Exec] Warning: failed to stop viewer: {e}")
            self._viewer = None

        if hasattr(self, '_executor') and self._executor is not None:
            self.log("[Exec] Stopping current executor...")
            try:
                self._executor.stop()
            except Exception as e:
                self.log(f"[Exec] Warning: failed to stop executor: {e}")
            self._executor = None
        
        if hasattr(self, '_executor_info_window') and self._executor_info_window is not None:
            try:
                self._executor_info_window.close()
            except Exception:
                pass
            self._executor_info_window = None

        # Reset UI metrics
        if hasattr(self, 'throughput_value'):
            self.throughput_value.setText("0.00 FPS")
        if hasattr(self, 'drop_value'):
            self.drop_value.setText("0.00%")

        # Legacy subprocess cleanup (just in case)
        proc = getattr(self, '_executor_proc', None)
        if proc is not None:
            self.log("[Exec] Terminating legacy executor subprocess...")
            try:
                if proc.poll() is None:
                    import platform as _platform
                    import signal as _signal
                    import os as _os
                    if _platform.system() != 'Windows':
                        try:
                            _os.killpg(proc.pid, _signal.SIGINT)
                        except Exception:
                            pass
                    proc.terminate()
                    proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._executor_proc = None
            
        self._running_combo_name = None

    def _launch_executor_direct(self, schedule_path: str, combo_name: str = None, duration: int = None):
        """Launch UnifiedViewer direct (controller-less mode)."""
        self.log(f"[Exec] Launching viewer direct for: {combo_name or 'All'}")
        
        try:
            # Initialize UnifiedViewer directly
            duration = duration or 60
            viewer = UnifiedViewer(
                schedule_file=schedule_path,
                combination_name=combo_name,
                info_window=None,
                hide_info_window=True
            )
            
            # Show viewer
            viewer.show()
            self._viewer = viewer
            self._running_combo_name = combo_name
            
            # Start execution directly using parent.start_execution logic flow
            self.start_execution(duration)

            # Start metrics timer
            self._metrics_timer.start()

            return True
        except Exception as e:
            self.log(f"[Error] Failed to launch viewer: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False

    def start_execution(self, duration: int):
        """Start execution on the current viewer."""
        if self._viewer:
            self.log(f"[Exec] Starting execution for {duration} seconds...")
            self._viewer.start_execution(duration)

    def stop_execution(self):
        """Stop execution on the current viewer."""
        if self._viewer:
            self.log("[Exec] Stopping execution...")
            self._viewer.stop_execution()
            self._viewer = None

    def stop_execution_async(self):
        """Stop execution asynchronously on the current viewer."""
        if hasattr(self, '_metrics_timer'):
            self._metrics_timer.stop()
        if self._viewer:
            self.log("[Exec] Stopping execution (async)...")
            self._viewer.stop_execution_async()
            self._viewer = None

    def _update_live_metrics(self):
        """Update live throughput and drop rate metrics from the active viewer."""
        if not self._viewer:
            self._metrics_timer.stop()
            return

        try:
            # 1. Throughput calculation
            total_fps = 0.0
            scheduled_count = 0
            
            # Determine scheduled views
            views_without_model = getattr(self._viewer, 'views_without_model', set())
            scheduled_views = [v for v in ["view1", "view2", "view3", "view4"] if v not in views_without_model]
            
            for v_name in scheduled_views:
                handler = getattr(self._viewer, f"{v_name}_handler", None)
                if handler:
                    avg_fps = float(getattr(handler, 'avg_fps', 0.0) or 0.0)
                    total_fps += avg_fps
                    scheduled_count += 1
            
            avg_throughput = total_fps / scheduled_count if scheduled_count > 0 else 0.0
            
            # 2. Drop rate percentage calculation
            total_drops = 0
            total_inferences = 0
            
            # Check video_feeder (YOLO)
            v_feeder = getattr(self._viewer, 'video_feeder', None)
            if v_feeder:
                drop_counts = getattr(v_feeder, 'drop_counts', {})
                for count in drop_counts.values():
                    total_drops += count
            # Check resnet_feeder (ResNet)
            r_feeder = getattr(self._viewer, 'resnet_feeder', None)
            if r_feeder:
                drop_counts = getattr(r_feeder, 'drop_counts', {})
                for count in drop_counts.values():
                    total_drops += count

            # Calculate total inferences across all scheduled views
            for v_name in scheduled_views:
                handler = getattr(self._viewer, f"{v_name}_handler", None)
                if handler:
                    total_inferences += int(getattr(handler, 'infer_count', 0) or 0)
            
            # Drop rate = (drops / (inferences + drops)) * 100
            total_frames = total_inferences + total_drops
            drop_rate_pct = (total_drops / total_frames * 100.0) if total_frames > 0 else 0.0
            
            # Update UI
            if hasattr(self, 'throughput_value'):
                self.throughput_value.setText(f"{avg_throughput:.2f} FPS")
            if hasattr(self, 'drop_value'):
                self.drop_value.setText(f"{drop_rate_pct:.2f}%")
                
        except Exception as e:
            # Silently ignore errors during live update to avoid flickering/crashes
            pass

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
        # Ensure input rates are loaded for selected models if available
        self._load_default_input_rates()

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
                        duration = 30
                        # duration = 100000
                        self._launch_executor_direct(schedule_path, combo_name=self._current_best_combo, duration=duration)
                        return
                    # Case B: selection same -> if same best combination is already running, keep running; else compare scores
                    else:
                        # If currently running the same combination, keep running regardless of score difference
                        try:
                            running_name = getattr(self, '_running_combo_name', None)
                            is_viewer_alive = self._viewer is not None and self._viewer.isVisible()
                            is_executor_alive = self._executor is not None and getattr(self._executor, '_running', False)
                            is_alive = is_viewer_alive or is_executor_alive
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
                                    self._launch_executor_direct(schedule_path, combo_name=self._current_best_combo, duration=duration)
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
        self._launch_executor_direct(schedule_path, combo_name=best_combo, duration=duration)

    def on_predict_best_clicked(self):
        """Handler invoked when predict_best_button is clicked.
        Also manages internal state for previous/current selections and scores.
        """
        # Ensure input rates are loaded for selected models if available
        self._load_default_input_rates()

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
            for _name in ('throughput_value', 'drop_value'):
                if hasattr(self, _name):
                    try:
                        if _name == 'throughput_value':
                            getattr(self, _name).setText('0.00 FPS')
                        elif _name == 'drop_value':
                            getattr(self, _name).setText('0.00%')
                        else:
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
            for _name in ('throughput_value', 'drop_value'):
                if hasattr(self, _name):
                    try:
                        if _name == 'throughput_value':
                            getattr(self, _name).setText('0.00 FPS')
                        elif _name == 'drop_value':
                            getattr(self, _name).setText('0.00%')
                        else:
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
                            self.throughput_value.setText(f"{top_fps:.2f} FPS")
                        except Exception:
                            self.throughput_value.setText(f"{top_fps} FPS")
                    if hasattr(self, 'drop_value'):
                        try:
                            self.drop_value.setText(f"{top_drop:.3f}")
                        except Exception:
                            self.drop_value.setText(str(top_drop))
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