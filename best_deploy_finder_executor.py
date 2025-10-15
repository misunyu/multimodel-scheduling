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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog


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
        # Wire up predict button
        if hasattr(self, 'predict_best_button'):
            self.predict_best_button.clicked.connect(self.on_predict_best_clicked)

        # Initialize line edits if present
        if hasattr(self, 'deployment_model_input'):
            self.deployment_model_input.setText(self.models_root)

        # Initialize log window if present
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.setReadOnly(True)

        # Default outputs
        self.generated_schedule_path = os.path.join(os.path.dirname(__file__), 'model_schedules.yaml')

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
        path, _ = QFileDialog.getOpenFileName(self, 'Select Prediction Model', os.getcwd(), 'All Files (*);;Model Files (*.model *.bin)')
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
                lname = model.lower()
                if "resnet50" in lname:
                    infps = 2
                elif "yolov3" in lname:
                    infps = 30
                schedules[combo_name][model_id] = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}",
                    **({"infps": infps} if infps is not None else {}),
                }
        # Write YAML
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("# model_schedules.yaml\n")
                f.write("# Auto-generated\n\n")
                f.write(yaml.dump(schedules, default_flow_style=False))
        except Exception as e:
            raise RuntimeError(f"Failed to write schedule YAML '{out_path}': {e}")
        return out_path

    def predict_best_combination(self, schedule_yaml_path: str, model_path: str):
        """Run prediction using xgboost_model/deploy_selector_xgb_suite logic. Returns (best_combination_name, df)."""
        import joblib
        import pandas as pd
        from xgboost_model.deploy_selector_xgb_suite import rows_from_schedule_yaml
        # Infer feats path from model path directory
        model_dir = os.path.dirname(model_path)
        feats_path = os.path.join(model_dir, 'xgb_deploy_selector.feats')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Prediction model not found: {model_path}")
        if not os.path.exists(feats_path):
            raise FileNotFoundError(f"Features file not found next to model: {feats_path}")

        model = joblib.load(model_path)
        dv = joblib.load(feats_path)
        rows = rows_from_schedule_yaml(schedule_yaml_path)
        if not rows:
            raise ValueError("No combinations found in schedule YAML.")
        X = dv.transform([r["features"] for r in rows])
        preds = model.predict(X)
        import numpy as np
        # Build DataFrame like the CLI tool
        df = pd.DataFrame({
            "source": os.path.basename(schedule_yaml_path),
            "combination": [r["combination"] for r in rows],
            "pred_score": preds,
        })
        df = df.sort_values(["pred_score"], ascending=[False]).reset_index(drop=True)
        # Save predictions.csv (overwrite)
        try:
            out_csv = os.path.join(os.path.dirname(__file__), 'predictions.csv')
            df.to_csv(out_csv, index=False)
            self.log(f"[Predict] Saved predictions to {out_csv}")
        except Exception as e:
            self.log(f"[Warn] Failed to save predictions.csv: {e}")
        best_combo = str(df.iloc[0]["combination"]) if len(df) > 0 else None
        return best_combo, df

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

        # Step 1: Generate schedule YAML
        try:
            schedule_path = self.generated_schedule_path
            schedule_path = self.build_schedule_from_selection(models_root, checked_dirs, device_conf, schedule_path)
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