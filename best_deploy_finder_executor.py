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
from PyQt5.QtCore import Qt, QProcess
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QBrush
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QLabel,
                             QDoubleSpinBox, QWidget, QHBoxLayout, QGridLayout, QMessageBox,
                             QAbstractItemView)


# The platform the user picks in the Device Configuration combo. Each choice ties
# together the XGBoost predictor trained on that platform and the device pair the
# placement enumeration draws from, so the two can never drift apart.
DEVICE_PROFILES = {
    "CPU-NPU": {"prefix": "xgboost_model/artifacts/deploy_cpu_npu",
                "devices": ["cpu", "npu"]},
    "CPU-GPU": {"prefix": "xgboost_model/artifacts/deploy_cpu_gpu",
                "devices": ["cpu", "gpu"]},
}
DEFAULT_DEVICE_PROFILE = "CPU-NPU"

# Demo input sources, passed to the executor through the environment. Each has a
# default that works out of the box: the demo must run on a bare button press.
DEMO_INPUT_DEFAULTS = {
    "DEMO_VIDEO": "stockholm_1280x720.mp4",        # detection + VLM frames
    "DEMO_IMAGE_DIR": "imagenet-sample-images",     # classification, cycled
    "DEMO_LLM_PROMPT": "",                          # "" -> the engine's own prompt list
    "DEMO_VLM_PROMPT": "",                          # "" -> the engine's default prompt
}

# Executors still running when the GUI closes are moved here so Python does not
# garbage-collect the QProcess (which would take the running child down with it).
# The old Popen-based launch let the executor outlive the GUI; keep that.
_DETACHED_EXECUTORS = []


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


class WorkingSetListModel(QStandardItemModel):
    """Single-selection list of the working sets the predictor was trained on.

    One row per set, read from `<prefix>_coverage.json` -- never hardcoded. Because
    the user can only pick a set the predictor actually saw, an out-of-distribution
    selection is unconstructible rather than merely warned about.

    The member models live on `Qt.UserRole` so downstream code reads them directly
    instead of parsing the display label. Rows whose models are not all deployable
    on this machine are disabled, with the blocking reason in the tooltip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Trained Working Set"])

    def populate(self, model_sets, blockers_for):
        """`model_sets`: list of lists of model names. `blockers_for(models)` ->
        {model: reason} for the members that cannot be deployed here."""
        self.removeRows(0, self.rowCount())
        for idx, models in enumerate(model_sets, start=1):
            names = sorted(models)
            item = QStandardItem(f"Set {idx} — {', '.join(names)}  ({len(names)} models)")
            item.setEditable(False)
            item.setCheckable(False)
            item.setData(list(names), Qt.UserRole)
            blockers = blockers_for(names)
            if blockers:
                item.setEnabled(False)
                item.setSelectable(False)
                item.setForeground(QBrush(Qt.gray))
                item.setToolTip("Not deployable here — "
                                + "; ".join(f"{m}: {r}" for m, r in sorted(blockers.items())))
            else:
                item.setToolTip("Trained working set: in-distribution for this predictor.")
            self.appendRow(item)

    def models_at(self, row):
        item = self.item(row)
        return list(item.data(Qt.UserRole) or []) if item is not None else []

    def enabled_rows(self):
        return [r for r in range(self.rowCount()) if self.item(r).isEnabled()]


class BestDeployFinderApp(QMainWindow):
    def __init__(self, models_root=None):
        super().__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'best_deploy_finder_executor.ui'), self)

        # Default models root to ./models
        self.models_root = models_root or os.path.join(os.path.dirname(__file__), 'models')

        # The tree view lists the predictor's TRAINED WORKING SETS, one per row, pick
        # exactly one. Populated by on_device_config_changed below (it depends on the
        # predictor prefix, which the Device Configuration combo decides).
        self.model_list = WorkingSetListModel(self)
        self.model_tree_view.setModel(self.model_list)
        self.model_tree_view.setRootIsDecorated(False)
        self.model_tree_view.setHeaderHidden(False)
        self.model_tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.model_tree_view.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Wire up browse buttons if present
        if hasattr(self, 'deploy_model_browse_button'):
            self.deploy_model_browse_button.clicked.connect(self.select_models_folder)
        if hasattr(self, 'prediction_model_browse_button'):
            self.prediction_model_browse_button.clicked.connect(self.select_prediction_model)
        # Wire up custom buttons
        if hasattr(self, 'input_rate_button'):
            self.input_rate_button.clicked.connect(self.on_input_rate_clicked)
        if hasattr(self, 'configure_inputs_button'):
            self.configure_inputs_button.clicked.connect(self.on_configure_inputs_clicked)
        if hasattr(self, 'predict_best_button'):
            self.predict_best_button.clicked.connect(self.on_predict_best_clicked)
        if hasattr(self, 'load_execute_best_button'):
            self.load_execute_best_button.clicked.connect(self.on_load_execute_best_clicked)
        if hasattr(self, 'run_worst_button'):
            self.run_worst_button.clicked.connect(self.on_run_worst_clicked)
        if hasattr(self, 'run_comparison_button'):
            self.run_comparison_button.clicked.connect(self.on_run_comparison_clicked)

        # Initialize line edits if present
        if hasattr(self, 'deployment_model_input'):
            self.deployment_model_input.setText(self.models_root)

        # Initialize log window if present
        if hasattr(self, 'log_text_edit'):
            self.log_text_edit.setReadOnly(True)

        # State: input FPS mapping per model. Set before the combo is wired, because
        # applying the default selection reads it.
        self.input_fps_by_model = {}

        # Demo inputs. Every field has a working default, so the demo runs end to end
        # even if the user never opens the Configure Inputs dialog.
        self.demo_inputs = dict(DEMO_INPUT_DEFAULTS)

        # Default outputs
        self.generated_schedule_path = os.path.join(os.path.dirname(__file__), 'model_schedules.yaml')

        # Device Configuration combo: picking a platform selects the predictor, the
        # device pair placements are enumerated over, AND the working sets offered
        # (each predictor has its own coverage file). Default to CPU-NPU.
        self.platform_devices = list(DEVICE_PROFILES[DEFAULT_DEVICE_PROFILE]["devices"])
        if hasattr(self, 'device_config_combo'):
            self.device_config_combo.clear()
            self.device_config_combo.addItems(list(DEVICE_PROFILES))
            self.device_config_combo.setCurrentText(DEFAULT_DEVICE_PROFILE)
            self.device_config_combo.currentTextChanged.connect(self.on_device_config_changed)
            self.on_device_config_changed(self.device_config_combo.currentText())

    def _log(self, message: str):
        if hasattr(self, 'log_text_edit') and self.log_text_edit is not None:
            # QPlainTextEdit supports appendPlainText, not append
            self.log_text_edit.appendPlainText(message)
        else:
            print(message)

    def _model_blockers(self):
        """{model: reason} for every model that cannot be deployed on this machine.

        Reuses the per-model discovery: vision models need a `.onnx` (to run on CPU)
        and, only when the platform includes the NPU, a `.mxq`; generative models need
        a static profile entry. A model absent from this map is deployable.
        """
        file_backed, cpu_only, npu_only = discover_file_backed_models(self.models_root)
        try:
            hf_backed = set(discover_hf_backed_models(self._resolve_static_json()))
        except Exception:
            hf_backed = set()
        needs_mxq = 'npu' in (getattr(self, 'platform_devices', None) or [])

        blockers = {}
        for m in npu_only:                      # has .mxq, no .onnx -> cannot run on CPU
            blockers[m] = "missing .onnx (needed for CPU)"
        if needs_mxq:
            for m in cpu_only:                  # has .onnx, no .mxq -> cannot run on NPU
                blockers[m] = "missing .mxq (needed for NPU)"
        self._deployable = set(file_backed) | hf_backed | (set() if needs_mxq else set(cpu_only))
        return blockers

    def _blockers_for(self, models):
        """{model: reason} restricted to `models` -- what stops this working set."""
        known = self._model_blockers()
        out = {}
        for m in models:
            if m in known:
                out[m] = known[m]
            elif m not in getattr(self, '_deployable', set()):
                # Not a file-backed model and not a profiled generative one.
                out[m] = "no static profile / no model file"
        return out

    def _reload_working_sets(self):
        """Repopulate the list from the CURRENT predictor's coverage file.

        The trained working sets are the single source of truth for what may be
        selected; they are read from `<prefix>_coverage.json` and never hardcoded.
        """
        pred = self.prediction_model_input.text() if hasattr(self, 'prediction_model_input') else ''
        try:
            sets = self.trained_working_sets(pred)
        except Exception as e:
            sets = []
            self._log(f"[Error] Could not read the predictor's training coverage: {e}")

        self.model_list.populate(sets, self._blockers_for)

        if not sets:
            self._log(f"[Error] No trained working sets found for '{pred}'. Expected model_sets "
                      f"in <prefix>_coverage.json. Nothing can be selected.")
            self._set_predict_enabled(False, "no trained working sets")
            return

        usable = self.model_list.enabled_rows()
        self._log(f"[Sets] {len(sets)} trained working set(s) from "
                  f"{Path(_infer_model_prefix(Path(pred))).name}_coverage.json; "
                  f"{len(usable)} deployable here.")
        for r in range(self.model_list.rowCount()):
            if not self.model_list.item(r).isEnabled():
                self._log(f"[Sets] Disabled — {self.model_list.item(r).toolTip()}")

        if not usable:
            self._set_predict_enabled(False, "no working set is fully deployable on this machine")
            return

        self._set_predict_enabled(True)
        # Default to the first selectable row, so there is never a "nothing selected" state.
        self.model_tree_view.setCurrentIndex(self.model_list.index(usable[0], 0))

    def _set_predict_enabled(self, enabled: bool, why: str = ""):
        if hasattr(self, 'predict_best_button'):
            self.predict_best_button.setEnabled(enabled)
        if not enabled:
            self._log(f"[Error] 'Predict Best Deployment' disabled: {why}.")
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText(f"n/a — {why}")

    def _get_selected_model_names(self):
        """The models of the selected working set (read off Qt.UserRole, not parsed)."""
        idx = self.model_tree_view.currentIndex()
        if not idx.isValid():
            return []
        models = self.model_list.models_at(idx.row())
        if models:
            self._log(f"[Info] Selected working set: {', '.join(models)}")
        return models

    def on_input_rate_clicked(self):
        models = self._get_selected_model_names()
        if not models:
            self._log("[Warning] No working set selected. Pick one in the list above.")
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

    def on_configure_inputs_clicked(self):
        """Pick what each model kind consumes in the demo.

        Vision gets a video file, classification an image folder, and the generative
        models a prompt each. Fields are pre-filled with the working defaults, so
        Cancel (or never opening this at all) still leaves a runnable demo.
        """
        from PyQt5.QtWidgets import QLineEdit, QPushButton, QDialogButtonBox, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("Configure Demo Inputs")
        grid = QGridLayout()
        fields = {}

        def _row(row, label, key, browse=None, tip=""):
            grid.addWidget(QLabel(label), row, 0)
            edit = QLineEdit(self.demo_inputs.get(key, ""))
            edit.setMinimumWidth(320)
            if tip:
                edit.setPlaceholderText(tip)
            grid.addWidget(edit, row, 1)
            fields[key] = edit
            if browse == "file":
                b = QPushButton("Browse..")
                b.clicked.connect(lambda: self._pick_into(edit, folder=False))
                grid.addWidget(b, row, 2)
            elif browse == "dir":
                b = QPushButton("Browse..")
                b.clicked.connect(lambda: self._pick_into(edit, folder=True))
                grid.addWidget(b, row, 2)

        _row(0, "Video (detection, VLM):", "DEMO_VIDEO", browse="file")
        _row(1, "Image folder (resnet50):", "DEMO_IMAGE_DIR", browse="dir")
        _row(2, "LLM prompt (llama1b):", "DEMO_LLM_PROMPT",
             tip="empty = built-in prompt rotation")
        _row(3, "VLM prompt (qwen2_vl):", "DEMO_VLM_PROMPT",
             tip="empty = 'Describe what is happening in one short sentence.'")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addLayout(grid)
        layout.addWidget(buttons)

        if dlg.exec_() == QDialog.Accepted:
            for key, edit in fields.items():
                self.demo_inputs[key] = edit.text().strip()
            for key, val in self.demo_inputs.items():
                self.log(f"[Inputs] {key} = {val or '(default)'}")
            self._warn_missing_inputs()

    def _pick_into(self, edit, folder: bool):
        path = (QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.dirname(__file__))
                if folder else
                QFileDialog.getOpenFileName(self, 'Select File', os.path.dirname(__file__),
                                            'Video (*.mp4 *.avi *.mov);;All Files (*)')[0])
        if path:
            edit.setText(path)

    def _warn_missing_inputs(self):
        """Log (do not block) inputs that point nowhere -- the view will say so too."""
        root = os.path.dirname(os.path.abspath(__file__))
        v = self.demo_inputs.get("DEMO_VIDEO") or ""
        d = self.demo_inputs.get("DEMO_IMAGE_DIR") or ""
        vp = v if os.path.isabs(v) else os.path.join(root, v)
        dp = d if os.path.isabs(d) else os.path.join(root, d)
        if v and not os.path.isfile(vp):
            self.log(f"[Warning] Video not found: {vp} — detection/VLM views will show 'No input'.")
        if d and not os.path.isdir(dp):
            self.log(f"[Warning] Image folder not found: {dp} — resnet50 view will show 'No input'.")

    def select_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Models Folder', self.models_root)
        if folder:
            self.models_root = folder
            if hasattr(self, 'deployment_model_input'):
                self.deployment_model_input.setText(folder)
            # Availability of each working set depends on what is in the models folder.
            self._reload_working_sets()

    def select_prediction_model(self):
        # Expect a folder that contains <prefix>_y1.json and <prefix>_y2.json
        path = QFileDialog.getExistingDirectory(self, 'Select Prediction Model Folder', os.getcwd())
        if path and hasattr(self, 'prediction_model_input'):
            self.prediction_model_input.setText(path)
            # A different predictor means a different coverage file, hence different sets.
            self._reload_working_sets()

    def on_device_config_changed(self, label: str):
        """Apply the platform picked in the Device Configuration combo.

        Points the prediction-model field at the predictor trained on that platform
        and records the device pair `build_schedule_from_selection` enumerates over.
        The field stays editable so a power user can still point at another predictor.
        """
        profile = DEVICE_PROFILES.get(label)
        if profile is None:
            self._log(f"[Warning] Unknown device configuration '{label}'; keeping the previous one.")
            return
        self.platform_devices = list(profile["devices"])
        prefix = os.path.join(os.path.dirname(__file__), *profile["prefix"].split("/"))
        if hasattr(self, 'prediction_model_input'):
            self.prediction_model_input.setText(prefix)
        self._log(f"[Device] {label}: devices={'/'.join(self.platform_devices)}, predictor={prefix}")
        # Each predictor has its own coverage file, so the offered working sets --
        # and which of them are deployable -- change with the platform.
        self._reload_working_sets()

    def get_checked_top_level_dirs(self):
        """Backwards-compatible alias: the tree now holds trained working sets."""
        return self._get_selected_model_names()

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

    def preflight_check(self, pred_model):
        """Everything the prediction needs, checked before we bother the user.

        Runs ahead of the coverage check so that a broken environment (no xgboost)
        or a missing artifact reports itself as such, instead of surfacing as an
        out-of-distribution warning followed by a stack trace.
        Returns a list of blocking errors; empty means good to go.
        """
        errors = []
        try:
            import xgboost  # noqa: F401
        except Exception as e:
            errors.append(f"xgboost is not installed in this environment ({e}). "
                          f"The placement predictor cannot run. Launch the GUI with the "
                          f"interpreter from runtime_env.sh ($PYTHON_BIN).")

        try:
            prefix = _infer_model_prefix(Path(pred_model))
        except Exception as e:
            errors.append(f"Prediction model prefix could not be resolved: {e}")
            return errors

        missing = [Path(str(prefix) + suffix).name
                   for suffix in ("_y1.json", "_y2.json", "_y3.json",
                                  "_features.json", "_coverage.json")
                   if not Path(str(prefix) + suffix).exists()]
        if missing:
            errors.append(f"Predictor artifacts missing next to {prefix.name}: "
                          f"{', '.join(missing)}.")
        return errors

    def _load_coverage(self, pred_model):
        """The predictor's `<prefix>_coverage.json` as a dict ({} when unreadable)."""
        import json
        try:
            prefix = _infer_model_prefix(Path(pred_model))
            cov_path = Path(str(prefix) + "_coverage.json")
            return json.loads(cov_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def trained_working_sets(self, pred_model):
        """The model sets this predictor was actually trained on, as lists of names."""
        cov = self._load_coverage(pred_model)
        return [s.split(",") for s in cov.get("model_sets", []) if s]

    def check_selection_coverage(self, model_names, pred_model):
        """Compare the checked models against the predictor's training coverage.

        `<prefix>_coverage.json` records the model sets, view counts and devices
        seen during training. A selection outside that distribution still yields
        a prediction, but the ranking is extrapolated and should not be trusted.
        Returns a list of human-readable warnings (empty when in-distribution).
        """
        import json
        prefix = _infer_model_prefix(Path(pred_model))
        cov_path = Path(str(prefix) + "_coverage.json")
        if not cov_path.exists():
            return [f"No training-coverage record next to the predictor "
                    f"({cov_path.name}); cannot verify the selection is in-distribution."]
        try:
            cov = json.loads(cov_path.read_text(encoding="utf-8"))
        except Exception as e:
            return [f"Could not read {cov_path.name}: {e}"]

        selected = sorted(model_names)
        warnings = []

        if ",".join(selected) in set(cov.get("model_sets", [])):
            return []  # exact training working set

        unseen = [m for m in selected if m not in set(cov.get("models", []))]
        if unseen:
            warnings.append(f"Never trained on: {', '.join(unseen)}. "
                            f"Trained models: {', '.join(cov.get('models', []))}.")

        view_counts = cov.get("view_counts") or []
        if view_counts and len(selected) not in view_counts:
            warnings.append(f"{len(selected)} models selected, but training only covered "
                            f"working sets of {', '.join(str(n) for n in view_counts)} models.")

        # Models present in every training working set but missing from the selection.
        sets = [set(s.split(",")) for s in cov.get("model_sets", []) if s]
        if sets:
            always = set.intersection(*sets)
            missing = sorted(always - set(selected))
            if missing:
                warnings.append(f"Every training working set included {', '.join(missing)}, "
                                f"which the selection omits; contention behaviour will differ.")
        return warnings

    def build_schedule_from_selection(self, model_names, out_path: str) -> str:
        """Generate placement candidates for the selected models on the chosen platform.

        The platform comes from the Device Configuration combo (CPU-NPU or CPU-GPU);
        both its devices are shareable, matching the correspondingly trained predictor.
        Every checked model is placed on one of the two devices; per-view infps
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

        platform_devices = list(getattr(self, 'platform_devices', None)
                                or DEVICE_PROFILES[DEFAULT_DEVICE_PROFILE]["devices"])
        self.log(f"[Info] Platform: {' + '.join(d.upper() for d in platform_devices)} "
                 f"(both shareable). Models: {', '.join(models)}")

        # Baseline rates for default infps (1x). Overridable per model via the dialog.
        static_json = self._resolve_static_json()
        baseline = {}
        try:
            baseline = {r['model']: r.get('baseline_rate')
                        for r in json.loads(Path(static_json).read_text()).get('total_data', [])}
        except Exception as e:
            self.log(f"[Warn] Could not load baseline rates: {e}")

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
        self.log(f"[Info] Wrote {len(schedules)} "
                 f"{'/'.join(d.upper() for d in platform_devices)} combinations to {out_path}")
        return out_path

    def generate_all_combinations(self) -> str:
        """Enumerate placements of the models checked in the tree view.

        The device pair comes from the Device Configuration combo. Per-model input
        rates come from the input-rate dialog, else the baseline rate.
        Returns the generated schedule YAML path.
        """
        models = self._get_selected_model_names()
        if not models:
            raise ValueError("No models selected. Check at least one model in the list.")
        return self.build_schedule_from_selection(models, self.generated_schedule_path)

    def _validate_feature_vector(self, X, model_prefix):
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
                f"the prediction would be meaningless. Profile them first (profile_models.py).")

        rows = []
        for name, combo_blob in combos:
            X = featurize_from_combo(S, combo_blob)
            self._validate_feature_vector(X, model_prefix)
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

    def _launch_executor_subprocess(self, schedule_path: str, combo_name: str = None,
                                    duration: int = None, on_finished=None):
        """Launch schedule_executor_main.py in a separate process to avoid nested QApps.
        If combo_name is provided, run executor-only mode for that single combination.

        `on_finished(exit_code, results_path)` is called when the run ends. The results
        path is scraped from the child's own "[Executor] Initialized results file:" line
        rather than guessed, so the comparison reads the metrics of the run it started
        and not some other run's file that happens to be newest on disk.

        Run under QProcess, not a bare Popen, for two reasons the old code got wrong:
        its output was never read, so a child that died on startup did so invisibly --
        no window, no error, nothing; and it inherited this GUI's cwd, while the
        executor resolves its assets (the .ui files, the video, models/) relative to
        the repo root, so launching the GUI from anywhere else killed the child
        instantly. Pin the working directory and stream the child's output into the log.
        """
        from collections import deque
        from PyQt5.QtCore import QProcessEnvironment
        root = os.path.dirname(os.path.abspath(__file__))

        # Launch via schedule_executor_main.sh, NOT sys.executable. sys.executable is
        # whatever interpreter started this GUI -- from an IDE that is typically some
        # unrelated venv with no Mobilint SDK and no transformers, and the child
        # inherited it, so every view worker died on import ("No module named
        # 'mblt_model_zoo'") and the views rendered nothing. The shell script sources
        # runtime_env.sh, which is the one place that knows the right interpreter.
        launcher = os.path.join(root, 'schedule_executor_main.sh')
        program, args = launcher, ['--schedule', schedule_path]
        if not os.path.exists(launcher):
            program = sys.executable or 'python'
            args = ['-u', os.path.join(root, 'schedule_executor_main.py'),
                    '--schedule', schedule_path]
            self.log(f"[Warning] {os.path.basename(launcher)} not found; falling back to "
                     f"{program}, which may lack the Mobilint SDK.")
        if duration is not None:
            try:
                d = int(duration)
                args += ['--duration', str(max(1, d))]
            except Exception:
                pass
        if combo_name:
            args += ['--schedule_name', combo_name]

        proc = QProcess(self)
        proc.setWorkingDirectory(root)
        proc.setProgram(program)
        proc.setArguments(args)
        # Unbuffered, so the child's output reaches the log as it happens instead of
        # sitting in a pipe buffer that is lost if it dies.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        # Demo input sources chosen in "Configure Inputs..". Empty means "use the
        # built-in default", so only set the ones the user actually filled in.
        for key, val in (getattr(self, 'demo_inputs', None) or {}).items():
            if val:
                env.insert(key, str(val))
        proc.setProcessEnvironment(env)
        stderr_tail = deque(maxlen=20)
        seen = {"results": None}

        def _drain_stdout():
            text = bytes(proc.readAllStandardOutput()).decode('utf-8', 'replace')
            for line in text.splitlines():
                if not line.strip():
                    continue
                if "Initialized results file:" in line:
                    seen["results"] = line.split("Initialized results file:", 1)[1].strip()
                self.log(f"[Exec] {line}")

        def _drain_stderr():
            text = bytes(proc.readAllStandardError()).decode('utf-8', 'replace')
            for line in text.splitlines():
                if line.strip():
                    stderr_tail.append(line)
                    self.log(f"[Exec][stderr] {line}")

        def _finished(code, status):
            _drain_stdout()
            _drain_stderr()
            if code != 0:
                self.log(f"[Error] executor exited with code {code}")
                for line in stderr_tail:
                    self.log(f"[Error]   {line}")
            else:
                self.log("[Exec] executor exited normally (code 0).")
            self._executor_procs.discard(proc)
            if on_finished is not None:
                on_finished(code, seen["results"])

        def _error(err):
            self.log(f"[Error] Failed to launch executor: {proc.errorString()} ({err})")
            self._executor_procs.discard(proc)
            if on_finished is not None:
                on_finished(-1, None)

        proc.readyReadStandardOutput.connect(_drain_stdout)
        proc.readyReadStandardError.connect(_drain_stderr)
        proc.finished.connect(_finished)
        proc.errorOccurred.connect(_error)

        # Hold a reference: a QProcess that gets garbage-collected is killed.
        if not hasattr(self, '_executor_procs'):
            self._executor_procs = set()
        self._executor_procs.add(proc)

        self.log(f"[Exec] Launching executor (cwd={root}): {program} {' '.join(args)}")
        proc.start()
        return proc

    def closeEvent(self, event):
        """Let a running executor outlive this window, instead of dying with it.

        The output callbacks are bound to widgets that are about to be destroyed, so
        disconnect them first: otherwise they fire on a deleted QProcess and the GUI
        goes down with a RuntimeError on the way out.
        """
        for proc in list(getattr(self, '_executor_procs', ())):
            for signal in (proc.readyReadStandardOutput, proc.readyReadStandardError,
                           proc.finished, proc.errorOccurred):
                try:
                    signal.disconnect()
                except TypeError:
                    pass  # nothing was connected to this one
            if proc.state() != QProcess.NotRunning:
                proc.setParent(None)
                _DETACHED_EXECUTORS.append(proc)
        super().closeEvent(event)

    # ---- Best-vs-worst comparison ------------------------------------------------

    def _read_predictions(self):
        """predictions.csv rows, already score-descending (the predictor wrote them so).

        Returns [] when Predict has not been run; callers must say so rather than
        silently doing nothing.
        """
        import csv
        path = os.path.join(os.path.dirname(__file__), 'predictions.csv')
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                rows = [r for r in csv.DictReader(f) if r.get('combination')]
        except Exception as e:
            self.log(f"[Error] Could not read predictions.csv: {e}")
            return []
        for r in rows:
            for k in ('pred_score', 'pred_norm_throughput', 'pred_deadline_miss_rate',
                      'pred_norm_tokens'):
                try:
                    r[k] = float(r.get(k))
                except Exception:
                    r[k] = float('nan')
        return rows

    def _combo_placement(self, combo_name):
        """{model: device} for a combination in the generated schedule YAML."""
        import yaml
        try:
            doc = yaml.safe_load(open(self.generated_schedule_path, encoding='utf-8'))
            return {v['model']: v['execution'] for v in (doc.get(combo_name) or {}).values()}
        except Exception as e:
            self.log(f"[Warn] Could not read placement for {combo_name}: {e}")
            return {}

    def _combo_is_runnable(self, combo_name):
        """A combination is runnable when every model sits on a device it allows.

        The enumeration already honours DEVICE_CONSTRAINTS, so this should always
        pass; it is here because the worst-scoring row is the one most likely to be
        odd, and running an impossible combination would fail mid-demo.
        """
        import model_registry as reg
        placement = self._combo_placement(combo_name)
        if not placement:
            return False, "no placement found in the schedule"
        for model, dev in placement.items():
            try:
                allowed = reg.allowed_devices(model)
            except Exception:
                continue
            if dev not in allowed:
                return False, f"{model} cannot run on {dev} (allowed: {'/'.join(allowed)})"
        return True, ""

    def _pick_best_and_worst(self):
        """(best_row, worst_row) or (None, reason).

        Worst is the lowest-scoring *runnable* combination, walking up from the bottom.
        """
        rows = self._read_predictions()
        if not rows:
            return None, ("No predictions found. Press 'Predict Best Deployment' first — "
                          "the comparison ranks the combinations it produced.")
        if len(rows) < 2:
            return None, ("Only one combination was predicted, so there is nothing to "
                          "compare it against.")

        best = rows[0]
        ok, why = self._combo_is_runnable(best['combination'])
        if not ok:
            return None, f"The best combination ({best['combination']}) is not runnable: {why}"

        worst = None
        for row in reversed(rows[1:]):
            ok, why = self._combo_is_runnable(row['combination'])
            if ok:
                worst = row
                break
            self.log(f"[Compare] Skipping {row['combination']} (lowest score but not "
                     f"runnable: {why}); trying the next one up.")
        if worst is None:
            return None, "No runnable combination to compare against the best one."

        if worst['combination'] == best['combination']:
            return None, "Best and worst are the same combination; nothing to compare."
        if abs(worst['pred_score'] - best['pred_score']) < 1e-9:
            return None, ("Every combination scored the same, so the predictor is not "
                          "expressing a preference; a comparison would show nothing.")
        return (best, worst), ""

    def _measured_from_results(self, results_path, combo_name):
        """The executor's own metrics for `combo_name`. No new measurement is invented."""
        import json
        if not results_path or not os.path.exists(results_path):
            self.log(f"[Warn] No results file for {combo_name} at {results_path}")
            return None
        try:
            blob = json.loads(open(results_path, encoding='utf-8').read())
        except Exception as e:
            self.log(f"[Warn] Could not read {results_path}: {e}")
            return None
        entries = [e for e in (blob if isinstance(blob, list) else [blob])
                   if e.get('combination') == combo_name]
        if not entries:
            self.log(f"[Warn] {os.path.basename(results_path)} has no entry for {combo_name}.")
            return None
        e = entries[-1]
        total = e.get('total') or {}
        return {
            'combination': combo_name,
            'throughput_fps': total.get('total_throughput_fps'),
            'tokens_per_s': total.get('total_tokens_per_s'),
            'deadline_miss_rate': total.get('deadline_miss_rate'),
            'window_sec': e.get('window_sec'),
            'views': {v: {'model': d.get('model'), 'execution': d.get('execution'),
                          'throughput_fps': d.get('throughput_fps'),
                          'tokens_per_s': d.get('tokens_per_s')}
                      for v, d in (e.get('models') or {}).items()},
        }

    def _run_duration(self):
        try:
            return max(1, int(self.duration_input.text()))
        except Exception:
            return 60

    def _set_run_buttons_enabled(self, enabled):
        for name in ('run_worst_button', 'run_comparison_button',
                     'load_execute_best_button', 'predict_best_button'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(enabled)

    def _set_run_status(self, text):
        if hasattr(self, 'run_status_label'):
            self.run_status_label.setText(text)

    def on_run_worst_clicked(self):
        picked, reason = self._pick_best_and_worst()
        if picked is None:
            self._compare_unavailable(reason)
            return
        _, worst = picked
        self.log(f"[Compare] Running WORST: {worst['combination']} "
                 f"(predicted score {worst['pred_score']:.4f})")
        self._set_run_status(f"Running: WORST ({worst['combination']})")
        self._launch_executor_subprocess(self.generated_schedule_path,
                                         combo_name=worst['combination'],
                                         duration=self._run_duration())

    def on_run_comparison_clicked(self):
        """Run best, then worst, then show them side by side.

        Sequential on purpose: the NPU time-slices between whatever is resident, so
        running both at once would have each stealing the other's throughput and both
        measurements would be junk.
        """
        picked, reason = self._pick_best_and_worst()
        if picked is None:
            self._compare_unavailable(reason)
            return
        best, worst = picked
        duration = self._run_duration()
        self._comparison = {'best_row': best, 'worst_row': worst,
                            'best': None, 'worst': None}

        self.log(f"[Compare] Sequential comparison, {duration}s each "
                 f"(~{2 * duration + 30}s total, plus model load).")
        self.log(f"[Compare] BEST  {best['combination']}: predicted score {best['pred_score']:.4f}, "
                 f"placement {self._placement_str(best['combination'])}")
        self.log(f"[Compare] WORST {worst['combination']}: predicted score {worst['pred_score']:.4f}, "
                 f"placement {self._placement_str(worst['combination'])}")

        self._set_run_buttons_enabled(False)
        self._set_run_status(f"Running: BEST ({best['combination']})  —  1 of 2")

        def after_best(code, results_path):
            self._comparison['best'] = self._measured_from_results(
                results_path, best['combination'])
            if self._comparison['best'] is None:
                self.log("[Compare] The best run produced no metrics; aborting the comparison.")
                self._set_run_status("Comparison aborted: the best run produced no metrics.")
                self._set_run_buttons_enabled(True)
                return
            self.log(f"[Compare] BEST measured: {self._fmt_measured(self._comparison['best'])}")
            self._set_run_status(f"Running: WORST ({worst['combination']})  —  2 of 2   "
                                 f"[BEST done: {self._fmt_measured(self._comparison['best'])}]")
            self._launch_executor_subprocess(self.generated_schedule_path,
                                             combo_name=worst['combination'],
                                             duration=duration, on_finished=after_worst)

        def after_worst(code, results_path):
            self._comparison['worst'] = self._measured_from_results(
                results_path, worst['combination'])
            self._set_run_buttons_enabled(True)
            if self._comparison['worst'] is None:
                self.log("[Compare] The worst run produced no metrics; nothing to compare.")
                self._set_run_status("Comparison incomplete: the worst run produced no metrics.")
                return
            self.log(f"[Compare] WORST measured: {self._fmt_measured(self._comparison['worst'])}")
            self.show_comparison(best, worst, self._comparison['best'], self._comparison['worst'])

        self._launch_executor_subprocess(self.generated_schedule_path,
                                         combo_name=best['combination'],
                                         duration=duration, on_finished=after_best)

    def _compare_unavailable(self, reason):
        self.log(f"[Compare] Unavailable: {reason}")
        self._set_run_status("")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Comparison unavailable")
        box.setText(reason)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()

    def _placement_str(self, combo_name):
        p = self._combo_placement(combo_name)
        return ", ".join(f"{m}→{d.upper()}" for m, d in sorted(p.items())) or "?"

    @staticmethod
    def _fmt_measured(m):
        if not m:
            return "-"
        def _n(v, s):
            return f"{v:{s}}" if isinstance(v, (int, float)) else "-"
        return (f"{_n(m.get('throughput_fps'), '.1f')} FPS, "
                f"{_n(m.get('tokens_per_s'), '.1f')} tok/s, "
                f"miss {_n(m.get('deadline_miss_rate'), '.1%')}")

    def show_comparison(self, best_row, worst_row, best_m, worst_m):
        """Side-by-side panel: what the predictor said, and what actually happened.

        Both predicted and measured figures are shown. If the measured order
        contradicts the predicted one, that is stated plainly -- the honest result is
        the interesting one, and hiding it would defeat the point of the comparison.
        """
        from PyQt5.QtWidgets import (QTableWidget, QTableWidgetItem, QVBoxLayout,
                                     QDialogButtonBox, QHeaderView)
        from PyQt5.QtGui import QColor, QFont

        def num(v):
            return v if isinstance(v, (int, float)) and v == v else None

        b_fps, w_fps = num(best_m.get('throughput_fps')), num(worst_m.get('throughput_fps'))
        b_tok, w_tok = num(best_m.get('tokens_per_s')), num(worst_m.get('tokens_per_s'))
        b_miss, w_miss = num(best_m.get('deadline_miss_rate')), num(worst_m.get('deadline_miss_rate'))

        b_place, w_place = self._combo_placement(best_row['combination']), \
            self._combo_placement(worst_row['combination'])
        moved = sorted(m for m in set(b_place) | set(w_place)
                       if b_place.get(m) != w_place.get(m))

        # higher_is_better per row; None = do not highlight
        rows = [
            ("Predicted score", f"{best_row['pred_score']:.4f}", f"{worst_row['pred_score']:.4f}", True),
            ("Predicted throughput (norm)", f"{best_row['pred_norm_throughput']:.3f}",
             f"{worst_row['pred_norm_throughput']:.3f}", True),
            ("Predicted deadline-miss (norm)", f"{best_row['pred_deadline_miss_rate']:.3f}",
             f"{worst_row['pred_deadline_miss_rate']:.3f}", False),
            ("Placement", self._placement_str(best_row['combination']),
             self._placement_str(worst_row['combination']), None),
            ("Moved between devices", ", ".join(
                f"{m}: {w_place.get(m, '-').upper()}→{b_place.get(m, '-').upper()}" for m in moved) or "-",
             "", None),
            ("── MEASURED ──", "", "", None),
            ("Throughput (FPS)", f"{b_fps:.1f}" if b_fps is not None else "-",
             f"{w_fps:.1f}" if w_fps is not None else "-", True),
            ("Tokens/s (LLM+VLM)", f"{b_tok:.1f}" if b_tok is not None else "-",
             f"{w_tok:.1f}" if w_tok is not None else "-", True),
            ("Deadline-miss rate", f"{b_miss:.1%}" if b_miss is not None else "-",
             f"{w_miss:.1%}" if w_miss is not None else "-", False),
            ("Measured window (s)", str(best_m.get('window_sec') or '-'),
             str(worst_m.get('window_sec') or '-'), None),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Best vs Worst placement")
        table = QTableWidget(len(rows), 3, dlg)
        table.setHorizontalHeaderLabels([
            "", f"BEST — {best_row['combination']}", f"WORST — {worst_row['combination']}"])
        table.verticalHeader().setVisible(False)

        better = QColor(210, 245, 210)
        worse = QColor(250, 220, 220)
        bold = QFont()
        bold.setBold(True)

        for r, (label, bval, wval, higher_better) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(label))
            bi, wi = QTableWidgetItem(str(bval)), QTableWidgetItem(str(wval))
            if higher_better is not None:
                try:
                    bnum = float(str(bval).rstrip('%'))
                    wnum = float(str(wval).rstrip('%'))
                    b_wins = (bnum > wnum) if higher_better else (bnum < wnum)
                    win_item, lose_item = (bi, wi) if b_wins else (wi, bi)
                    if bnum != wnum:
                        win_item.setBackground(better)
                        win_item.setFont(bold)
                        lose_item.setBackground(worse)
                except ValueError:
                    pass
            table.setItem(r, 1, bi)
            table.setItem(r, 2, wi)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.resizeRowsToContents()

        # Did the prediction hold? Say so either way.
        verdict = QLabel(dlg)
        verdict.setWordWrap(True)
        if b_fps is None or w_fps is None:
            verdict.setText("Measured throughput unavailable for one of the runs.")
        elif b_fps > w_fps:
            verdict.setText(
                f"✔ The predictor was right: the combination it ranked first measured "
                f"{b_fps:.1f} FPS against {w_fps:.1f} FPS for the one it ranked last "
                f"({(b_fps / w_fps if w_fps else float('inf')):.2f}× the throughput).")
            verdict.setStyleSheet("color: #1a7f37; font-weight: bold; padding: 6px;")
        else:
            verdict.setText(
                f"✘ The predicted order did NOT hold: the best-ranked combination measured "
                f"{b_fps:.1f} FPS but the worst-ranked one measured {w_fps:.1f} FPS. "
                f"The ranking did not reflect the hardware on this run.")
            verdict.setStyleSheet("color: #b35900; font-weight: bold; padding: 6px;")

        buttons = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)

        layout = QVBoxLayout(dlg)
        layout.addWidget(table)
        layout.addWidget(verdict)
        layout.addWidget(buttons)
        dlg.resize(860, 460)

        # Mirror the headline numbers into the status bar the .ui already has.
        if hasattr(self, 'throughput_value') and b_fps is not None:
            self.throughput_value.setText(f"{b_fps:.1f} (best) vs {w_fps:.1f} (worst)")
        if hasattr(self, 'drop_value') and b_miss is not None and w_miss is not None:
            self.drop_value.setText(f"{b_miss:.1%} (best) vs {w_miss:.1%} (worst)")
        if hasattr(self, 'score_value'):
            self.score_value.setText(
                f"{best_row['pred_score']:.3f} (best) vs {worst_row['pred_score']:.3f} (worst)")

        self._set_run_status(
            f"Comparison done — BEST {best_row['combination']}: {self._fmt_measured(best_m)}   |   "
            f"WORST {worst_row['combination']}: {self._fmt_measured(worst_m)}")
        self.log(f"[Compare] {verdict.text()}")
        dlg.exec_()

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
        duration = self._run_duration()
        # 4) Launch executor in a subprocess with selected combo
        self._set_run_status(f"Running: BEST ({best_combo})")
        self._launch_executor_subprocess(schedule_path, combo_name=best_combo, duration=duration)

    def on_predict_best_clicked(self):
        """Handler invoked when predict_best_button is clicked."""
        models_root = self.deployment_model_input.text() if hasattr(self, 'deployment_model_input') else self.models_root
        pred_model = self.prediction_model_input.text() if hasattr(self, 'prediction_model_input') else ''

        # Models checked in the tree view (names, no extension)
        selected = self._get_selected_model_names()

        # Log inputs
        self.log(f"[Predict] models_root={models_root}")
        self.log(f"[Predict] prediction_model={pred_model}")
        self.log(f"[Predict] platform_devices={'/'.join(getattr(self, 'platform_devices', []))}")
        self.log(f"[Predict] selected_models={selected}")

        # Validate. The platform comes from the Device Configuration combo, so no
        # device config file is required. `pred_model` may be a prefix, a directory,
        # or a _y*.json.
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

        # Step 0a: Preflight. Runs BEFORE the coverage check so a broken environment
        # or a missing artifact never masquerades as an out-of-distribution warning.
        problems = self.preflight_check(pred_model)
        if problems:
            for p in problems:
                self.log(f"[Error][Preflight] {p}")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Cannot run the predictor")
            box.setText("The placement predictor cannot run in this environment.")
            box.setInformativeText("\n\n".join(f"• {p}" for p in problems))
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
            if hasattr(self, 'label_best_deploy_value'):
                self.label_best_deploy_value.setText('-')
            return

        # Step 0b: Internal consistency check. The list only offers working sets the
        # predictor was trained on, so this must always pass -- a warning here means
        # the list-building logic is broken, not that the user did something wrong.
        # Log it and continue rather than blocking them with a dialog.
        try:
            ood = self.check_selection_coverage(selected, pred_model)
        except Exception as e:
            ood = [f"Coverage check failed: {e}"]
        for w in ood:
            self.log(f"[BUG] selection should always be in-distribution: {w}")

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