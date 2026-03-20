"""
Main UnifiedViewer class for the multimodel scheduling application.
"""
import os
import json
import signal
import yaml
import queue
from datetime import datetime
from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic
from queue import Queue
from threading import Event, Thread
import threading

# Import local modules
from utils import get_cpu_metrics, create_x_image, convert_cv_to_qt
from view_handlers import ModelSignals, YoloViewHandler, ResNetViewHandler, VideoFeeder, ResnetImageFeeder
from model_processors import (
    video_reader_process,
    run_yolo_cpu_process,
    run_resnet_cpu_process,
    run_yolo_gpu_process,
    run_resnet_gpu_process,
    run_yolo_npu_process,
    run_resnet_npu_process
)

class InfoWindow(QWidget):
    """Main window for displaying system and model information."""
    
    def __init__(self, parent=None):
        """Initialize the InfoWindow."""
        super().__init__()
        # Load UI from file instead of creating components programmatically
        uic.loadUi("info_window.ui", self)
        # Set window flags to make it behave like a main window and always stay on top of other app windows
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # Store parent reference for callbacks
        self.parent = parent
        
        # Add black borders to all labels
        border_style = "border: 1px solid black;"
        self.schedule_name_label.setStyleSheet(border_style)
        try:
            self.metrics_label.setStyleSheet(border_style)
        except Exception:
            pass
        # Trigger text under metrics: left aligned; keep border for consistency
        try:
            self.trigger_below_metrics_label.setStyleSheet(border_style)
        except Exception:
            pass
        self.model_performance_label.setStyleSheet(border_style)
        self.cpu_info_label.setStyleSheet(border_style)
        try:
            self.npu_info_label.setStyleSheet(border_style)
        except Exception:
            pass
        
        # Initialize the start button
        self.start_button.clicked.connect(self.on_start_button_clicked)
        
        # Initialize the stop button
        self.stop_button.clicked.connect(self.on_stop_button_clicked)

        # Initialize the best button (open JSON file dialog)
        try:
            self.best_button.clicked.connect(self.load_best_schedule)
        except Exception:
            # If best_button is not present for some reason, ignore gracefully
            pass

        # Connect the Browse... button to select a schedule folder
        try:
            self.browse_schedule_folder_button.clicked.connect(self.on_browse_schedule_folder_clicked)
        except Exception:
            # If the button is not present, ignore gracefully
            pass
        
        # Default execution duration is 60 seconds
        self.duration_edit.setText("10")
        
    def on_start_button_clicked(self):
        """Handle the start button click event."""
        try:
            duration = int(self.duration_edit.text())
            if duration <= 0:
                print("Duration must be a positive number")
                return
            
            # Call the parent's start_execution method if available
            if self.parent and hasattr(self.parent, 'start_execution'):
                self.parent.start_execution(duration)
            else:
                print(f"Starting execution with duration: {duration} seconds")
        except ValueError:
            print("Please enter a valid number for duration")
            
    def on_stop_button_clicked(self):
        """Handle the stop button click event."""
        # Prefer async stop if available to keep UI responsive
        if self.parent and hasattr(self.parent, 'stop_execution_async'):
            self.parent.stop_execution_async()
        elif self.parent and hasattr(self.parent, 'stop_execution'):
            self.parent.stop_execution()
        else:
            print("Stopping execution")

    def load_best_schedule(self):
        """Open a file dialog to select a .json schedule and load it. If canceled, do nothing."""
        try:
            # Open file dialog restricted to JSON files
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Best Schedule JSON",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            # If the dialog was canceled or closed (e.g., ESC), do nothing
            if not file_path:
                return
            # Remember selection and show in the schedule name label
            self.selected_best_json = file_path
            try:
                self.update_schedule_name(f"Best schedule: {os.path.basename(file_path)}")
            except Exception:
                # Fallback: set text directly
                self.schedule_name_label.setText(f"Best schedule: {os.path.basename(file_path)}")
            
            # If parent viewer can handle it, notify parent
            if self.parent and hasattr(self.parent, 'load_best_schedule'):
                try:
                    self.parent.load_best_schedule(file_path)
                except Exception as e:
                    print(f"[InfoWindow] load_best_schedule failed: {e}")
        except Exception as e:
            print(f"[InfoWindow] Failed to open file dialog: {e}")
        
    def on_browse_schedule_folder_clicked(self):
        """Open a directory selection dialog and set the chosen path into schedule_folder_input."""
        try:
            try:
                start_dir = self.schedule_folder_input.text().strip()
            except Exception:
                start_dir = ""
            if not start_dir:
                start_dir = os.getcwd()
            folder = QFileDialog.getExistingDirectory(self, "Select Schedule Folder", start_dir)
            if folder:
                try:
                    self.schedule_folder_input.setText(folder)
                except Exception:
                    pass
        except Exception as e:
            print(f"[InfoWindow] Failed to open select folder dialog: {e}")
        
    def get_execution_duration(self):
        """Get the execution duration from the input field."""
        try:
            return int(self.duration_edit.text())
        except ValueError:
            return 60  # Default value if input is invalid
    
    def update_model_performance(self, text):
        """Update the model performance label."""
        self.model_performance_label.setText(text)
    
    def update_trigger_below_metrics(self, text):
        """Update the trigger status text placed under metrics (left-aligned)."""
        try:
            self.trigger_below_metrics_label.setStyleSheet("border: 1px solid black;")
        except Exception:
            pass
        self.trigger_below_metrics_label.setText(text)
    
    def update_cpu_info(self, text):
        """Update the CPU info label."""
        self.cpu_info_label.setText(text)
    
    def update_npu_info(self, text):
        """Update the NPU info label."""
        self.npu_info_label.setText(text)
        
    def update_schedule_name(self, text):
        """Update the schedule name label."""
        self.schedule_name_label.setText(text)
        
    def update_metrics(self, text):
        """Update the compact metrics label below the schedule name."""
        try:
            self.metrics_label.setText(text)
        except Exception:
            pass
        
    def closeEvent(self, event):
        """Handle window close event - terminate the application (non-blocking)."""
        print("[InfoWindow] Close event triggered - ignoring hide/close")
        event.ignore()
        # If we have a parent (UnifiedViewer), call its async shutdown method if available
        if self.parent and hasattr(self.parent, 'shutdown_all_async'):
            try:
                self.parent.shutdown_all_async()
                return
            except Exception as e:
                pass
        # Fallback to synchronous shutdown_all or direct exit
        if self.parent and hasattr(self.parent, 'shutdown_all'):
            self.parent.shutdown_all()
        else:
            print("[InfoWindow] Closing application directly")
            try:
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    QTimer.singleShot(50, app.quit)
                else:
                    import sys
                    sys.exit(0)
            except Exception:
                import sys
                sys.exit(0)

class UnifiedViewer(QMainWindow):
    """Main viewer class for the multimodel scheduling application."""
    
    def __init__(self, schedule_file='model_schedules.yaml', combination_name=None, info_window=None, hide_info_window=False):
        """Initialize the UnifiedViewer.
        
        Args:
            schedule_file (str): Path to the model scheduling information file.
            combination_name (str|None): Specific combination key to use from the YAML. If None, default logic applies.
            info_window (InfoWindow|None): Pre-created InfoWindow instance to use.
            hide_info_window (bool): If True, the InfoWindow will be hidden even if created/passed.
        """
        super().__init__()
        uic.loadUi("schedule_executor_display.ui", self)
        
        # Set up signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Flags for async shutdown/stop
        self._shutdown_in_progress = False
        self._stop_in_progress = False

        # Store the schedule file path and requested combination
        self.schedule_file = schedule_file
        self.requested_combination = combination_name
        
        # Create or reuse the info window as the main window (persistent across runs)
        if info_window is not None:
            self.info_window = info_window
            # Rebind controller so buttons control this viewer instance
            try:
                self.info_window.parent = self
            except Exception:
                pass
            
            # If hide_info_window is True, force-set the hidden_headless flag
            if hide_info_window:
                try:
                    self.info_window.hidden_headless = True
                    self.info_window.hide()
                except Exception:
                    pass
            
            # Show the info window only when not marked as headless/hidden
            try:
                if not getattr(self.info_window, 'hidden_headless', False):
                    self.info_window.show()
                else:
                    self.info_window.hide()
            except Exception:
                # Fallback to show to preserve legacy behavior if attribute missing
                try:
                    if not hide_info_window:
                        self.info_window.show()
                    else:
                        self.info_window.hide()
                except Exception:
                    pass
        else:
            self.info_window = InfoWindow(parent=self)
            # When created internally, if hide_info_window is requested, mark it.
            if hide_info_window:
                try:
                    self.info_window.hidden_headless = True
                except Exception:
                    pass
            
            # When created internally, we don't show it by default to avoid unexpected InfoWindows
            # especially for direct execution modes like best_deploy_finder_executor.
            # If the user wants to see it, they should pass an InfoWindow instance.
            try:
                self.info_window.hide()
            except Exception:
                pass
        
        # Set window title to indicate it's a secondary window
        self.setWindowTitle("Schedule Executor Display (Secondary Window)")

        # Initialize model settings and views
        self.initialize_model_settings()
        self.initialize_ui_components()
        self.initialize_state_variables()
        # Don't start processes and threads automatically
        # They will be started when the Start Execution button is clicked
        
        # CPU/NPU monitoring
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)
    
    def initialize_model_settings(self, schedule_file=None, combination_name=None):
        """Initialize model settings from YAML configuration."""
        if schedule_file is not None:
            self.schedule_file = schedule_file
        if combination_name is not None:
            self.requested_combination = combination_name

        self.model_settings = {}
        self.views_without_model = set()  # Track views without specified models
        # Headless models: run without occupying any of view1..view4
        self.headless_ids = []           # list of headless identifiers
        # Note: we keep hidden_views for backward compatibility, but we no longer
        # auto-map display:none models into views. The set remains empty unless
        # explicitly manipulated elsewhere.
        self.hidden_views = set()
        # Default combination, can be overridden by requested_combination
        self.current_combination = self.requested_combination or "combination1"
        
        # Update the schedule name label
        self.info_window.update_schedule_name(f"Current Schedule: {self.current_combination}")
        
        try:
            # Helper to resolve a model identifier (possibly a folder name) to an actual ONNX file path
            def _resolve_model_path(model_id: str) -> str:
                try:
                    mid = str(model_id or '').strip()
                    if not mid:
                        return ''
                    # If it already points to a file that exists, return as-is
                    if os.path.isabs(mid) and os.path.isfile(mid):
                        return mid
                    # If it looks like a relative path containing a separator, try relative to CWD
                    if any(sep in mid for sep in ['/', os.sep, '\\']):
                        cand = os.path.abspath(mid)
                        if os.path.isfile(cand):
                            return cand
                        # If it's a directory, try model.onnx inside
                        if os.path.isdir(cand):
                            p = os.path.join(cand, 'model.onnx')
                            if os.path.isfile(p):
                                return p
                            # else first .onnx inside
                            for n in os.listdir(cand):
                                if n.lower().endswith('.onnx'):
                                    return os.path.join(cand, n)
                    # Otherwise, interpret as a model name possibly equal to a folder under models_onnx
                    models_root = os.path.join(os.getcwd(), 'models_onnx')
                    folder = os.path.join(models_root, mid)
                    if os.path.isdir(folder):
                        p = os.path.join(folder, 'model.onnx')
                        if os.path.isfile(p):
                            return p
                        for n in os.listdir(folder):
                            if n.lower().endswith('.onnx'):
                                return os.path.join(folder, n)
                    # Fallback: try models_root/<mid>.onnx
                    direct = os.path.join(models_root, f"{mid}.onnx")
                    if os.path.isfile(direct):
                        return direct
                except Exception:
                    pass
                return ''

            # Load configuration from the specified schedule file
            with open(self.schedule_file, "r") as f:
                config = yaml.safe_load(f) or {}
                
            # Create a mapping from views to model configurations based on the display field
            view_to_model_map = {}
            
            # If a specific combination was not requested or not found, choose the first available
            if (self.current_combination not in config) and config:
                self.current_combination = next(iter(config))
                # Update the schedule name label with the actual combination being used
                self.info_window.update_schedule_name(f"Current Schedule: {self.current_combination}")
                
            # Use the selected combination configuration
            # models with display suppressed; run headlessly and DO NOT bind to views
            if self.current_combination in config:
                for model_config_name, model_config in (config[self.current_combination] or {}).items():
                    if isinstance(model_config, dict) and "display" in model_config:
                        view_name = model_config.get("display")
                        vnorm = str(view_name).strip().lower() if view_name is not None else ""
                        # If display is set to a 'none/off' value, schedule as headless (no on-screen drawing)
                        if (not vnorm) or (vnorm in {"none", "off", "hidden", "no", "false", "0"}):
                            # Build a stable headless id using the config key
                            safe_name = str(model_config_name).replace(" ", "_")
                            hid = f"headless_{safe_name}"
                            self.headless_ids.append(hid)
                            # Store headless config inside model_settings under its id so
                            # feeders/process starters can use common paths without special cases
                            mval = model_config.get("model", "")
                            self.model_settings[hid] = {
                                "model": mval,
                                "model_path": _resolve_model_path(mval),
                                "execution": model_config.get("execution", "cpu"),
                                "infps": model_config.get("infps", None)
                            }
                            continue
                        # Only allow known view labels
                        if vnorm in {"view1", "view2", "view3", "view4"}:
                            view_key = vnorm
                        else:
                            try:
                                print(f"[UnifiedViewer] Unknown display label '{view_name}' for {model_config_name}; scheduling hidden")
                            except Exception:
                                pass
                            # Treat unknown labels as headless to be safe
                            safe_name = str(model_config_name).replace(" ", "_")
                            hid = f"headless_{safe_name}"
                            self.headless_ids.append(hid)
                            mval = model_config.get("model", "")
                            self.model_settings[hid] = {
                                "model": mval,
                                "model_path": _resolve_model_path(mval),
                                "execution": model_config.get("execution", "cpu"),
                                "infps": model_config.get("infps", None)
                            }
                            continue
                        mval = model_config.get("model", "")
                        view_to_model_map[view_key] = {
                            "model": mval,
                            "model_path": _resolve_model_path(mval),
                            "execution": model_config.get("execution", "cpu"),
                            "infps": model_config.get("infps", None)
                        }
            
            # Assign model configurations to views
            for view in ["view1", "view2", "view3", "view4"]:
                if view in view_to_model_map:
                    self.model_settings[view] = view_to_model_map[view]
                else:
                    # Mark this view as not having a specified model
                    self.views_without_model.add(view)
                    # Still add default settings for compatibility with existing code
                    self.model_settings[view] = {
                        "model": "",
                        "model_path": "",
                        "execution": "cpu"
                    }
                    # Informational: this view is simply unused by the selected combination
                    print(f"[UnifiedViewer] {view} not used in this combination (no model assigned) [{os.path.basename(self.schedule_file)}]")

            # Detect PyTorch availability for NPU execution and remap to CPU if unavailable
            torch_available = True
            try:
                import torch  # noqa: F401
            except Exception:
                torch_available = False
            if not torch_available:
                # Remap any NPU executions to CPU to avoid runtime import errors
                for v, cfg in self.model_settings.items():
                    exec_dev = (cfg or {}).get("execution", "cpu")
                    if isinstance(exec_dev, str) and exec_dev.lower().startswith("npu"):
                        cfg["execution"] = "cpu"
                        print(f"[UnifiedViewer] PyTorch not found; falling back to CPU for {v} (was {exec_dev})")

            print(f"[UnifiedViewer] Loaded model settings from {self.schedule_file} for {self.current_combination}")
        except Exception as e:
            print(f"[UnifiedViewer ERROR] Failed to load {self.schedule_file}: {e}")
            # Set default settings if file loading fails
            self.model_settings = {
                "view1": {"model": "", "model_path": "", "execution": "cpu"},
                "view2": {"model": "resnet50_small", "model_path": "", "execution": "cpu"},
                "view3": {"model": "", "model_path": "", "execution": "cpu"},
                "view4": {"model": "resnet50_small", "model_path": "", "execution": "cpu"}
            }
            # No views are marked as without model in case of error
    
    def initialize_ui_components(self):
        """Initialize UI components."""
        self.view1 = self.findChild(QLabel, "view1")
        self.view2 = self.findChild(QLabel, "view2")
        self.view3 = self.findChild(QLabel, "view3")
        self.view4 = self.findChild(QLabel, "view4")
        
        # Define and connect signals
        self.model_signals = ModelSignals()
        self.model_signals.update_view1_display.connect(self.update_view1_display)
        self.model_signals.update_view2_display.connect(self.update_view2_display)
        self.model_signals.update_view3_display.connect(self.update_view3_display)
        self.model_signals.update_view4_display.connect(self.update_view4_display)
        
        # Initialize placeholders for views without associated (visible) model
        try:
            self._init_placeholders()
        except Exception:
            pass
    
    def _init_placeholders(self):
        """Show 'No model specified' placeholder on views with no visible model mapping.
        - For views in views_without_model: always show placeholder.
        - For views in hidden_views: show placeholder; actual model will run headlessly and updates are suppressed.
        """
        try:
            x_img = create_x_image()
            pix = convert_cv_to_qt(x_img)
        except Exception:
            pix = None
        for vname in ["view1", "view2", "view3", "view4"]:
            try:
                if (hasattr(self, 'views_without_model') and vname in self.views_without_model) or (hasattr(self, 'hidden_views') and vname in self.hidden_views):
                    lbl = getattr(self, vname, None)
                    if lbl is not None and pix is not None and not pix.isNull():
                        lbl.setPixmap(pix)
                        lbl.setScaledContents(True)
            except Exception:
                continue
    
    def initialize_state_variables(self):
        """Initialize state variables."""
        # Global flag for signaling threads to exit
        self.global_exit_flag = False
        
        # Initialize common state variables
        self.shutdown_flag = Event()
        self.prev_cpu_stats = get_cpu_metrics(interval=0)
        
        # Ensure stop_execution is idempotent: save throughput only once per schedule
        self._already_stopped = False
        # Track if a run is currently active to prevent duplicate starts
        self._run_active = False

        # Store the requested execution window duration (seconds) for saving into results
        self.window_duration_sec = None

        # Initialize queues and events
        self.video_frame_queue = Queue(maxsize=2)
        self.video_shutdown_event = Event()
        
        # View1 queues and events
        self.view1_frame_queue = Queue(maxsize=2)
        self.view1_output_queue = Queue(maxsize=1)
        self.view1_shutdown_event = Event()
        
        # View2 queues and events
        self.view2_frame_queue = Queue(maxsize=2)
        self.view2_output_queue = Queue(maxsize=1)
        self.view2_shutdown_event = Event()
        
        # View3 queues and events
        self.view3_frame_queue = Queue(maxsize=2)
        self.view3_result_queue = Queue(maxsize=1)
        self.view3_shutdown_event = Event()
        
        # View4 queues and events
        self.view4_frame_queue = Queue(maxsize=2)
        self.view4_result_queue = Queue(maxsize=1)
        self.view4_shutdown_event = Event()
        
        # Initialize a dictionary to track which views are running YOLO models (need video frames)
        self.yolo_views = set()
        # Track ResNet views that need image feeder at 10 Hz
        self.resnet_views = set()
        # Headless resources (queues/events/processes)
        self.headless_frame_queues = {}
        self.headless_output_queues = {}
        self.headless_shutdown_events = {}
        self.headless_processes = []
    
    def initialize_processes(self):
        """Initialize and start model workers (single-process, multi-thread)."""
        # Start video reader thread only if any model requires YOLO video (yolov4)
        need_video = any("yolov4" in (cfg or {}).get("model", "") for cfg in self.model_settings.values())
        self.video_reader_proc = None
        if need_video:
            self.video_reader_proc = Thread(
                target=video_reader_process,
                args=("stockholm_1280x720.mp4", self.video_frame_queue, self.video_shutdown_event),
                daemon=True,
            )
            self.video_reader_proc.start()
        else:
            pass
        
        # Start view worker threads
        self.start_view_process("view1")
        self.start_view_process("view2")
        self.start_view_process("view3")
        self.start_view_process("view4")

        # Start headless worker threads (do not occupy UI views)
        for hid in list(getattr(self, 'headless_ids', []) or []):
            # Prepare queues/events for this headless id
            if hid not in self.headless_frame_queues:
                self.headless_frame_queues[hid] = Queue(maxsize=2)
            if hid not in self.headless_output_queues:
                # YOLO uses output_queue; ResNet uses result_queue name-wise, but both are simple queues
                self.headless_output_queues[hid] = Queue(maxsize=1)
            if hid not in self.headless_shutdown_events:
                self.headless_shutdown_events[hid] = Event()

            cfg = self.model_settings.get(hid, {})
            model = cfg.get("model", "")
            execution = cfg.get("execution", "cpu")

            frame_queue = self.headless_frame_queues[hid]
            output_queue = self.headless_output_queues[hid]
            shutdown_event = self.headless_shutdown_events[hid]

            # Register into yolo/resnet sets so feeders can send inputs
            if "yolov4" in model:
                self.yolo_views.add(hid)
                if execution == "gpu":
                    process = Thread(
                        target=run_yolo_gpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid, model),
                        daemon=True,
                    )
                elif execution in ("npu0", "npu1"):
                    print(f"[UnifiedViewer] Warning: execution={execution} is deprecated. Falling back to GPU for {hid} ({model}).")
                    process = Thread(
                        target=run_yolo_gpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid, model),
                        daemon=True,
                    )
                else:
                    process = Thread(
                        target=run_yolo_cpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid),
                        daemon=True,
                    )
            else:
                self.resnet_views.add(hid)
                if execution == "gpu":
                    process = Thread(
                        target=run_resnet_gpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid),
                        daemon=True,
                    )
                elif execution in ("npu0", "npu1"):
                    print(f"[UnifiedViewer] Warning: execution={execution} is deprecated. Falling back to GPU for {hid}.")
                    process = Thread(
                        target=run_resnet_gpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid),
                        daemon=True,
                    )
                else:
                    process = Thread(
                        target=run_resnet_cpu_process,
                        args=(frame_queue, output_queue, shutdown_event, hid),
                        daemon=True,
                    )
            process.start()
            self.headless_processes.append(process)
        
        # Start drainers for headless outputs to avoid queue backpressure
        def _make_drain(q, ev, name, cfg):
            def _drain():
                import queue as _q
                while not ev.is_set() and not self.shutdown_flag.is_set():
                    try:
                        item = q.get(timeout=1)
                    except _q.Empty:
                        continue
                    except Exception:
                        break

                    # Print concise command-line logs for headless model results
                    try:
                        ts = datetime.now().strftime("%H:%M:%S")
                        model_name = str((cfg or {}).get("model", "") or name)
                        exec_dev = str((cfg or {}).get("execution", "cpu") or "cpu").upper()
                        combo = getattr(self, 'current_combination', '')
                        run_id = getattr(self, 'run_id', '')

                        # Try to interpret common payload shapes from model_processors:
                        # - ResNet: (img, class_name, infer_time_ms)
                        # - YOLO: (result_img, infer_time_ms, wait_ms)
                        # Fallback: generic repr length-limited
                        msg = None
                        if isinstance(item, tuple):
                            if len(item) == 3 and isinstance(item[1], str):
                                # ResNet
                                class_name = item[1]
                                try:
                                    infer_ms = float(item[2])
                                except Exception:
                                    infer_ms = None
                                if infer_ms is not None:
                                    msg = f"class={class_name} infer={infer_ms:.1f}ms"
                                else:
                                    msg = f"class={class_name}"
                            elif len(item) == 3:
                                # YOLO (image, infer_ms, wait_ms)
                                try:
                                    infer_ms = float(item[1])
                                except Exception:
                                    infer_ms = None
                                try:
                                    wait_ms = float(item[2])
                                except Exception:
                                    wait_ms = None
                                if infer_ms is not None and wait_ms is not None:
                                    msg = f"infer={infer_ms:.1f}ms wait={wait_ms:.1f}ms"
                                elif infer_ms is not None:
                                    msg = f"infer={infer_ms:.1f}ms"
                            elif len(item) == 2 and isinstance(item[1], (int, float)):
                                # Some pipelines may return (payload, infer_ms)
                                try:
                                    infer_ms = float(item[1])
                                    msg = f"infer={infer_ms:.1f}ms"
                                except Exception:
                                    msg = None
                        if msg is None:
                            # Fallback: avoid dumping large arrays/images
                            msg = "result received"

                        # Accumulate basic headless stats for throughput saving
                        try:
                            stats = getattr(self, 'headless_stats', None)
                            if stats is None:
                                self.headless_stats = {}
                                stats = self.headless_stats
                            s = stats.setdefault(name, {
                                'count': 0,
                                'sum_infer_ms': 0.0,
                                'sum_wait_ms': 0.0,
                                'wait_count': 0,
                                'model': model_name,
                                'execution': exec_dev,
                            })
                            s['count'] += 1
                            try:
                                if infer_ms is not None:
                                    s['sum_infer_ms'] += float(infer_ms)
                            except Exception:
                                pass
                            try:
                                if wait_ms is not None:
                                    s['sum_wait_ms'] += float(wait_ms)
                                    s['wait_count'] += 1
                            except Exception:
                                pass
                        except Exception:
                            pass

                        print(f"[Headless][{ts}][{combo}][{run_id}] {name} {model_name} {exec_dev}: {msg}")
                    except Exception:
                        # Never let logging break the drainer
                        pass

            t = Thread(target=_drain, daemon=True, name=f"Drainer-{name}")
            t.start()
            return t
        self._headless_drainers = []
        # Ensure stats container exists for headless jobs
        try:
            if getattr(self, 'headless_stats', None) is None:
                self.headless_stats = {}
        except Exception:
            self.headless_stats = {}
        for hid in list(getattr(self, 'headless_ids', []) or []):
            q = self.headless_output_queues.get(hid)
            ev = self.headless_shutdown_events.get(hid)
            cfg = self.model_settings.get(hid, {})
            if q is not None and ev is not None:
                self._headless_drainers.append(_make_drain(q, ev, hid, cfg))
    
    def start_view_process(self, view_name):
        """
        Start a worker thread for a specific view.
        
        Args:
            view_name: Name of the view (view1, view2, etc.)
        """
        # If no model is assigned for this view, do not start any worker
        if hasattr(self, 'views_without_model') and view_name in self.views_without_model:
            print(f"[UnifiedViewer] Skipping worker start for {view_name}: no model assigned")
            return
        
        model = self.model_settings.get(view_name, {}).get("model", "")
        execution = self.model_settings.get(view_name, {}).get("execution", "cpu")
        
        frame_queue = getattr(self, f"{view_name}_frame_queue")
        output_queue = getattr(self, f"{view_name}_output_queue") if view_name in ["view1", "view2"] else getattr(self, f"{view_name}_result_queue")
        shutdown_event = getattr(self, f"{view_name}_shutdown_event")
        
        if "yolov4" in model:
            # YOLOv4 model
            self.yolo_views.add(view_name)
            if execution == "gpu":
                print(f"[UnifiedViewer] Starting {view_name} with {model} GPU (thread)")
                process = Thread(
                    target=run_yolo_gpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name, model),
                    daemon=True,
                )
            elif execution in ("npu0", "npu1"):
                # NPU execution is deprecated; fall back to GPU to align with new policy
                print(f"[UnifiedViewer] Warning: execution={execution} is deprecated. Falling back to GPU for {view_name} ({model}).")
                process = Thread(
                    target=run_yolo_gpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name, model),
                    daemon=True,
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU (thread)")
                process = Thread(
                    target=run_yolo_cpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
                    daemon=True,
                )
        else:
            # ResNet model
            self.resnet_views.add(view_name)
            if execution == "gpu":
                print(f"[UnifiedViewer] Starting {view_name} with {model} GPU (thread)")
                process = Thread(
                    target=run_resnet_gpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
                    daemon=True,
                )
            elif execution in ("npu0", "npu1"):
                print(f"[UnifiedViewer] Warning: execution={execution} is deprecated. Falling back to GPU for {view_name} ({model}).")
                process = Thread(
                    target=run_resnet_gpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
                    daemon=True,
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU (thread)")
                process = Thread(
                    target=run_resnet_cpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
                    daemon=True,
                )
        
        setattr(self, f"{view_name}_process", process)
        process.start()
    
    def initialize_threads(self):
        """Initialize and start view handler threads."""
        # Create view frame queues dictionary
        view_frame_queues = {
            "view1": self.view1_frame_queue,
            "view2": self.view2_frame_queue,
            "view3": self.view3_frame_queue,
            "view4": self.view4_frame_queue
        }
        # Extend with headless frame queues so feeders can push inputs
        for hid, fq in (getattr(self, 'headless_frame_queues', {}) or {}).items():
            view_frame_queues[hid] = fq
        
        # Start video feeder thread only if there are YOLOv4 views
        self.video_feeder = None
        if self.yolo_views:
            self.video_feeder = VideoFeeder(
                self.video_frame_queue,
                view_frame_queues,
                self.yolo_views,
                self.shutdown_flag,
                model_settings=self.model_settings
            )
            self.video_feeder.start_feed_thread()
        # Start ResNet image feeder honoring per-view infps (defaulting to 2 FPS)
        self.resnet_feeder = ResnetImageFeeder(
            image_dir="./imagenet-sample-images",
            view_frame_queues=view_frame_queues,
            resnet_views=self.resnet_views,
            shutdown_flag=self.shutdown_flag,
            model_settings=self.model_settings,
            default_interval_sec=0.5
        )
        self.resnet_feeder.start_feed_thread()
        
        # Start view handler threads
        self.initialize_view_handlers()
    
    def initialize_view_handlers(self):
        """Initialize and start view handler threads."""
        # View1 handler
        view1_model = self.model_settings.get("view1", {}).get("model", "")
        if "yolov4" in view1_model:
            self.view1_handler = YoloViewHandler(
                "view1",
                self.model_settings,
                self.view1_frame_queue,
                self.view1_output_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        else:
            self.view1_handler = ResNetViewHandler(
                "view1",
                self.model_settings,
                self.view1_frame_queue,
                self.view1_output_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        self.view1_handler.start_display_thread()
        
        # View2 handler
        view2_model = self.model_settings.get("view2", {}).get("model", "")
        if "yolov4" in view2_model:
            self.view2_handler = YoloViewHandler(
                "view2",
                self.model_settings,
                self.view2_frame_queue,
                self.view2_output_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        else:
            self.view2_handler = ResNetViewHandler(
                "view2",
                self.model_settings,
                self.view2_frame_queue,
                self.view2_output_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        self.view2_handler.start_display_thread()
        
        # View3 handler
        view3_model = self.model_settings.get("view3", {}).get("model", "")
        if "yolov4" in view3_model:
            self.view3_handler = YoloViewHandler(
                "view3",
                self.model_settings,
                self.view3_frame_queue,
                self.view3_result_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        else:
            self.view3_handler = ResNetViewHandler(
                "view3",
                self.model_settings,
                self.view3_frame_queue,
                self.view3_result_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        self.view3_handler.start_display_thread()
        
        # View4 handler
        view4_model = self.model_settings.get("view4", {}).get("model", "")
        if "yolov4" in view4_model:
            self.view4_handler = YoloViewHandler(
                "view4",
                self.model_settings,
                self.view4_frame_queue,
                self.view4_result_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        else:
            self.view4_handler = ResNetViewHandler(
                "view4",
                self.model_settings,
                self.view4_frame_queue,
                self.view4_result_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model
            )
        self.view4_handler.start_display_thread()
    
    # View update methods
    def update_view1_display(self, pixmap):
        """Update view1 display."""
        if hasattr(self, 'hidden_views') and 'view1' in self.hidden_views:
            return  # suppress drawing when hidden
        self.view1.setPixmap(pixmap)
        self.view1.setScaledContents(True)
    
    def update_view2_display(self, pixmap):
        """Update view2 display."""
        if hasattr(self, 'hidden_views') and 'view2' in self.hidden_views:
            return
        self.view2.setPixmap(pixmap)
        self.view2.setScaledContents(True)
    
    def update_view3_display(self, pixmap):
        """Update view3 display."""
        if hasattr(self, 'hidden_views') and 'view3' in self.hidden_views:
            return
        self.view3.setPixmap(pixmap)
        self.view3.setScaledContents(True)
    
    def update_view4_display(self, pixmap):
        """Update view4 display."""
        if hasattr(self, 'hidden_views') and 'view4' in self.hidden_views:
            return
        self.view4.setPixmap(pixmap)
        self.view4.setScaledContents(True)
    
    # Signal handling and shutdown methods
    def signal_handler(self, sig, frame):
        """Handle SIGINT signal (Ctrl+C)."""
        print("\n[SIGINT] Caught Ctrl+C, shutting down...")
        # Immediately set shutdown flags to stop video generation
        self.shutdown_flag.set()
        self.global_exit_flag = True
        
        # Set all shutdown events to stop processes
        for name in ['view1_shutdown_event', 'view2_shutdown_event',
                     'view3_shutdown_event', 'view4_shutdown_event',
                     'video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
                
        # Exit immediately without cleaning up queues
        print("[SIGINT] Requesting application quit")
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(50, app.quit)
        except Exception:
            pass
    
    def closeEvent(self, event):
        """Handle window close event.
        - In normal GUI mode: ignore close and keep app running.
        - In executor-only mode (--schedule_name): shutdown asynchronously.
        """
        # If running in executor-only mode, shut down everything asynchronously; otherwise ignore close
        if getattr(self, 'executor_only', False):
            event.accept()
            print("[UnifiedViewer] Close event in executor-only mode - terminating application (async)")
            try:
                if hasattr(self, 'shutdown_flag') and self.shutdown_flag:
                    self.shutdown_flag.set()
            except Exception:
                pass
            self.global_exit_flag = True
            for name in ['view1_shutdown_event', 'view2_shutdown_event',
                         'view3_shutdown_event', 'view4_shutdown_event',
                         'video_shutdown_event']:
                try:
                    ev = getattr(self, name, None)
                    if ev:
                        ev.set()
                except Exception:
                    pass
            # Defer heavy stopping/cleanup to background to avoid freezing GUI
            try:
                self.stop_execution_async()
            except Exception:
                # Fallback to synchronous stop if async path unavailable
                try:
                    self.stop_execution()
                except Exception:
                    pass
            try:
                if self.info_window:
                    self.info_window.hide()
            except Exception:
                pass
            try:
                self.shutdown_all_async()
            except Exception:
                # Fallback to sync
                self.shutdown_all()
        else:
            event.ignore()
            print("[UnifiedViewer] Close event ignored - window remains open; Use Info window buttons to stop")
    
    def shutdown_all(self):
        """Clean up resources and shut down the application gracefully."""
        # First stop all model execution
        try:
            self.stop_execution()
        except Exception as e:
            pass
        # Request application quit without forcing interpreter exit
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                print("[Shutdown] Requesting application quit")
                QTimer.singleShot(50, app.quit)
        except Exception as e:
            pass
    
    def update_combination(self, schedule_file, combination_name):
        """Update the viewer with a new schedule file and combination name for reuse.
        
        Args:
            schedule_file (str): Path to the new model scheduling information file.
            combination_name (str): New combination key to use from the YAML.
        """
        print(f"[UnifiedViewer] Updating combination to: {combination_name} (File: {schedule_file})")
        self.schedule_file = schedule_file
        self.requested_combination = combination_name
        
        # Stop any active run before updating settings
        try:
            self.stop_execution()
        except Exception:
            pass
            
        # Re-initialize with new combination settings
        self.initialize_model_settings(schedule_file, combination_name)
        # Note: UI components (placeholders) are typically static once created,
        # but we re-initialize state variables for the new run.
        self.initialize_state_variables()
        
        # Bring window to front if needed, but maintain position/geometry
        try:
            if not self.isVisible():
                self.show()
            else:
                self.raise_()
                self.activateWindow()
        except Exception:
            pass

    # Monitoring and statistics methods
    def start_execution(self, duration):
        """
        Start execution with a specified duration.
        
        Args:
            duration (int): Duration in seconds for the execution to run.
        """
        print(f"Starting execution with duration: {duration} seconds")
        # If a stop/shutdown is in progress, retry shortly to avoid race and UI freeze
        try:
            if getattr(self, '_stop_in_progress', False) or getattr(self, '_shutdown_in_progress', False):
                print("[Start] Stop/shutdown in progress; retrying in 200 ms")
                QTimer.singleShot(200, lambda d=duration: self.start_execution(d))
                return
        except Exception:
            pass
        # Prevent duplicate start within the same run (can be triggered by multiple signals)
        try:
            if getattr(self, '_run_active', False):
                print("[Start] Run already active; ignoring duplicate start request")
                return
        except Exception:
            pass
        # Ensure the model result display window is visible when starting a run
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        # We treat the first 5 seconds as warmup; only measure after that.
        try:
            self.window_duration_sec = max(0.0, float(duration) - 5.0)
        except Exception:
            self.window_duration_sec = None
        
        # Reset idempotent stop flag for new run
        self._already_stopped = False
        try:
            pass
        except Exception:
            pass

        # Create a run_id and export to environment so child processes can log it
        try:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.environ["RUN_ID"] = self.run_id
        except Exception:
            self.run_id = ""
        
        # Initialize and start processes if they're not already running
        vid_proc = getattr(self, 'video_reader_proc', None)
        # Note: vid_proc can legitimately be None when no yolov4 models are scheduled.
        # Treat that as "not running" so we (re)initialize processes/threads for ResNet-only runs too.
        if (vid_proc is None) or (hasattr(vid_proc, 'is_alive') and not vid_proc.is_alive()):
            # Reset runtime state (events/queues/flags) for a fresh run after Stop
            try:
                self.initialize_state_variables()
                # Also clear handler references from previous run to aid GC
                for name in ['view1_handler','view2_handler','view3_handler','view4_handler','video_feeder','resnet_feeder']:
                    if hasattr(self, name):
                        try:
                            setattr(self, name, None)
                        except Exception:
                            pass
            except Exception as e:
                pass
            self.initialize_processes()
            self.initialize_threads()
        
        # Mark run as active only after processes/threads are initialized
        try:
            self._run_active = True
        except Exception:
            pass
        
        # Schedule a warmup window: reset all metrics after 5 seconds from start
        try:
            QTimer.singleShot(5000, self._begin_measurement_window)
        except Exception:
            pass
            
        # Schedule stopping execution after the specified duration (includes warmup)
        QTimer.singleShot(duration * 1000, self.timed_shutdown)
    
    def timed_shutdown(self):
        """Stop execution after the scheduled duration without closing the application."""
        print("Execution duration completed, stopping model execution...")
        try:
            self.stop_execution_async()
        except Exception:
            self.stop_execution()
        
    def stop_execution_async(self):
        """Start non-blocking stop of model execution and cleanup in a background thread."""
        if getattr(self, '_stop_in_progress', False):
            print("[Stop Execution] Async stop already in progress")
            return
        self._stop_in_progress = True
        def _run_stop():
            try:
                self.stop_execution()
            finally:
                self._stop_in_progress = False
                pass
        try:
            t = threading.Thread(target=_run_stop, name="StopExecutionThread", daemon=True)
            t.start()
        except Exception as e:
            print(f"[Stop Execution] Failed to start async stop thread: {e}")
            # Fallback to synchronous stop
            try:
                self.stop_execution()
            finally:
                self._stop_in_progress = False

    def shutdown_all_async(self):
        """Shut down entire application asynchronously to avoid freezing the UI."""
        if getattr(self, '_shutdown_in_progress', False):
            print("[Shutdown] Shutdown already in progress")
            return
        self._shutdown_in_progress = True
        def _run_shutdown():
            try:
                self.shutdown_all()
            finally:
                self._shutdown_in_progress = False
        try:
            t = threading.Thread(target=_run_shutdown, name="ShutdownThread", daemon=True)
            t.start()
        except Exception as e:
            print(f"[Shutdown] Failed to start shutdown thread: {e}")
            try:
                self.shutdown_all()
            finally:
                self._shutdown_in_progress = False
        
    def _begin_measurement_window(self):
        """Reset all per-view and feeder counters after warmup to start measurement."""
        try:
            import time as _t
            # Mark measurement window start (used for elapsed-based drop rate)
            self.measurement_start_ts = _t.time()
        except Exception:
            pass
        try:
            for name in ['view1_handler', 'view2_handler', 'view3_handler', 'view4_handler']:
                handler = getattr(self, name, None)
                if handler and hasattr(handler, 'reset_stats'):
                    handler.reset_stats()
        except Exception as e:
            print(f"[UnifiedViewer] Warmup reset warning (handlers): {e}")
        try:
            feeder = getattr(self, 'video_feeder', None)
            if feeder and hasattr(feeder, 'reset_counters'):
                feeder.reset_counters()
        except Exception as e:
            print(f"[UnifiedViewer] Warmup reset warning (feeder): {e}")
        try:
            rfeeder = getattr(self, 'resnet_feeder', None)
            if rfeeder and hasattr(rfeeder, 'reset_counters'):
                rfeeder.reset_counters()
        except Exception as e:
            print(f"[UnifiedViewer] Warmup reset warning (resnet feeder): {e}")
        
    def stop_execution(self):
        """Stop model execution without closing the application."""
        # If already stopped once for this run, skip duplicate work and saving
        if getattr(self, '_already_stopped', False):
            print("[Stop Execution] Already stopped for this run. Skipping duplicate save.")
            return
        # Mark as stopped to ensure idempotency
        self._already_stopped = True
        # Mark run inactive immediately to avoid duplicate starts racing during stop
        try:
            self._run_active = False
        except Exception:
            pass

        # Signal all threads (feeders/handlers) to stop
        try:
            if hasattr(self, 'shutdown_flag') and self.shutdown_flag:
                self.shutdown_flag.set()
        except Exception as e:
            print(f"[Stop Execution] Warning setting shutdown_flag: {e}")
        
        # Set shutdown events to signal workers to stop
        for name in ['view1_shutdown_event', 'view2_shutdown_event',
                     'view3_shutdown_event', 'view4_shutdown_event',
                     'video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
        # Headless shutdown events
        try:
            for ev in (getattr(self, 'headless_shutdown_events', {}) or {}).values():
                try:
                    ev.set()
                except Exception:
                    pass
        except Exception:
            pass
        
        # Gracefully stop all worker threads
        process_names = ['view1_process', 'view2_process', 'view3_process', 'view4_process', 'video_reader_proc']
        processes = [getattr(self, name, None) for name in process_names if hasattr(self, name) and getattr(self, name, None)]
        # Include headless processes
        try:
            processes.extend(list(getattr(self, 'headless_processes', []) or []))
        except Exception:
            pass
        try:
            pass
        except Exception:
            pass
        
        # Give threads time to exit their loops and run cleanup
        for p in processes:
            if p and p.is_alive():
                try:
                    p.join(timeout=3.0)
                except Exception as e:
                    print(f"[Stop Execution] Worker join warning: {e}")
        try:
            pass
        except Exception:
            pass
                    
        # Save throughput data once
        try:
            self.save_throughput_data()
        except Exception as e:
            print(f"[Stop Execution] Error saving throughput data: {e}")

        # After stopping, compute and insert average pre/post/load times at top when enabled
        try:
            record_time = int(os.environ.get("RECORD_TIME", "0"))
            if record_time == 1:
                self.save_pre_post_time_average()
        except Exception as e:
            print(f"[Stop Execution] Error saving pre/post timing averages: {e}")

        # After a schedule ends: drain all queues and explicitly close them
        try:
            self._drain_and_close_all_queues()
        except Exception as e:
            print(f"[Stop Execution] Queue cleanup error: {e}")
            
        try:
            # Hide the model result display window after stopping, per requirement
            # BUT: in executor-only mode (reused window), we keep it visible as requested
            if getattr(self, 'executor_only', False):
                print("[Stop Execution] Model execution stopped, display window kept visible for reuse")
            else:
                # Must execute on the Qt main thread to avoid macOS SIGTRAP from cross-thread UI calls
                from PyQt5.QtCore import QTimer as _QtTimer
                _QtTimer.singleShot(0, lambda: (self.hide(), print("[Stop Execution] Model execution stopped, display window hidden; Info window remains open")))
        except Exception:
            print("[Stop Execution] Model execution stopped (could not schedule hide)")
        
    def _drain_and_close_all_queues(self):
        """Drain all inter-process queues and explicitly close them.
        This prevents residual items from a finished schedule affecting the next run
        and ensures background feeder/handler threads do not leak resources.
        """
        def drain_queue(q):
            if not q:
                return
            try:
                # Non-blocking drain
                while True:
                    try:
                        _ = q.get_nowait()
                    except queue.Empty:
                        break
                    except (EOFError, BrokenPipeError, OSError):
                        break
                    except Exception:
                        # Keep draining on any unexpected item error
                        continue
            except Exception:
                pass
        # Enumerate all queues used in the viewer
        queue_names = [
            'video_frame_queue',
            'view1_frame_queue', 'view1_output_queue',
            'view2_frame_queue', 'view2_output_queue',
            'view3_frame_queue', 'view3_result_queue',
            'view4_frame_queue', 'view4_result_queue',
        ]
        for name in queue_names:
            q = getattr(self, name, None)
            drain_queue(q)
        # Drain headless queues
        try:
            for q in (getattr(self, 'headless_frame_queues', {}) or {}).values():
                drain_queue(q)
            for q in (getattr(self, 'headless_output_queues', {}) or {}).values():
                drain_queue(q)
        except Exception:
            pass
        
    def update_cpu_npu_usage(self):
        """Update CPU and NPU usage information."""
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]
        
        # Get performance statistics from view handlers if they exist
        view1_avg_fps = getattr(self, 'view1_handler', None).avg_fps if hasattr(self, 'view1_handler') else 0.0
        view1_avg_infer_time = getattr(self, 'view1_handler', None).avg_infer_time if hasattr(self, 'view1_handler') else 0.0
        
        view2_avg_fps = getattr(self, 'view2_handler', None).avg_fps if hasattr(self, 'view2_handler') else 0.0
        view2_avg_infer_time = getattr(self, 'view2_handler', None).avg_infer_time if hasattr(self, 'view2_handler') else 0.0
        
        view3_avg_fps = getattr(self, 'view3_handler', None).avg_fps if hasattr(self, 'view3_handler') else 0.0
        view3_avg_infer_time = getattr(self, 'view3_handler', None).avg_infer_time if hasattr(self, 'view3_handler') else 0.0
        
        view4_avg_fps = getattr(self, 'view4_handler', None).avg_fps if hasattr(self, 'view4_handler') else 0.0
        view4_avg_infer_time = getattr(self, 'view4_handler', None).avg_infer_time if hasattr(self, 'view4_handler') else 0.0
        
        # Determine which views are actually scheduled in this combination
        scheduled_views = [v for v in ["view1", "view2", "view3", "view4"] if v not in self.views_without_model]

        # Calculate total average FPS (total throughput) over scheduled views only
        per_view_fps = {
            "view1": view1_avg_fps,
            "view2": view2_avg_fps,
            "view3": view3_avg_fps,
            "view4": view4_avg_fps,
        }
        total_fps = sum(per_view_fps[v] for v in scheduled_views)
        scheduled_count = len(scheduled_views)
        total_avg_fps = total_fps / scheduled_count if scheduled_count > 0 else 0.0
        
        # Get model and execution mode for each view
        view1_model = self.model_settings.get("view1", {}).get("model", "")
        view1_mode = self.model_settings.get("view1", {}).get("execution", "cpu").upper()
        
        view2_model = self.model_settings.get("view2", {}).get("model", "")
        view2_mode = self.model_settings.get("view2", {}).get("execution", "cpu").upper()
        
        view3_model = self.model_settings.get("view3", {}).get("model", "")
        view3_mode = self.model_settings.get("view3", {}).get("execution", "cpu").upper()
        
        view4_model = self.model_settings.get("view4", {}).get("model", "")
        view4_mode = self.model_settings.get("view4", {}).get("execution", "cpu").upper()
        
        # Create performance text and append reallocation trigger status based on recent results
        # NOTE: Reallocation trigger condition checks are disabled per requirement.
        # The UI will display a fixed message to indicate triggers are disabled.
        name1 = "drop-rate surge"
        name2 = "Queueing-delay pressure"
        line1 = f"Reallocation trigger [{name1}]: Disabled"
        line2 = f"[{name2}]: Disabled"
        self.info_window.update_trigger_below_metrics(line1 + "\n" + line2)
        trigger_text = ""

        # Build per-view lines only for visible (non-hidden) scheduled views
        visible_views = [v for v in scheduled_views if v not in getattr(self, 'hidden_views', set())]
        per_view_lines = []
        if "view1" in visible_views:
            per_view_lines.append(
                f"<b>View1 ({view1_model} {view1_mode})</b> Avg FPS: {view1_avg_fps:.1f} (<span style='color: gray;'>{view1_avg_infer_time:.1f} ms</span>)"
            )
        if "view2" in visible_views:
            per_view_lines.append(
                f"<b><span style='color: purple;'>View2 ({view2_model} {view2_mode})</span></b> Avg FPS: <span style='color: purple;'>{view2_avg_fps:.1f}</span> (<span style='color: purple;'>{view2_avg_infer_time:.1f} ms</span>)"
            )
        if "view3" in visible_views:
            per_view_lines.append(
                f"<b><span style='color: green;'>View3 ({view3_model} {view3_mode})</span></b> Avg FPS: <span style='color: green;'>{view3_avg_fps:.1f}</span> (<span style='color: green;'>{view3_avg_infer_time:.1f} ms</span>)"
            )
        if "view4" in visible_views:
            per_view_lines.append(
                f"<b><span style='color: blue;'>View4 ({view4_model} {view4_mode})</span></b> Avg FPS: <span style='color: blue;'>{view4_avg_fps:.1f}</span> (<span style='color: blue;'>{view4_avg_infer_time:.1f} ms</span>)"
            )

        # Build a section listing models that are running headlessly (display: none)
        headless_lines = []
        try:
            for hid in list(getattr(self, 'headless_ids', []) or []):
                cfg = (self.model_settings or {}).get(hid, {})
                model_name = str(cfg.get('model', '') or '')
                exec_dev_raw = str(cfg.get('execution', 'cpu') or 'cpu').upper()
                # Normalize device naming to CPU/GPU/NPU (hide numeric suffixes)
                if exec_dev_raw.startswith('NPU'):
                    exec_dev = 'NPU'
                elif exec_dev_raw.startswith('GPU'):
                    exec_dev = 'GPU'
                else:
                    exec_dev = 'CPU'
                if model_name:
                    headless_lines.append(f"- {model_name} <span style='color: gray;'>({exec_dev})</span>")
        except Exception:
            pass
        
        # Compose performance text: visible per-view lines and then hidden models section (names must be shown even if no view)
        sections = [
            f"<b>Total Throughput: {total_fps:.1f} FPS</b>",
            f"<b>Total Average Throughput: {total_avg_fps:.1f} FPS</b>",
        ]
        if per_view_lines:
            sections.append("<br>".join(per_view_lines))
        if headless_lines:
            sections.append("<b>Models running without display</b><br>" + "<br>".join(headless_lines))
        performance_text = ("<br>".join(sections))
        
        # Create CPU info text
        cpu_info_text = (
            f"<b><span style='color: blue;'>CPU</span></b><br>"
            f"Usage: {current['CPU_Usage_percent']:.1f} %<br>"
            f"LoadAvg: {load1:.2f} / {load5:.2f} / {load15:.2f}<br>"
            f"CtxSwitches/sec: {delta_ctx} | Int/sec: {delta_int}"
        )
        
        # Create NPU info text
        npu_info_text = (
            f"<b><span style='color: green;'>NPU</span></b><br>"
            f"Usage: 42.0 %<br>"
            f"LoadAvg: 0.12 / 0.10 / 0.08<br>"
            f"CtxSwitches/sec: 12 | Int/sec: 3"
        )

        # Update info window labels
        self.info_window.update_model_performance(performance_text)
        self.info_window.update_cpu_info(cpu_info_text)
        # NPU panel removed from UI; skip updating NPU info label

        # Compute compact metrics line
        try:
            scheduled_views = [v for v in ["view1", "view2", "view3", "view4"] if v not in self.views_without_model]
            # per_view_stats fields aligned with _get_device_metrics_default expectations
            per_view_stats = {
                "view1": (view1_avg_fps, view1_avg_infer_time, getattr(self, 'view1_handler', None).infer_count if hasattr(self, 'view1_handler') else 0, view1_model, view1_mode, getattr(getattr(self, 'view1_handler', None), 'avg_wait_ms', 0.0), int(getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}).get('view1', 0))),
                "view2": (view2_avg_fps, view2_avg_infer_time, getattr(self, 'view2_handler', None).infer_count if hasattr(self, 'view2_handler') else 0, view2_model, view2_mode, getattr(getattr(self, 'view2_handler', None), 'avg_wait_ms', 0.0), int(getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}).get('view2', 0))),
                "view3": (view3_avg_fps, view3_avg_infer_time, getattr(self, 'view3_handler', None).infer_count if hasattr(self, 'view3_handler') else 0, view3_model, view3_mode, getattr(getattr(self, 'view3_handler', None), 'avg_wait_ms', 0.0), int(getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}).get('view3', 0))),
                "view4": (view4_avg_fps, view4_avg_infer_time, getattr(self, 'view4_handler', None).infer_count if hasattr(self, 'view4_handler') else 0, view4_model, view4_mode, getattr(getattr(self, 'view4_handler', None), 'avg_wait_ms', 0.0), int(getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}).get('view4', 0))),
            }
            devices_used = set(self.model_settings.get(v, {}).get('execution', 'CPU').upper() for v in scheduled_views)
            dev_metrics = self._get_device_metrics_default(devices_used, per_view_stats, scheduled_views)
            # choose max queue_wait_ms_p95 across devices
            q_waits = []
            for k, val in (dev_metrics or {}).items():
                try:
                    q_waits.append(float(val.get('queue_wait_ms_p95', 0.0) or 0.0))
                except Exception:
                    pass
            max_q_wait = max(q_waits) if q_waits else 0.0
            # Drop rate (frames/sec) across CNN and YOLO (exclude language models)
            # Sum drops from both feeders over all active ids (visible + headless)
            import time as _t
            active_ids = set(list(getattr(self, 'yolo_views', set()) or set())) | set(list(getattr(self, 'resnet_views', set()) or set()))
            # YOLO/video feeder drops
            vf = getattr(self, 'video_feeder', None)
            vf_map = getattr(vf, 'drop_counts', {}) if vf else {}
            vf_drops = sum(int(vf_map.get(v, 0) or 0) for v in active_ids)
            # ResNet/cnn feeder drops
            rf = getattr(self, 'resnet_feeder', None)
            rf_map = getattr(rf, 'drop_counts', {}) if rf else {}
            rf_drops = sum(int(rf_map.get(v, 0) or 0) for v in active_ids)
            total_drops = vf_drops + rf_drops
            # Normalize by elapsed measurement time since warm-up
            try:
                elapsed = float(max(0.001, (_t.time() - float(getattr(self, 'measurement_start_ts', 0.0)))) )
            except Exception:
                elapsed = float(self.window_duration_sec) if getattr(self, 'window_duration_sec', None) else 1.0
            drop_rate_fps = total_drops / elapsed if elapsed > 0 else 0.0
            score = total_fps - 0.2 * drop_rate_fps
            # Show each metric on its own line
            metrics_line = (
                f"Total: {total_fps:.2f} FPS\n"
                f"q95: {max_q_wait:.1f} ms\n"
                f"Drop: {drop_rate_fps:.2f} FPS\n"
                f"Score: {score:.2f}"
            )
            try:
                self.info_window.update_metrics(metrics_line)
            except Exception:
                pass
        except Exception as e:
            # Don't crash UI updates due to metrics calculation
            # print(f"[Viewer] metrics label update failed: {e}")
            pass
        
        # Update previous stats for next calculation
        self.prev_cpu_stats = current
    
    def save_throughput_data(self):
        """Save the current throughput of each model and the total throughput to a unique JSON under results/ starting with performance_."""
        try:
            # Prepare results directory and file path
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            results_path = getattr(self, 'results_path', None)
            if not results_path:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(results_dir, f"performance_{timestamp_str}.json")
                # Persist for potential subsequent calls within same viewer instance
                try:
                    self.results_path = results_path
                except Exception:
                    pass
            # Get model and execution mode for each view
            view1_model = self.model_settings.get("view1", {}).get("model", "")
            view1_mode = self.model_settings.get("view1", {}).get("execution", "cpu").upper()
            
            view2_model = self.model_settings.get("view2", {}).get("model", "resnet50_small")
            view2_mode = self.model_settings.get("view2", {}).get("execution", "cpu").upper()
            
            view3_model = self.model_settings.get("view3", {}).get("model", "")
            view3_mode = self.model_settings.get("view3", {}).get("execution", "cpu").upper()
            
            view4_model = self.model_settings.get("view4", {}).get("model", "resnet50_small")
            view4_mode = self.model_settings.get("view4", {}).get("execution", "cpu").upper()
            
            # Get performance statistics from view handlers (robust to missing handlers)
            def _stats_for(view_name: str):
                handler = getattr(self, f"{view_name}_handler", None)
                if handler is None:
                    return 0.0, 0.0, 0
                try:
                    avg_fps = float(getattr(handler, 'avg_fps', 0.0) or 0.0)
                except Exception:
                    avg_fps = 0.0
                try:
                    avg_infer_time = float(getattr(handler, 'avg_infer_time', 0.0) or 0.0)
                except Exception:
                    avg_infer_time = 0.0
                try:
                    infer_count = int(getattr(handler, 'infer_count', 0) or 0)
                except Exception:
                    infer_count = 0
                return avg_fps, avg_infer_time, infer_count

            view1_avg_fps, view1_avg_infer_time, view1_infer_count = _stats_for('view1')
            view2_avg_fps, view2_avg_infer_time, view2_infer_count = _stats_for('view2')
            view3_avg_fps, view3_avg_infer_time, view3_infer_count = _stats_for('view3')
            view4_avg_fps, view4_avg_infer_time, view4_infer_count = _stats_for('view4')
            
            # Determine which views are actually scheduled in this combination
            scheduled_views = [v for v in ["view1", "view2", "view3", "view4"] if v not in self.views_without_model]

            # Map helpers for per-view stats, including avg_wait_ms (if available) and dropped frames
            view1_wait = getattr(getattr(self, 'view1_handler', None), 'avg_wait_ms', 0.0)
            view2_wait = getattr(getattr(self, 'view2_handler', None), 'avg_wait_ms', 0.0)
            view3_wait = getattr(getattr(self, 'view3_handler', None), 'avg_wait_ms', 0.0)
            view4_wait = getattr(getattr(self, 'view4_handler', None), 'avg_wait_ms', 0.0)

            # Drop counts from feeders (0 if not present)
            drop_map = {}
            # Check video_feeder (YOLO)
            v_feeder = getattr(self, 'video_feeder', None)
            if v_feeder:
                v_drops = getattr(v_feeder, 'drop_counts', {})
                for v_name, count in v_drops.items():
                    drop_map[v_name] = drop_map.get(v_name, 0) + count
            # Check resnet_feeder (ResNet)
            r_feeder = getattr(self, 'resnet_feeder', None)
            if r_feeder:
                r_drops = getattr(r_feeder, 'drop_counts', {})
                for v_name, count in r_drops.items():
                    drop_map[v_name] = drop_map.get(v_name, 0) + count

            per_view_stats = {
                "view1": (view1_avg_fps, view1_avg_infer_time, view1_infer_count, view1_model, view1_mode, view1_wait, int(drop_map.get("view1", 0))),
                "view2": (view2_avg_fps, view2_avg_infer_time, view2_infer_count, view2_model, view2_mode, view2_wait, int(drop_map.get("view2", 0))),
                "view3": (view3_avg_fps, view3_avg_infer_time, view3_infer_count, view3_model, view3_mode, view3_wait, int(drop_map.get("view3", 0))),
                "view4": (view4_avg_fps, view4_avg_infer_time, view4_infer_count, view4_model, view4_mode, view4_wait, int(drop_map.get("view4", 0))),
            }

            # Calculate total throughput for scheduled views (with headless later)
            total_fps = sum(per_view_stats[v][0] for v in scheduled_views)
            scheduled_count = len(scheduled_views)
            total_avg_fps = total_fps / scheduled_count if scheduled_count > 0 else 0.0

            # Prepare throughput data including all scheduled views (even if 0 inferences)
            throughput_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "window_sec": float(self.window_duration_sec) if self.window_duration_sec is not None else None,
                "combination": self.current_combination,
                "models": {},
                "total": {
                    "total_throughput_fps": round(total_fps, 2),
                    "avg_throughput_fps": round(total_avg_fps, 2)
                }
            }
            
            # Add all scheduled views to the models dictionary (include zeros if no inferences)
            devices_used = set()
            for v in scheduled_views:
                avg_fps, avg_time, infer_cnt, model_name, exec_mode, avg_wait_ms, dropped = per_view_stats[v]
                throughput_data["models"][v] = {
                    "model": model_name,
                    "execution": exec_mode,
                    "throughput_fps": round(avg_fps, 2),
                    "avg_inference_time_ms": round(avg_time, 2),
                    "inference_count": int(infer_cnt),
                    "avg_wait_to_preprocess_ms": round(avg_wait_ms or 0.0, 2),
                    "dropped_frames_due_to_full_queue": int(dropped or 0)
                }
                devices_used.add(exec_mode)

            # Include headless jobs (no views) into models and totals
            try:
                headless_ids = list(getattr(self, 'headless_ids', []) or [])
                hstats = getattr(self, 'headless_stats', {}) or {}
                # elapsed time since measurement start; fallback to window duration
                try:
                    import time as _t
                    elapsed = float(max(0.001, (_t.time() - float(getattr(self, 'measurement_start_ts', 0.0)))))
                except Exception:
                    elapsed = float(self.window_duration_sec) if getattr(self, 'window_duration_sec', None) else 1.0
                for hid in headless_ids:
                    cfg = self.model_settings.get(hid, {})
                    model_name = cfg.get('model', '')
                    exec_mode = str(cfg.get('execution', 'cpu')).upper()
                    s = hstats.get(hid, {})
                    count = int(s.get('count', 0) or 0)
                    sum_infer_ms = float(s.get('sum_infer_ms', 0.0) or 0.0)
                    sum_wait_ms = float(s.get('sum_wait_ms', 0.0) or 0.0)
                    wait_count = int(s.get('wait_count', 0) or 0)
                    avg_time = (sum_infer_ms / count) if count > 0 else 0.0
                    avg_wait_ms = (sum_wait_ms / wait_count) if wait_count > 0 else 0.0
                    avg_fps = (count / elapsed) if elapsed > 0 else 0.0
                    # Update totals (headless contributes to total throughput)
                    total_fps += avg_fps
                    # Models entry uses the headless id as key
                    throughput_data["models"][hid] = {
                        "model": model_name,
                        "execution": exec_mode,
                        "throughput_fps": round(avg_fps, 2),
                        "avg_inference_time_ms": round(avg_time, 2),
                        "inference_count": int(count),
                        "avg_wait_to_preprocess_ms": round(avg_wait_ms or 0.0, 2),
                        "dropped_frames_due_to_full_queue": 0
                    }
                    devices_used.add(exec_mode)
                # Recompute average throughput across all scheduled entities (views + headless)
                total_entities = scheduled_count + len(headless_ids)
                total_avg_fps = total_fps / total_entities if total_entities > 0 else 0.0
                # Reflect updated totals
                throughput_data["total"]["total_throughput_fps"] = round(total_fps, 2)
                throughput_data["total"]["avg_throughput_fps"] = round(total_avg_fps, 2)
            except Exception as e:
                try:
                    print(f"[Save Throughput] Warning: failed to include headless metrics: {e}")
                except Exception:
                    pass

            # Compute per-device queue metrics with fallback when timing logs are unavailable
            try:
                device_metrics = self._get_device_metrics_default(devices_used, per_view_stats, scheduled_views)
                throughput_data["devices"] = device_metrics
            except Exception as e:
                print(f"[Save Throughput] Warning: failed to compute device metrics: {e}")
            
            # Determine if the current combination is the first schedule in the YAML
            # BUT: if results_path was explicitly set by an external executor, we should ALWAYS append
            # to preserve its initialization (e.g. ScheduleExecutor sets it to '[]' at start).
            is_first_schedule = False
            external_executor = hasattr(self, 'executor_only') and self.executor_only

            if not external_executor:
                try:
                    with open(self.schedule_file, "r", encoding="utf-8") as sf:
                        cfg = yaml.safe_load(sf) or {}
                    if cfg:
                        first_key = next(iter(cfg))
                        is_first_schedule = (self.current_combination == first_key)
                except Exception as e:
                    # If we cannot read the YAML, default to not-first to avoid accidental truncation
                    print(f"[Save Throughput] Warning: failed to read schedule file {self.schedule_file}: {e}")
                    is_first_schedule = False
            else:
                # In executor mode, ScheduleExecutor handles file initialization; viewer just appends.
                is_first_schedule = False

            # Prepare aggregated results list. If this is the first schedule, clear previous contents.
            results = []
            if not is_first_schedule:
                try:
                    if os.path.exists(results_path):
                        with open(results_path, "r", encoding="utf-8") as rf:
                            loaded = json.load(rf)
                            if isinstance(loaded, list):
                                results = loaded
                            elif isinstance(loaded, dict):
                                # Backward compatibility: wrap single dict into list
                                results = [loaded]
                            else:
                                results = []
                except Exception as read_err:
                    print(f"[Save Throughput] Warning: failed to read existing results file: {read_err}. Starting new list.")
                    results = []
            else:
                # Explicitly clear previous results when saving the first schedule
                print("[Save Throughput] First schedule detected. Starting a new results file.")

            results.append(throughput_data)

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            print(f"[Shutdown] Throughput data saved to {results_path}")
        except Exception as e:
            print(f"[Shutdown ERROR] Failed to save throughput data: {e}")

    def _get_device_metrics_default(self, devices_used, per_view_stats, scheduled_views):
        """Return device metrics using timing logs when available, otherwise fallback to estimates.
        devices_used: set of exec modes like {"CPU","NPU0","NPU1"}
        per_view_stats: dict view -> tuple(..., exec_mode, avg_wait_ms, dropped)
        scheduled_views: list of views considered in this window.
        """
        # Compute detailed metrics from timing logs (may be partial)
        detailed = self._compute_device_metrics(devices_used)

        # Always compute fallback estimates from per-view average waits
        dev_waits = {}
        for v in scheduled_views:
            try:
                _, _, _, _, exec_mode, avg_wait_ms, _ = per_view_stats[v]
            except Exception:
                exec_mode = None
                avg_wait_ms = 0.0
            if not exec_mode:
                continue
            dev_waits.setdefault(exec_mode, []).append(float(avg_wait_ms or 0.0))
        fallback = {}
        for dev in devices_used:
            key = str(dev).lower()
            waits = dev_waits.get(dev, [])
            max_wait = max(waits) if waits else 0.0
            entry = {"queue_wait_ms_p95": round(max_wait, 2)}
            if str(dev).upper().startswith("NPU"):
                entry["t_preload_ms"] = 0.0
            fallback[key] = entry

        # If no detailed metrics at all, return fallback
        if not isinstance(detailed, dict) or not detailed:
            return fallback

        # Merge device-wise: prefer detailed only if it has meaningful values; otherwise use fallback
        merged = {}
        for dev in devices_used:
            key = str(dev).lower()
            det = detailed.get(key)
            fb = fallback.get(key, {"queue_wait_ms_p95": 0.0})
            if str(dev).upper().startswith("NPU"):
                fb.setdefault("t_preload_ms", 0.0)
            if isinstance(det, dict):
                q = float(det.get("queue_wait_ms_p95", 0.0) or 0.0)
                tpre = float(det.get("t_preload_ms", 0.0) or 0.0)
                if (q > 0.0) or (tpre > 0.0):
                    merged[key] = det
                else:
                    merged[key] = fb
            else:
                merged[key] = fb
        return merged

    def _compute_device_metrics(self, devices_used):
        """Compute per-device queue p95 and NPU preload time from timing logs for current run.
        devices_used: set like {"CPU", "NPU0", "NPU1"}
        Returns dict like {"cpu": {"queue_wait_ms_p95": x}, "npu0": {"queue_wait_ms_p95": y, "t_preload_ms": z}, ...}
        """
        try:
            path = "result_pre_post_time.json"
            if not os.path.exists(path):
                return {}
            # Load JSON lines
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            import json as _json
            records = []
            for ln in lines:
                try:
                    records.append(_json.loads(ln))
                except Exception:
                    continue
            if not records:
                return {}
            # Filter by run_id if available
            run_id = getattr(self, 'run_id', os.environ.get('RUN_ID', ''))
            if run_id:
                records_for_run = [r for r in records if r.get("run_id", "") == run_id]
            else:
                records_for_run = list(records)
            # If there are no records for the current run, we will still try to provide fallback t_preload_ms
            # by looking at the most recent model_load entries across all records below.
            # However, queue_waits will be empty in that case.
            # Group waits by device label
            def _norm_dev(d):
                if not d:
                    return None
                d = str(d).upper()
                if d.startswith("NPU"):
                    # keep exact like NPU0/NPU1
                    return d
                return "CPU"
            waits = {}
            for r in records_for_run:
                if r.get("kind") == "inference":
                    dev = _norm_dev(r.get("device"))
                    if dev not in devices_used:
                        continue
                    w = r.get("wait_to_preprocess_ms")
                    if isinstance(w, (int, float)):
                        waits.setdefault(dev, []).append(float(w))
            # Percentile helper
            def p95(vals):
                if not vals:
                    return 0.0
                vs = sorted(vals)
                # nearest-rank method
                import math
                k = max(1, int(math.ceil(0.95 * len(vs))))
                return float(vs[k - 1])
            # Collect preload times for NPUs
            # First, from current run (records_for_run)
            preload_current = {}
            for r in records_for_run:
                if r.get("kind") == "model_load":
                    dev = _norm_dev(r.get("device"))
                    if dev and dev.startswith("NPU") and dev in devices_used:
                        t = r.get("npu_memory_load_time_ms")
                        if isinstance(t, (int, float)):
                            preload_current.setdefault(dev, []).append(float(t))
            # Second, prepare a latest-across-all-records fallback per device
            latest_preload_all = {}
            latest_ts_all = {}
            for r in records:
                if r.get("kind") == "model_load":
                    dev = _norm_dev(r.get("device"))
                    if not (dev and dev.startswith("NPU") and dev in devices_used):
                        continue
                    t = r.get("npu_memory_load_time_ms")
                    if not isinstance(t, (int, float)):
                        continue
                    ts = r.get("timestamp")
                    try:
                        from datetime import datetime as _dt
                        tsv = _dt.strptime(ts, "%Y-%m-%d %H:%M:%S") if ts else None
                    except Exception:
                        tsv = None
                    # Track the most recent timestamp for this device
                    prev = latest_ts_all.get(dev)
                    if prev is None or (tsv and prev and tsv > prev) or (tsv and prev is None):
                        latest_ts_all[dev] = tsv
                        latest_preload_all[dev] = float(t)
            # Build result dict
            result = {}
            for dev in devices_used:
                key = dev.lower()  # CPU -> cpu, NPU0 -> npu0
                res = {"queue_wait_ms_p95": round(p95(waits.get(dev, [])), 2)}
                if dev.startswith("NPU"):
                    # Priority: current-run value -> latest-across-file -> environment -> 0.0
                    val = None
                    vals_cur = preload_current.get(dev, [])
                    if vals_cur:
                        val = float(vals_cur[0])
                    if val is None:
                        val = latest_preload_all.get(dev)
                    if val is None:
                        # Environment variable fallback: NPU0_PRELOAD_MS, NPU1_PRELOAD_MS
                        try:
                            val = float(os.environ.get(f"{dev}_PRELOAD_MS", "nan"))
                        except Exception:
                            val = None
                        if val is not None and (val != val):  # NaN check
                            val = None
                    res["t_preload_ms"] = round(val, 2) if isinstance(val, (int, float)) else 0.0
                result[key] = res
            return result
        except Exception as e:
            print(f"[Device Metrics] Failed to compute device metrics: {e}")
            return {}

    def _evaluate_drop_rate_trigger(self, threshold_fps=30.0, min_windows=2):
        """
        Evaluate the drop rate trigger condition at runtime every fixed redeploy window.
        NOTE: Disabled per requirement; returns (None, 'disabled').
        Rules:
        - redeploy_window_sec is fixed to 3 seconds.
        - Do NOT read any files. Use in-memory drop counters from VideoFeeder.
        - Condition: in two consecutive 3-second windows, total drop rate (drops/sec) > threshold_fps.
        Returns:
            (status, details)
            - status: True if trigger satisfied, False if not, None if insufficient data.
            - details: short text with computed rates for the last two windows.
        """
        # Disabled: immediate no-decision
        return None, "disabled"

    def _evaluate_queue_delay_trigger(self, threshold_val=35.0, min_windows=2):
        """
        Evaluate Queueing-delay pressure at runtime per 3s window.
        NOTE: Disabled per requirement; returns (None, 'disabled').
        """
        return None, "disabled"

    def save_pre_post_time_average(self):
        """Compute averages for the last run_id and insert a summary line at the top of result_pre_post_time.json (JSON Lines)."""
        try:
            run_id = getattr(self, 'run_id', os.environ.get('RUN_ID', ''))
            path = "result_pre_post_time.json"
            if not os.path.exists(path):
                print("[Timing] No timing file to summarize.")
                return
            # Read all lines
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                print("[Timing] Timing file empty.")
                return
            import json as _json
            records = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = _json.loads(ln)
                    records.append(rec)
                except Exception:
                    continue
            if not records:
                print("[Timing] No valid records to summarize.")
                return
            # Filter for last run_id; if missing, infer last by max timestamp
            target_run = run_id
            if not target_run:
                # fallback: determine the most recent run_id appearing in records by timestamp
                # build map run_id -> latest timestamp
                from datetime import datetime as _dt
                latest_by_run = {}
                for r in records:
                    rid = r.get("run_id", "")
                    ts = r.get("timestamp")
                    try:
                        tsv = _dt.strptime(ts, "%Y-%m-%d %H:%M:%S") if ts else _dt.min
                    except Exception:
                        tsv = _dt.min
                    if rid not in latest_by_run or tsv > latest_by_run[rid]:
                        latest_by_run[rid] = tsv
                if latest_by_run:
                    target_run = max(latest_by_run, key=lambda k: latest_by_run[k])
                else:
                    target_run = ""
            run_records = [r for r in records if r.get("run_id", "") == target_run] if target_run else records
            if not run_records:
                print("[Timing] No records for the current run.")
                return
            # Compute averages
            import math
            def _avg(vals):
                vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
                return round(sum(vals) / len(vals), 3) if vals else 0.0
            pre_list = [r.get("preprocess_time_ms") for r in run_records if r.get("kind") == "inference"]
            infer_list = [r.get("inference_time_ms") for r in run_records if r.get("kind") == "inference"]
            post_list = [r.get("postprocess_time_ms") for r in run_records if r.get("kind") == "inference"]
            load_list = [r.get("model_load_time_ms") for r in run_records if r.get("kind") == "model_load"]
            npu_mem_load_list = [r.get("npu_memory_load_time_ms") for r in run_records if r.get("kind") == "model_load"]

            summary = {
                "type": "average_summary",
                "run_id": target_run,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "averages": {
                    "preprocess_time_ms": _avg(pre_list),
                    "inference_time_ms": _avg(infer_list),
                    "postprocess_time_ms": _avg(post_list),
                    "model_load_time_ms": _avg(load_list),
                    "npu_memory_load_time_ms": _avg(npu_mem_load_list),
                }
            }
            # Prepend as the first line
            new_lines = [json.dumps(summary, ensure_ascii=False) + "\n"] + lines
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print("[Timing] Average summary inserted at the top of result_pre_post_time.json")
        except Exception as e:
            print(f"[Timing ERROR] Failed to compute/save averages: {e}")