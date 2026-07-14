"""
Main UnifiedViewer class for the multimodel scheduling application.
"""
import os
import json
import signal
import yaml
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QLabel, QWidget, QVBoxLayout, QFileDialog,
                             QPushButton)
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic
from multiprocessing import Process, Queue, Event

# Globally suppress noisy BrokenPipeError/"handle is closed" tracebacks from background threads
try:
    import threading, sys
    _orig_exhook = getattr(threading, 'excepthook', None)
    def _suppress_broken_pipe_excepthook(args):
        try:
            e = args.exc_value
            msg = str(e).lower()
            if isinstance(e, BrokenPipeError) or ('broken pipe' in msg) or ('handle is closed' in msg):
                return  # swallow silently
        except Exception:
            pass
        try:
            if _orig_exhook:
                return _orig_exhook(args)
        except Exception:
            pass
        try:
            sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            pass
    try:
        threading.excepthook = _suppress_broken_pipe_excepthook
    except Exception:
        pass
except Exception:
    pass

# Import local modules
from utils import get_cpu_metrics
from view_handlers import (
    ModelSignals, YoloViewHandler, ResNetViewHandler, LLMViewHandler,
    VideoFeeder, ResnetImageFeeder,
)
from model_processors import (
    video_reader_process,
    run_detection_process,
    run_classification_process,
    run_llm_process,
    run_vlm_process,
)
import model_registry as reg

# Repo root, resolved from this file rather than the cwd. The .ui files and the
# sample video live here; resolving them relatively meant that launching the app
# from any other directory killed it on startup with a bare FileNotFoundError.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Demo input defaults. Every one of these has a value, so pressing the button with
# nothing configured still runs a full demo. The GUI's "Configure Inputs.." dialog
# overrides them through the environment.
IMAGENET_DIR = os.environ.get("DEMO_IMAGE_DIR", "imagenet-sample-images")
DEMO_VIDEO = os.environ.get("DEMO_VIDEO", "stockholm_1280x720.mp4")
DEMO_LLM_PROMPT = os.environ.get("DEMO_LLM_PROMPT") or None   # None -> engine defaults
DEMO_VLM_PROMPT = os.environ.get("DEMO_VLM_PROMPT") or None


def _asset(name: str) -> str:
    return name if os.path.isabs(name) else os.path.join(_HERE, name)

# Only these models are rendered. Everything else in the deployment still loads,
# infers and is measured exactly as before -- it simply gets no view. The single
# source of truth: do not test model names for "renderability" anywhere else.
VISUALIZABLE_MODELS = {"qwen2_vl", "resnet50", "yolo11s", "llama1b"}

# The .ui provides four QLabel slots and four display signals. A schedule may
# activate more concurrent models than that; the extra views run headless —
# they execute and contribute to the statistics, they just aren't rendered.
DISPLAY_SLOTS = ["view1", "view2", "view3", "view4"]
MAX_VIEWS = 8
# Colors cycled through when rendering the per-view performance summary.
VIEW_COLORS = ["gray", "purple", "green", "blue", "darkorange", "brown", "teal", "magenta"]

class InfoWindow(QWidget):
    """Main window for displaying system and model information."""
    
    def __init__(self, parent=None):
        """Initialize the InfoWindow."""
        super().__init__()
        # Load UI from file instead of creating components programmatically
        uic.loadUi(_asset("info_window.ui"), self)
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
        # Call the parent's stop_execution method if available
        if self.parent and hasattr(self.parent, 'stop_execution'):
            try:
                self.parent.stop_execution()
            except Exception:
                pass
            # Also close the viewer window so it disappears immediately on Ubuntu
            try:
                if hasattr(self.parent, 'close'):
                    self.parent.close()
            except Exception:
                pass
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
        """Handle window close event - terminate the application."""
        print("[InfoWindow] Close event triggered - terminating application")
        event.accept()
        
        # If we have a parent (UnifiedViewer), call its shutdown method
        if self.parent and hasattr(self.parent, 'shutdown_all'):
            self.parent.shutdown_all()
        else:
            # If no parent, exit directly
            print("[InfoWindow] Closing application directly")
            os._exit(0)

class UnifiedViewer(QMainWindow):
    """Main viewer class for the multimodel scheduling application."""
    
    def __init__(self, schedule_file='model_schedules.yaml', combination_name=None, info_window=None):
        """Initialize the UnifiedViewer.
        
        Args:
            schedule_file (str): Path to the model scheduling information file.
            combination_name (str|None): Specific combination key to use from the YAML. If None, default logic applies.
        """
        super().__init__()
        uic.loadUi(_asset("schedule_executor_display.ui"), self)
        
        # Set up signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)

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
        else:
            self.info_window = InfoWindow(parent=self)
        # Show the info window only when not marked as headless/hidden
        try:
            if not getattr(self.info_window, 'hidden_headless', False):
                self.info_window.show()
        except Exception:
            # Fallback to show to preserve legacy behavior if attribute missing
            try:
                self.info_window.show()
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
    
    def initialize_model_settings(self):
        """Initialize model settings from YAML configuration."""
        self.model_settings = {}
        self.views_without_model = set()  # Track views without specified models
        # Default combination, can be overridden by requested_combination
        self.current_combination = self.requested_combination or "combination1"
        
        # Update the schedule name label
        self.info_window.update_schedule_name(f"Current Schedule: {self.current_combination}")
        
        try:
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
            if self.current_combination in config:
                for model_config_name, model_config in (config[self.current_combination] or {}).items():
                    if isinstance(model_config, dict) and "display" in model_config:
                        view_name = model_config.get("display")
                        if view_name:
                            view_to_model_map[view_name] = {
                                "model": model_config.get("model", ""),
                                "execution": model_config.get("execution", "cpu"),
                                "infps": model_config.get("infps", None)
                            }
            
            # Views actually scheduled, in numeric order (view1..viewN, N <= MAX_VIEWS).
            def _view_index(v):
                digits = ''.join(ch for ch in v if ch.isdigit())
                return int(digits) if digits else 0

            scheduled = sorted(view_to_model_map.keys(), key=_view_index)[:MAX_VIEWS]
            self.view_names = scheduled
            for view in scheduled:
                self.model_settings[view] = view_to_model_map[view]

            # Display slots with no model just show the placeholder image.
            for view in DISPLAY_SLOTS:
                if view not in view_to_model_map:
                    self.views_without_model.add(view)
                    print(f"[UnifiedViewer] {view} not used in this combination (no model assigned) [{os.path.basename(self.schedule_file)}]")
            print(f"[UnifiedViewer] Loaded model settings from {self.schedule_file} for {self.current_combination}")
            self._assign_display_slots()
        except Exception as e:
            print(f"[UnifiedViewer ERROR] Failed to load {self.schedule_file}: {e}")
            # Set default settings if file loading fails
            self.model_settings = {
                "view1": {"model": "yolo11s", "execution": "cpu"},
                "view2": {"model": "resnet50", "execution": "cpu"},
                "view3": {"model": "yolo11s", "execution": "cpu"},
                "view4": {"model": "resnet50", "execution": "cpu"}
            }
            self.view_names = list(DISPLAY_SLOTS)
            self._assign_display_slots()

    def _assign_display_slots(self):
        """Decide which scheduled views get a display slot: whitelist only, max four.

        This is purely a rendering decision. Every scheduled view still gets a worker
        process, still runs inference and still feeds the throughput / drop-rate /
        score figures -- a model that is not visualizable is not excluded from the
        deployment, it just has no tile.
        """
        self.display_slot_of = {}
        skipped = []

        # Data collection runs with rendering off: drawing costs CPU in the same
        # process that feeds the workers, and the whitelist means only some models in
        # a set pay it -- so leaving it on puts a per-model rendering cost into the
        # training labels that has nothing to do with placement. The demo keeps it on.
        if os.environ.get("DISABLE_VISUALIZATION", "0") == "1":
            print("[UnifiedViewer] DISABLE_VISUALIZATION=1 -- rendering off; "
                  "all views run headless (execution and measurement unchanged).")
            return

        for view in self.view_names:
            model = (self.model_settings.get(view, {}) or {}).get("model", "")
            if model in VISUALIZABLE_MODELS:
                if len(self.display_slot_of) < len(DISPLAY_SLOTS):
                    self.display_slot_of[view] = DISPLAY_SLOTS[len(self.display_slot_of)]
                else:
                    skipped.append(f"{model} ({view}, no slot left)")
            else:
                skipped.append(f"{model} ({view}, not visualizable)")

        shown = [f"{self.model_settings[v]['model']} -> {slot}"
                 for v, slot in self.display_slot_of.items()]
        print(f"[UnifiedViewer] Visualizing {len(shown)}/{len(self.view_names)} views: "
              f"{', '.join(shown) or 'none'}")
        if skipped:
            print(f"[UnifiedViewer] Running without a view (still executed and measured): "
                  f"{', '.join(skipped)}")
        if not self.display_slot_of:
            running = ", ".join((self.model_settings.get(v, {}) or {}).get('model', '?')
                                for v in self.view_names)
            print(f"[UnifiedViewer] No visualizable models in this deployment. "
                  f"Running: {running}. Execution and measurement continue as normal; "
                  f"the whitelist is: {', '.join(sorted(VISUALIZABLE_MODELS))}.")

    def initialize_ui_components(self):
        """Initialize UI components."""
        self.view1 = self.findChild(QLabel, "view1")
        self.view2 = self.findChild(QLabel, "view2")
        self.view3 = self.findChild(QLabel, "view3")
        self.view4 = self.findChild(QLabel, "view4")

        # Stop All: the demo needs one obvious way to halt everything on stage.
        self.stop_all_button = self.findChild(QPushButton, "stop_all_button")
        if self.stop_all_button is not None:
            self.stop_all_button.clicked.connect(self.on_stop_all_clicked)

        self._layout_visible_views()

        # Define and connect signals
        self.model_signals = ModelSignals()
        self.model_signals.update_view1_display.connect(self.update_view1_display)
        self.model_signals.update_view2_display.connect(self.update_view2_display)
        self.model_signals.update_view3_display.connect(self.update_view3_display)
        self.model_signals.update_view4_display.connect(self.update_view4_display)
    
    def _layout_visible_views(self):
        """Pack the used tiles, and cross out any cell the grid leaves over.

        A single view sitting in one corner of a 2x2 grid with three dead cells looks
        broken on stage, so the grid is re-flowed to the number of views: 1 -> one
        tile, 2 -> side by side, 3-4 -> 2x2. Re-flowing can still leave one cell over
        (three views in a 2x2), and that cell gets the X image rather than being left
        as a blank patch of window -- an empty tile should look deliberately empty.
        """
        from PyQt5.QtWidgets import QGridLayout
        from utils import create_x_image, convert_cv_to_qt
        grid = self.findChild(QGridLayout, "videoLayout")
        used = [DISPLAY_SLOTS[i] for i in range(len(getattr(self, 'display_slot_of', {}) or {}))]

        if grid is None:
            print("[UnifiedViewer] videoLayout not found; leaving the .ui layout as-is.")
            return

        for slot in DISPLAY_SLOTS:
            widget = getattr(self, slot, None)
            if widget is None:
                continue
            try:
                grid.removeWidget(widget)
                widget.setVisible(False)
            except Exception as e:
                print(f"[Layout] {slot}: {e}")

        if not used:
            self._show_no_visualizable_message(grid)
            return

        cols = 1 if len(used) <= 1 else 2
        rows = (len(used) + cols - 1) // cols
        for i, slot in enumerate(used):
            widget = getattr(self, slot, None)
            if widget is not None:
                widget.setVisible(True)
                grid.addWidget(widget, i // cols, i % cols)

        # Cells the re-flow could not fill: show the X placeholder in the spare slots.
        spare = [s for s in DISPLAY_SLOTS if s not in used]
        for cell in range(len(used), rows * cols):
            if not spare:
                break
            slot = spare.pop(0)
            widget = getattr(self, slot, None)
            if widget is None:
                continue
            try:
                pixmap = convert_cv_to_qt(create_x_image())
                if not pixmap.isNull():
                    widget.setPixmap(pixmap)
                    widget.setScaledContents(True)
                widget.setVisible(True)
                grid.addWidget(widget, cell // cols, cell % cols)
            except Exception as e:
                print(f"[Layout] X placeholder for {slot}: {e}")

    def _show_no_visualizable_message(self, grid):
        """Say why the window is empty instead of showing a black one.

        The deployment is still running and still being measured -- the window just has
        nothing it is allowed to draw. Leaving it blank would read as a crash.
        """
        running = ", ".join(sorted(
            (self.model_settings.get(v, {}) or {}).get('model', '?') for v in self.view_names))
        message = QLabel(
            "No visualizable models in this deployment.\n\n"
            f"Running (executing and being measured): {running or 'nothing'}\n"
            f"Visualizable models: {', '.join(sorted(VISUALIZABLE_MODELS))}\n\n"
            "Throughput, drop rate and score are unaffected.")
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        message.setStyleSheet("font-size: 16px; padding: 24px; color: #ddd; background: #202020;")
        self.no_visual_label = message
        grid.addWidget(message, 0, 0)

    def initialize_state_variables(self):
        """Initialize state variables."""
        # Global flag for signaling threads to exit
        self.global_exit_flag = False
        
        # Initialize common state variables
        self.shutdown_flag = Event()
        self.prev_cpu_stats = get_cpu_metrics(interval=0)
        
        # Ensure stop_execution is idempotent: save throughput only once per schedule
        self._already_stopped = False

        # Store the requested execution window duration (seconds) for saving into results
        self.window_duration_sec = None

        # Initialize queues and events
        self.video_frame_queue = Queue(maxsize=10)
        self.video_shutdown_event = Event()

        # Per-view queues and events, created for every scheduled view (any N).
        # Both `_output_queue` and `_result_queue` name the same queue so all
        # existing lookups resolve regardless of which alias they use.
        if not getattr(self, 'view_names', None):
            self.view_names = list(DISPLAY_SLOTS)
        for v in self.view_names:
            result_queue = Queue(maxsize=10)
            setattr(self, f"{v}_frame_queue", Queue(maxsize=10))
            setattr(self, f"{v}_output_queue", result_queue)
            setattr(self, f"{v}_result_queue", result_queue)
            setattr(self, f"{v}_shutdown_event", Event())

        # Track which views need which feeder / handler.
        self.yolo_views = set()    # detection: video frames
        self.resnet_views = set()  # classification: image feeder
        self.vlm_views = set()     # VLM: video frames + streamed text
        self.llm_views = set()     # LLM: no feeder (prompt-driven), streamed text
    
    def initialize_processes(self):
        """Initialize and start model processes."""
        # Start video reader process
        self.video_reader_proc = Process(
            target=video_reader_process,
            args=(_asset(DEMO_VIDEO), self.video_frame_queue, self.video_shutdown_event),
            daemon=True
        )
        self.video_reader_proc.start()
        
        # Start a worker process for every scheduled view
        for v in self.view_names:
            self.start_view_process(v)
    
    def start_view_process(self, view_name):
        """
        Start a process for a specific view.
        
        Args:
            view_name: Name of the view (view1, view2, etc.)
        """
        # If no model is assigned for this view, do not start any process
        if hasattr(self, 'views_without_model') and view_name in self.views_without_model:
            print(f"[UnifiedViewer] Skipping process start for {view_name}: no model assigned")
            return
        
        model = self.model_settings.get(view_name, {}).get("model", "")
        execution = self.model_settings.get(view_name, {}).get("execution", "cpu")
        infps = self.model_settings.get(view_name, {}).get("infps", None)
        device = reg.norm_device(execution)

        frame_queue = getattr(self, f"{view_name}_frame_queue")
        output_queue = getattr(self, f"{view_name}_result_queue")
        shutdown_event = getattr(self, f"{view_name}_shutdown_event")

        try:
            kind = reg.kind_of(model)
            task = reg.get(model).get("task")
        except Exception:
            kind, task = "vision", "detection"

        if kind == "vision" and task == "detection":
            self.yolo_views.add(view_name)
            print(f"[UnifiedViewer] Starting {view_name} with {model} on {device.upper()} (detection)")
            process = Process(target=run_detection_process,
                              args=(frame_queue, output_queue, shutdown_event, device, view_name, model))
        elif kind == "vision" and task == "classification":
            self.resnet_views.add(view_name)
            print(f"[UnifiedViewer] Starting {view_name} with {model} on {device.upper()} (classification)")
            process = Process(target=run_classification_process,
                              args=(frame_queue, output_queue, shutdown_event, device, view_name, model))
        elif kind == "vlm":
            # VLM consumes video frames and streams its generated text into the view.
            self.vlm_views.add(view_name)
            print(f"[UnifiedViewer] Starting {view_name} with {model} on {device.upper()} (VLM, streaming)")
            process = Process(target=run_vlm_process,
                              args=(frame_queue, output_queue, shutdown_event, device, view_name, model,
                                    float(infps) if infps else 1.0, 32, DEMO_VLM_PROMPT))
        else:
            # LLM: no input feeder (prompt-driven); streams its generated text.
            self.llm_views.add(view_name)
            print(f"[UnifiedViewer] Starting {view_name} with {model} on {device.upper()} (LLM, streaming)")
            process = Process(target=run_llm_process,
                              args=(frame_queue, output_queue, shutdown_event, device, view_name, model,
                                    float(infps) if infps else 1.0, 64, DEMO_LLM_PROMPT))

        setattr(self, f"{view_name}_process", process)
        process.start()
    
    def initialize_threads(self):
        """Initialize and start view handler threads."""
        # Create view frame queues dictionary
        view_frame_queues = {v: getattr(self, f"{v}_frame_queue") for v in self.view_names}
        
        # Detection and VLM views are fed from the video stream, honoring each view's
        # infps. LLM views need no feeder (they generate from prompts).
        self.video_feeder = VideoFeeder(
            self.video_frame_queue,
            view_frame_queues,
            self.yolo_views | self.vlm_views,
            self.shutdown_flag,
            model_settings=self.model_settings
        )
        self.video_feeder.start_feed_thread()

        # Classification is fed from the ImageNet samples, not the video: classifying
        # frames of one street scene shows the same label over and over, which reads
        # as a frozen view. Cycling the sample images makes the top-5 chart move.
        if self.resnet_views:
            image_dir = _asset(IMAGENET_DIR)
            if not os.path.isdir(image_dir):
                print(f"[UnifiedViewer] ImageNet samples not found at {image_dir}; "
                      f"classification views will show 'No input'.")
            self.resnet_feeder = ResnetImageFeeder(
                image_dir,
                view_frame_queues,
                self.resnet_views,
                self.shutdown_flag,
                model_settings=self.model_settings
            )
            self.resnet_feeder.start_feed_thread()

        # Start view handler threads
        self.initialize_view_handlers()
    
    def _handler_class_for(self, model_name):
        """Pick the view-handler class for a model by its registry kind/task."""
        try:
            kind = reg.kind_of(model_name)
            task = reg.get(model_name).get("task")
        except Exception:
            return ResNetViewHandler
        if kind == "vision" and task == "detection":
            return YoloViewHandler
        if kind == "vision" and task == "classification":
            return ResNetViewHandler
        return LLMViewHandler  # llm / vlm -> headless stats handler

    def initialize_view_handlers(self):
        """Initialize and start view handler threads (one per view, by model kind)."""
        view_queues = {v: getattr(self, f"{v}_result_queue") for v in self.view_names}
        for view_name, result_queue in view_queues.items():
            model_name = self.model_settings.get(view_name, {}).get("model", "")
            handler_cls = self._handler_class_for(model_name)
            handler = handler_cls(
                view_name,
                self.model_settings,
                getattr(self, f"{view_name}_frame_queue"),
                result_queue,
                self.shutdown_flag,
                self.model_signals,
                self.views_without_model,
                display_slot=(getattr(self, 'display_slot_of', {}) or {}).get(view_name),
            )
            setattr(self, f"{view_name}_handler", handler)
            handler.start_display_thread()
    
    # View update methods
    def _set_view_pixmap(self, view_name, pixmap):
        """Show a frame and remember it, so a stopped view can keep its last one."""
        widget = getattr(self, view_name, None)
        if widget is None:
            return
        if getattr(self, '_frozen', False):
            return  # execution has stopped; do not overwrite the frozen frame
        widget.setPixmap(pixmap)
        widget.setScaledContents(True)
        if not hasattr(self, '_last_pixmaps'):
            self._last_pixmaps = {}
        self._last_pixmaps[view_name] = pixmap

    def update_view1_display(self, pixmap):
        self._set_view_pixmap('view1', pixmap)

    def update_view2_display(self, pixmap):
        self._set_view_pixmap('view2', pixmap)

    def update_view3_display(self, pixmap):
        self._set_view_pixmap('view3', pixmap)

    def update_view4_display(self, pixmap):
        self._set_view_pixmap('view4', pixmap)

    def freeze_views(self, label="STOPPED"):
        """Keep each view's last frame and stamp it as stopped.

        The demo is explained *after* the run stops, so the final frames and the
        numbers on them have to stay on screen. Blanking the views here would throw
        away exactly what the presenter is about to point at.
        """
        from PyQt5.QtGui import QPainter, QColor, QFont
        self._frozen = True
        # Only the tiles actually in use; hidden ones must stay hidden, not be
        # resurrected as "stopped" black squares.
        slot_to_view = {slot: view for view, slot in
                        (getattr(self, 'display_slot_of', {}) or {}).items()}
        for view_name in slot_to_view:
            widget = getattr(self, view_name, None)
            if widget is None:
                continue
            try:
                pixmap = (getattr(self, '_last_pixmaps', {}) or {}).get(view_name)
                if pixmap is None or pixmap.isNull():
                    # Never produced a frame: say so rather than go black.
                    cfg = self.model_settings.get(slot_to_view[view_name], {}) or {}
                    import demo_render as dr
                    from utils import convert_cv_to_qt
                    pixmap = convert_cv_to_qt(dr.placeholder(
                        cfg.get('model', '-'), cfg.get('execution', '-'), f"{label} (no frames)"))
                    if pixmap.isNull():
                        continue
                stamped = pixmap.copy()
                painter = QPainter(stamped)
                w, h = stamped.width(), stamped.height()
                painter.fillRect(0, h - 34, w, 34, QColor(30, 30, 30, 220))
                font = QFont()
                font.setBold(True)
                font.setPixelSize(20)
                painter.setFont(font)
                painter.setPen(QColor(255, 90, 90))
                painter.drawText(10, h - 10, label)
                painter.end()
                widget.setPixmap(stamped)
                widget.setScaledContents(True)
            except Exception as e:
                # One view failing to freeze must not stop the others freezing.
                print(f"[Freeze] {view_name}: {e}")

    def on_stop_all_clicked(self):
        """The Stop All button: same cleanup path as closing the window."""
        print("[Stop All] Requested by the user.")
        self.shutdown_views(close_after=False)

    def shutdown_views(self, close_after: bool):
        """THE cleanup path. Both Stop All and the window's X come through here.

        Order: signal workers -> join/terminate -> (each worker disposes its model in
        its own finally) -> save metrics -> stop timers -> drain queues -> freeze the
        views. Every step is individually guarded: a view that fails to clean up must
        not prevent the rest from being cleaned up, or the NPU stays held and the next
        run dies with BadAlloc.
        """
        try:
            self.stop_execution()          # idempotent; safe to call twice
        except Exception as e:
            print(f"[Shutdown] stop_execution failed (continuing): {e}")
        try:
            self.freeze_views("STOPPED")
        except Exception as e:
            print(f"[Shutdown] freeze_views failed (continuing): {e}")
        try:
            if hasattr(self, 'stop_all_button') and self.stop_all_button is not None:
                self.stop_all_button.setEnabled(False)   # re-entry guard
                self.stop_all_button.setText("Stopped")
        except Exception as e:
            print(f"[Shutdown] button update failed (continuing): {e}")
        try:
            # One last refresh so the frozen screen shows the final numbers. The cpu
            # timer is stopped by now, so these values then simply stay put.
            self.update_cpu_npu_usage()
        except Exception as e:
            print(f"[Shutdown] final metrics refresh failed (continuing): {e}")
        if close_after:
            try:
                self.close()
            except Exception as e:
                print(f"[Shutdown] window close failed: {e}")


    # Signal handling and shutdown methods
    def signal_handler(self, sig, frame):
        """Handle SIGINT signal (Ctrl+C)."""
        print("\n[SIGINT] Caught Ctrl+C, shutting down...")
        # Immediately set shutdown flags to stop video generation
        self.shutdown_flag.set()
        self.global_exit_flag = True
        
        # Set all shutdown events to stop processes
        for name in [f"{v}_shutdown_event" for v in self.view_names] + ['video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
                
        # Exit immediately without cleaning up queues
        print("[SIGINT] Forcing exit")
        os._exit(0)
    
    def closeEvent(self, event):
        """The window's X. Runs exactly the same cleanup as the Stop All button.

        Keeping these two on one path is the whole point: if X cleaned up differently
        from Stop, one of the two would inevitably leave the NPU held and the next run
        would fail with BadAlloc -- mid-demo.
        """
        event.accept()
        self.shutdown_views(close_after=False)   # already closing; do not recurse

        try:
            if hasattr(self, 'cpu_timer') and self.cpu_timer is not None:
                self.cpu_timer.deleteLater()
        except Exception as e:
            print(f"[Shutdown] timer teardown failed (continuing): {e}")

        if getattr(self, 'executor_only', False):
            print("[UnifiedViewer] Close in executor-only mode - terminating application")
            try:
                if self.info_window:
                    self.info_window.close()
            except Exception as e:
                print(f"[Shutdown] info window close failed (continuing): {e}")
            self.shutdown_all()
        else:
            print("[UnifiedViewer] Close event triggered - closing viewer window")
            try:
                self.deleteLater()
            except Exception as e:
                print(f"[Shutdown] deleteLater failed: {e}")

    def shutdown_all(self):
        """Stop everything, then end the process -- cleanup first, never skipped."""
        try:
            self.shutdown_views(close_after=False)
        except Exception as e:
            print(f"[Shutdown] cleanup failed (exiting anyway): {e}")

        from PyQt5.QtWidgets import QApplication
        import threading
        print("[Shutdown] Cleanup complete; quitting the event loop.", flush=True)

        # Backstop FIRST, and on a plain threading.Timer -- not a QTimer. quit() does
        # not reliably make exec_() return here (the Mobilint runtime and torch leave
        # threads behind), and once the loop has stopped dispatching, a QTimer can
        # never fire: the backstop that was supposed to save us would be the one thing
        # guaranteed not to run, and the process would hang forever holding nothing but
        # a stale window. A threading.Timer is independent of Qt entirely.
        #
        # This forces the exit only AFTER cleanup above has finished, so nothing is
        # skipped: models are disposed, results are written, workers are reaped.
        def _force():
            print("[Shutdown] Event loop did not exit in time; forcing exit.", flush=True)
            os._exit(0)
        try:
            t = threading.Timer(3.0, _force)
            t.daemon = True
            t.start()
        except Exception as e:
            print(f"[Shutdown] Could not arm the exit backstop ({e}); exiting now.")
            os._exit(0)

        try:
            QApplication.instance().quit()
        except Exception as e:
            print(f"[Shutdown] quit() failed: {e}")
    
    # Monitoring and statistics methods
    def start_execution(self, duration):
        """
        Start execution with a specified duration.
        
        Args:
            duration (int): Duration in seconds for the execution to run.
        """
        print(f"Starting execution with duration: {duration} seconds")
        # We treat the first 5 seconds as warmup; only measure after that.
        try:
            self.window_duration_sec = max(0.0, float(duration) - 5.0)
        except Exception:
            self.window_duration_sec = None
        
        # Reset idempotent stop flag for new run
        self._already_stopped = False

        # Create a run_id and export to environment so child processes can log it
        try:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.environ["RUN_ID"] = self.run_id
        except Exception:
            self.run_id = ""
        
        # Initialize and start processes if they're not already running
        if not hasattr(self, 'video_reader_proc') or not self.video_reader_proc.is_alive():
            self.initialize_processes()
            self.initialize_threads()
        
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
        self.stop_execution()
        
    def _begin_measurement_window(self):
        """Reset all per-view and feeder counters after warmup to start measurement."""
        try:
            for name in [f"{v}_handler" for v in self.view_names]:
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
        
    def stop_execution(self):
        """Stop model execution without closing the application."""
        # If already stopped once for this run, skip duplicate work and saving
        if getattr(self, '_already_stopped', False):
            print("[Stop Execution] Already stopped for this run. Skipping duplicate save.")
            return
        # Mark as stopped to ensure idempotency
        self._already_stopped = True

        # Immediately signal all local threads/feeders to stop enqueuing
        try:
            if hasattr(self, 'shutdown_flag') and self.shutdown_flag:
                self.shutdown_flag.set()
            self.global_exit_flag = True
        except Exception:
            pass
        
        # Set shutdown events to signal processes to stop
        for name in [f"{v}_shutdown_event" for v in self.view_names] + ['video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
                
        # Gracefully stop all processes first, then force terminate if needed
        process_names = [f"{v}_process" for v in self.view_names] + ['video_reader_proc']
        processes = [getattr(self, name, None) for name in process_names if hasattr(self, name) and getattr(self, name, None)]
        
        # Give processes time to exit their loops and run cleanup (e.g., NPU driver close in finally)
        for p in processes:
            if p and p.is_alive():
                try:
                    p.join(timeout=3.0)
                except Exception as e:
                    print(f"[Stop Execution] Process join error: {e}")
        
        # Force terminate any stubborn processes that didn't exit
        for p in processes:
            if p and p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=0.5)
                except Exception as e:
                    print(f"[Stop Execution] Process termination error: {e}")
                    
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

        # Stop periodic timers to avoid lingering updates on Ubuntu
        try:
            if hasattr(self, 'cpu_timer') and self.cpu_timer is not None:
                self.cpu_timer.stop()
        except Exception:
            pass
        
        # After all processes are stopped, clear and close all queues to avoid BrokenPipe in next runs
        try:
            self._cleanup_queues()
        except Exception as e:
            print(f"[Stop Execution] Queue cleanup warning: {e}")
            
        print("[Stop Execution] Model execution stopped, window remains open with last results")
        
    def _cleanup_queues(self):
        """Drain, close, and nullify all multiprocessing queues safely between runs.

        Two things here are load-bearing, both learned the hard way:

        `cancel_join_thread()` must come FIRST. Each queue has a background feeder
        thread that can be blocked writing into a pipe whose reader (the worker
        process) has just been terminated. Closing before cancelling makes teardown
        wait on a thread that will never make progress.

        The drain must be BOUNDED. It used to be `while True: q.get_nowait()`, which
        hung the whole shutdown on the X path: a queue still being topped up by a
        blocked feeder never reports empty, so the loop never ends and the window
        never closes -- leaving the NPU held and the next run dead on arrival.
        """
        import time as _t

        DRAIN_LIMIT = 10000       # items
        DRAIN_SECONDS = 1.0       # per queue

        def _drain(q):
            if not q:
                return
            deadline = _t.time() + DRAIN_SECONDS
            for _ in range(DRAIN_LIMIT):
                if _t.time() > deadline:
                    print("[Cleanup] Drain deadline hit; abandoning the rest of the queue.")
                    return
                try:
                    q.get_nowait()
                except Exception:
                    return            # Empty, closed, or broken: nothing more to take

        def _cancel(q):
            if not q:
                return
            try:
                q.cancel_join_thread()   # before anything else: never wait on a stuck feeder
            except Exception as e:
                print(f"[Cleanup] cancel_join_thread failed (continuing): {e}")

        def _close(q):
            if not q:
                return
            try:
                q.close()
            except Exception as e:
                print(f"[Cleanup] queue close failed (continuing): {e}")
        # Give local feeders a moment to observe shutdown_flag
        try:
            _t.sleep(0.05)
        except Exception:
            pass
        # List of queue attribute names to cleanup
        q_names = ['video_frame_queue']
        for v in getattr(self, 'view_names', []):
            q_names += [f"{v}_frame_queue", f"{v}_result_queue"]
        for name in q_names:
            q = getattr(self, name, None)
            try:
                _cancel(q)
                _drain(q)
                _close(q)
            finally:
                try:
                    setattr(self, name, None)
                except Exception:
                    pass
        
    def update_cpu_npu_usage(self):
        """Update CPU and NPU usage information."""
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]
        
        # Per-view stats for every scheduled view (any N, not just the 4 display slots).
        views = list(self.view_names)
        stats = {}
        for v in views:
            cfg = self.model_settings.get(v, {}) or {}
            h = getattr(self, f"{v}_handler", None)
            stats[v] = {
                "fps": getattr(h, 'avg_fps', 0.0),
                "infer_ms": getattr(h, 'avg_infer_time', 0.0),
                "count": getattr(h, 'infer_count', 0),
                "model": cfg.get("model", ""),
                "mode": str(cfg.get("execution", "cpu")).upper(),
                "wait_ms": getattr(h, 'avg_wait_ms', 0.0),
            }

        # Calculate total / average FPS (total throughput)
        total_fps = sum(st["fps"] for st in stats.values())
        total_avg_fps = total_fps / len(views) if views else 0.0

        # Create performance text and append reallocation trigger status based on recent results
        # NOTE: Reallocation trigger condition checks are disabled per requirement.
        # The UI will display a fixed message to indicate triggers are disabled.
        name1 = "drop-rate surge"
        name2 = "Queueing-delay pressure"
        line1 = f"Reallocation trigger [{name1}]: Disabled"
        line2 = f"[{name2}]: Disabled"
        self.info_window.update_trigger_below_metrics(line1 + "\n" + line2)
        trigger_text = ""

        lines = [
            f"<b>Total Throughput: {total_fps:.1f} FPS</b><br>",
            f"<b>Total Average Throughput: {total_avg_fps:.1f} FPS</b><br><br>",
        ]
        for idx, v in enumerate(views):
            st = stats[v]
            color = VIEW_COLORS[idx % len(VIEW_COLORS)]
            suffix = "" if v in DISPLAY_SLOTS else " [headless]"
            lines.append(
                f"<b><span style='color: {color};'>{v.capitalize()} ({st['model']} {st['mode']}){suffix}</span></b> "
                f"Avg FPS: <span style='color: {color};'>{st['fps']:.1f}</span> "
                f"(<span style='color: {color};'>{st['infer_ms']:.1f} ms</span>)<br>"
            )
        performance_text = "".join(lines)

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
            scheduled_views = views
            drop_map = getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}) or {}
            # per_view_stats fields aligned with _get_device_metrics_default expectations
            per_view_stats = {
                v: (stats[v]["fps"], stats[v]["infer_ms"], stats[v]["count"],
                    stats[v]["model"], stats[v]["mode"], stats[v]["wait_ms"],
                    int(drop_map.get(v, 0) or 0))
                for v in scheduled_views
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
            # Deadline miss rate: (late completions + queue-full drops) / offered
            feeder = getattr(self, 'video_feeder', None)
            fdrop = getattr(feeder, 'drop_counts', {}) or {}
            foffered = getattr(feeder, 'offered_counts', {}) or {}
            miss = offered = 0
            for v in scheduled_views:
                h = getattr(self, f"{v}_handler", None)
                late = int(getattr(h, 'late_count', 0) or 0)
                dropped = int(fdrop.get(v, 0) or 0)
                off = int(foffered.get(v, 0) or 0) or (int(getattr(h, 'infer_count', 0) or 0) + dropped)
                miss += late + dropped
                offered += off
            miss_rate = (miss / offered) if offered > 0 else 0.0
            score = total_fps - 100.0 * miss_rate
            # Show each metric on its own line
            metrics_line = (
                f"Total: {total_fps:.2f} FPS\n"
                f"q95: {max_q_wait:.1f} ms\n"
                f"Deadline miss: {miss_rate*100:.1f}%\n"
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
            # Per-view stats, gathered for every scheduled view (any N).
            scheduled_views = list(self.view_names)
            drop_map = getattr(getattr(self, 'video_feeder', None), 'drop_counts', {}) or {}
            per_view_stats = {}
            for v in scheduled_views:
                cfg = self.model_settings.get(v, {}) or {}
                h = getattr(self, f"{v}_handler", None)
                per_view_stats[v] = (
                    getattr(h, 'avg_fps', 0.0),
                    getattr(h, 'avg_infer_time', 0.0),
                    getattr(h, 'infer_count', 0),
                    cfg.get("model", ""),
                    str(cfg.get("execution", "cpu")).upper(),
                    getattr(h, 'avg_wait_ms', 0.0),
                    int(drop_map.get(v, 0) or 0),
                )

            # Calculate total throughput for scheduled views
            total_fps = sum(per_view_stats[v][0] for v in scheduled_views)
            scheduled_count = len(scheduled_views)
            total_avg_fps = total_fps / scheduled_count if scheduled_count > 0 else 0.0

            # Per-view token throughput (non-zero only for LLM/VLM views)
            tokens_by_view = {
                v: float(getattr(getattr(self, f"{v}_handler", None), "avg_tokens_per_s", 0.0) or 0.0)
                for v in self.view_names
            }
            total_tokens_per_s = sum(tokens_by_view[v] for v in scheduled_views)

            # Deadline-miss accounting per view (demand-based):
            #   offered  = requests demanded over the window = infps * window_sec
            #   on_time  = completions whose end-to-end latency <= deadline
            #   miss     = offered - on_time  (dropped, late, and undelivered all count)
            wsec = float(self.window_duration_sec) if getattr(self, 'window_duration_sec', None) else 1.0
            miss_by_view = {}
            for v in self.view_names:
                handler = getattr(self, f"{v}_handler", None)
                completed = int(getattr(handler, "infer_count", 0) or 0)
                late = int(getattr(handler, "late_count", 0) or 0)
                on_time = max(0, completed - late)
                try:
                    infps = float((self.model_settings.get(v, {}) or {}).get("infps", 0.0) or 0.0)
                except Exception:
                    infps = 0.0
                offered = int(round(infps * wsec)) if infps > 0 else completed
                offered = max(offered, completed)  # never fewer than what completed
                misses = max(0, offered - on_time)
                rate = (misses / offered) if offered > 0 else 0.0
                miss_by_view[v] = {"miss": misses, "offered": offered, "rate": rate,
                                   "late": late, "on_time": on_time}
            # Window-level miss rate = mean of per-application miss rates (each
            # application instance weighted equally, so it is not dominated by the
            # highest-rate model when request rates are heterogeneous).
            per_view_rates = [miss_by_view[v]["rate"] for v in scheduled_views]
            total_miss_rate = min(1.0, (sum(per_view_rates) / len(per_view_rates)) if per_view_rates else 0.0)

            # Prepare throughput data including all scheduled views (even if 0 inferences)
            throughput_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "window_sec": float(self.window_duration_sec) if self.window_duration_sec is not None else None,
                "combination": self.current_combination,
                "models": {},
                "total": {
                    "total_throughput_fps": round(total_fps, 2),
                    "avg_throughput_fps": round(total_avg_fps, 2),
                    "total_tokens_per_s": round(total_tokens_per_s, 2),
                    "deadline_miss_rate": round(total_miss_rate, 4)
                }
            }

            # Add all scheduled views to the models dictionary (include zeros if no inferences)
            devices_used = set()
            for v in scheduled_views:
                avg_fps, avg_time, infer_cnt, model_name, exec_mode, avg_wait_ms, dropped = per_view_stats[v]
                mv = miss_by_view[v]
                throughput_data["models"][v] = {
                    "model": model_name,
                    "execution": exec_mode,
                    "throughput_fps": round(avg_fps, 2),
                    "tokens_per_s": round(tokens_by_view.get(v, 0.0), 2),
                    "avg_inference_time_ms": round(avg_time, 2),
                    "inference_count": int(infer_cnt),
                    "avg_wait_to_preprocess_ms": round(avg_wait_ms or 0.0, 2),
                    "deadline_miss_count": int(mv["miss"]),
                    "frames_total": int(mv["offered"]),
                    "on_time_count": int(mv["on_time"]),
                    "deadline_miss_rate": round(mv["rate"], 4)
                }
                devices_used.add(exec_mode)

            # Compute per-device queue metrics with fallback when timing logs are unavailable
            try:
                device_metrics = self._get_device_metrics_default(devices_used, per_view_stats, scheduled_views)
                throughput_data["devices"] = device_metrics
            except Exception as e:
                print(f"[Save Throughput] Warning: failed to compute device metrics: {e}")
            
            # Determine if the current combination is the first schedule in the YAML
            is_first_schedule = False
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
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
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