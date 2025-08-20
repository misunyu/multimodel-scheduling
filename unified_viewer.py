"""
Main UnifiedViewer class for the multimodel scheduling application.
"""
import os
import json
import signal
import yaml
from datetime import datetime
from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic
from multiprocessing import Process, Queue, Event

# Import local modules
from utils import get_cpu_metrics
from view_handlers import ModelSignals, YoloViewHandler, ResNetViewHandler, VideoFeeder, ResnetImageFeeder
from model_processors import (
    video_reader_process,
    run_yolo_cpu_process,
    run_yolo_npu_process,
    run_resnet_cpu_process,
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
        self.model_performance_label.setStyleSheet(border_style)
        self.cpu_info_label.setStyleSheet(border_style)
        self.npu_info_label.setStyleSheet(border_style)
        
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
        
        # Default execution duration is 60 seconds
        self.duration_edit.setText("60")
        
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
    
    def get_execution_duration(self):
        """Get the execution duration from the input field."""
        try:
            return int(self.duration_edit.text())
        except ValueError:
            return 60  # Default value if input is invalid
    
    def update_model_performance(self, text):
        """Update the model performance label."""
        self.model_performance_label.setText(text)
    
    def update_cpu_info(self, text):
        """Update the CPU info label."""
        self.cpu_info_label.setText(text)
    
    def update_npu_info(self, text):
        """Update the NPU info label."""
        self.npu_info_label.setText(text)
        
    def update_schedule_name(self, text):
        """Update the schedule name label."""
        self.schedule_name_label.setText(text)
        
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
        uic.loadUi("schedule_executor_display.ui", self)
        
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
        self.info_window.show()
        
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
                                "execution": model_config.get("execution", "cpu")
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
                        "model": "yolov3_small" if view in ["view1", "view3"] else "resnet50_small",
                        "execution": "cpu"
                    }
                    print(f"[UnifiedViewer] No model specified for {view} in {self.schedule_file}")
                    
            print(f"[UnifiedViewer] Loaded model settings from {self.schedule_file} for {self.current_combination}")
        except Exception as e:
            print(f"[UnifiedViewer ERROR] Failed to load {self.schedule_file}: {e}")
            # Set default settings if file loading fails
            self.model_settings = {
                "view1": {"model": "yolov3_small", "execution": "cpu"},
                "view2": {"model": "resnet50_small", "execution": "cpu"},
                "view3": {"model": "yolov3_small", "execution": "cpu"},
                "view4": {"model": "resnet50_small", "execution": "cpu"}
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
    
    def initialize_state_variables(self):
        """Initialize state variables."""
        # Global flag for signaling threads to exit
        self.global_exit_flag = False
        
        # Initialize common state variables
        self.shutdown_flag = Event()
        self.prev_cpu_stats = get_cpu_metrics(interval=0)
        
        # Ensure stop_execution is idempotent: save throughput only once per schedule
        self._already_stopped = False

        # Initialize queues and events
        self.video_frame_queue = Queue(maxsize=10)
        self.video_shutdown_event = Event()
        
        # View1 queues and events
        self.view1_frame_queue = Queue(maxsize=10)
        self.view1_output_queue = Queue(maxsize=5)
        self.view1_shutdown_event = Event()
        
        # View2 queues and events
        self.view2_frame_queue = Queue(maxsize=10)
        self.view2_output_queue = Queue(maxsize=5)
        self.view2_shutdown_event = Event()
        
        # View3 queues and events
        self.view3_frame_queue = Queue(maxsize=10)
        self.view3_result_queue = Queue(maxsize=5)
        self.view3_shutdown_event = Event()
        
        # View4 queues and events
        self.view4_frame_queue = Queue(maxsize=10)
        self.view4_result_queue = Queue(maxsize=5)
        self.view4_shutdown_event = Event()
        
        # Initialize a dictionary to track which views are running YOLO models (need video frames)
        self.yolo_views = set()
        # Track ResNet views that need image feeder at 10 Hz
        self.resnet_views = set()
    
    def initialize_processes(self):
        """Initialize and start model processes."""
        # Start video reader process
        self.video_reader_proc = Process(
            target=video_reader_process,
            args=("stockholm_1280x720.mp4", self.video_frame_queue, self.video_shutdown_event),
            daemon=True
        )
        self.video_reader_proc.start()
        
        # Start view processes
        self.start_view_process("view1")
        self.start_view_process("view2")
        self.start_view_process("view3")
        self.start_view_process("view4")
    
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
        
        frame_queue = getattr(self, f"{view_name}_frame_queue")
        output_queue = getattr(self, f"{view_name}_output_queue") if view_name in ["view1", "view2"] else getattr(self, f"{view_name}_result_queue")
        shutdown_event = getattr(self, f"{view_name}_shutdown_event")
        
        if model.startswith("yolov3"):
            # YOLO model
            self.yolo_views.add(view_name)
            if execution == "npu0" or execution == "npu1":
                npu_id = 0 if execution == "npu0" else 1
                print(f"[UnifiedViewer] Starting {view_name} with {model} NPU{npu_id}")
                process = Process(
                    target=run_yolo_npu_process,
                    args=(frame_queue, output_queue, shutdown_event, npu_id, view_name),
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU")
                process = Process(
                    target=run_yolo_cpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
                )
        else:
            # ResNet model
            self.resnet_views.add(view_name)
            if execution == "npu0" or execution == "npu1":
                npu_id = 0 if execution == "npu0" else 1
                print(f"[UnifiedViewer] Starting {view_name} with {model} NPU{npu_id}")
                process = Process(
                    target=run_resnet_npu_process,
                    args=(frame_queue, output_queue, shutdown_event, npu_id, view_name),
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU")
                process = Process(
                    target=run_resnet_cpu_process,
                    args=(frame_queue, output_queue, shutdown_event, view_name),
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
        
        # Start video feeder thread (for YOLO models)
        self.video_feeder = VideoFeeder(
            self.video_frame_queue,
            view_frame_queues,
            self.yolo_views,
            self.shutdown_flag
        )
        self.video_feeder.start_feed_thread()

        # Start ResNet image feeder at 10 Hz for ResNet views
        self.resnet_feeder = ResnetImageFeeder(
            image_dir="./imagenet-sample-images",
            view_frame_queues=view_frame_queues,
            resnet_views=self.resnet_views,
            shutdown_flag=self.shutdown_flag,
            interval_sec=0.5
        )
        self.resnet_feeder.start_feed_thread()
        
        # Start view handler threads
        self.initialize_view_handlers()
    
    def initialize_view_handlers(self):
        """Initialize and start view handler threads."""
        # View1 handler
        view1_model = self.model_settings.get("view1", {}).get("model", "")
        if view1_model.startswith("yolov3"):
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
        if view2_model.startswith("yolov3"):
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
        if view3_model.startswith("yolov3"):
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
        if view4_model.startswith("yolov3"):
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
        self.view1.setPixmap(pixmap)
        self.view1.setScaledContents(True)
    
    def update_view2_display(self, pixmap):
        """Update view2 display."""
        self.view2.setPixmap(pixmap)
        self.view2.setScaledContents(True)
    
    def update_view3_display(self, pixmap):
        """Update view3 display."""
        self.view3.setPixmap(pixmap)
        self.view3.setScaledContents(True)
    
    def update_view4_display(self, pixmap):
        """Update view4 display."""
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
        print("[SIGINT] Forcing exit")
        os._exit(0)
    
    def closeEvent(self, event):
        """Handle window close event."""
        print("[UnifiedViewer] Close event triggered - hiding window only")
        event.accept()
        # Immediately set shutdown flags to stop video generation
        self.shutdown_flag.set()
        self.global_exit_flag = True
        
        # Don't close the info window as it's now the main window
        # Instead, just hide this window
        self.hide()
        
        # Set all shutdown events to stop processes
        for name in ['view1_shutdown_event', 'view2_shutdown_event',
                     'view3_shutdown_event', 'view4_shutdown_event',
                     'video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
                
        # Call the stop_execution method to clean up resources but don't exit
        self.stop_execution()
        print("[UnifiedViewer] Window hidden, info_window remains open")
    
    def shutdown_all(self):
        """Clean up resources and shut down the application."""
        # First stop all model execution
        self.stop_execution()
            
        print("[Shutdown] Forcing exit")
        os._exit(0)
    
    # Monitoring and statistics methods
    def start_execution(self, duration):
        """
        Start execution with a specified duration.
        
        Args:
            duration (int): Duration in seconds for the execution to run.
        """
        print(f"Starting execution with duration: {duration} seconds")
        
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
            
        # Schedule stopping execution after the specified duration
        QTimer.singleShot(duration * 1000, self.timed_shutdown)
    
    def timed_shutdown(self):
        """Stop execution after the scheduled duration without closing the application."""
        print("Execution duration completed, stopping model execution...")
        self.stop_execution()
        
    def stop_execution(self):
        """Stop model execution without closing the application."""
        # If already stopped once for this run, skip duplicate work and saving
        if getattr(self, '_already_stopped', False):
            print("[Stop Execution] Already stopped for this run. Skipping duplicate save.")
            return
        # Mark as stopped to ensure idempotency
        self._already_stopped = True
        
        # Set shutdown events to signal processes to stop
        for name in ['view1_shutdown_event', 'view2_shutdown_event',
                     'view3_shutdown_event', 'view4_shutdown_event',
                     'video_shutdown_event']:
            event = getattr(self, name, None)
            if event:
                event.set()
                
        # Gracefully stop all processes first, then force terminate if needed
        process_names = ['view1_process', 'view2_process', 'view3_process', 'view4_process', 'video_reader_proc']
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
            
        print("[Stop Execution] Model execution stopped, window remains open with last results")
        
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
        
        # Calculate total average FPS (total throughput)
        total_fps = (view1_avg_fps + view2_avg_fps + view3_avg_fps + view4_avg_fps)
        total_avg_fps = total_fps / 4 if total_fps > 0 else 0.0
        
        # Get model and execution mode for each view
        view1_model = self.model_settings.get("view1", {}).get("model", "yolov3_small")
        view1_mode = self.model_settings.get("view1", {}).get("execution", "cpu").upper()
        
        view2_model = self.model_settings.get("view2", {}).get("model", "resnet50_small")
        view2_mode = self.model_settings.get("view2", {}).get("execution", "cpu").upper()
        
        view3_model = self.model_settings.get("view3", {}).get("model", "yolov3_small")
        view3_mode = self.model_settings.get("view3", {}).get("execution", "cpu").upper()
        
        view4_model = self.model_settings.get("view4", {}).get("model", "resnet50_small")
        view4_mode = self.model_settings.get("view4", {}).get("execution", "cpu").upper()
        
        # Create performance text
        performance_text = (
            f"<b>Total Throughput: {total_fps:.1f} FPS</b><br>"
            f"<b>Total Average Throughput: {total_avg_fps:.1f} FPS</b><br><br>"
            f"<b>View1 ({view1_model} {view1_mode})</b> Avg FPS: {view1_avg_fps:.1f} "
            f"(<span style='color: gray;'>{view1_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: purple;'>View2 ({view2_model} {view2_mode})</span></b> Avg FPS: "
            f"<span style='color: purple;'>{view2_avg_fps:.1f}</span> "
            f"(<span style='color: purple;'>{view2_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: green;'>View3 ({view3_model} {view3_mode})</span></b> Avg FPS: "
            f"<span style='color: green;'>{view3_avg_fps:.1f}</span> "
            f"(<span style='color: green;'>{view3_avg_infer_time:.1f} ms</span>)<br>"
            f"<b><span style='color: blue;'>View4 ({view4_model} {view4_mode})</span></b> Avg FPS: "
            f"<span style='color: blue;'>{view4_avg_fps:.1f}</span> "
            f"(<span style='color: blue;'>{view4_avg_infer_time:.1f} ms</span>)"
        )
        
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
        self.info_window.update_npu_info(npu_info_text)
        
        # Update previous stats for next calculation
        self.prev_cpu_stats = current
    
    def save_throughput_data(self):
        """Save the current throughput of each model and the total throughput to result_throughput.json."""
        try:
            # Get model and execution mode for each view
            view1_model = self.model_settings.get("view1", {}).get("model", "yolov3_small")
            view1_mode = self.model_settings.get("view1", {}).get("execution", "cpu").upper()
            
            view2_model = self.model_settings.get("view2", {}).get("model", "resnet50_small")
            view2_mode = self.model_settings.get("view2", {}).get("execution", "cpu").upper()
            
            view3_model = self.model_settings.get("view3", {}).get("model", "yolov3_small")
            view3_mode = self.model_settings.get("view3", {}).get("execution", "cpu").upper()
            
            view4_model = self.model_settings.get("view4", {}).get("model", "resnet50_small")
            view4_mode = self.model_settings.get("view4", {}).get("execution", "cpu").upper()
            
            # Check if view handlers exist
            if not all(hasattr(self, f'view{i}_handler') for i in range(1, 5)):
                print("[Save Throughput] No view handlers initialized, skipping throughput data save")
                return
                
            # Get performance statistics from view handlers
            view1_avg_fps = self.view1_handler.avg_fps
            view1_avg_infer_time = self.view1_handler.avg_infer_time
            view1_infer_count = self.view1_handler.infer_count
            
            view2_avg_fps = self.view2_handler.avg_fps
            view2_avg_infer_time = self.view2_handler.avg_infer_time
            view2_infer_count = self.view2_handler.infer_count
            
            view3_avg_fps = self.view3_handler.avg_fps
            view3_avg_infer_time = self.view3_handler.avg_infer_time
            view3_infer_count = self.view3_handler.infer_count
            
            view4_avg_fps = self.view4_handler.avg_fps
            view4_avg_infer_time = self.view4_handler.avg_infer_time
            view4_infer_count = self.view4_handler.infer_count
            
            # Determine which views are actually scheduled in this combination
            scheduled_views = [v for v in ["view1", "view2", "view3", "view4"] if v not in self.views_without_model]

            # Map helpers for per-view stats, including avg_wait_ms (if available) and dropped frames
            view1_wait = getattr(self.view1_handler, 'avg_wait_ms', 0.0)
            view2_wait = getattr(self.view2_handler, 'avg_wait_ms', 0.0)
            view3_wait = getattr(self.view3_handler, 'avg_wait_ms', 0.0)
            view4_wait = getattr(self.view4_handler, 'avg_wait_ms', 0.0)

            # Drop counts from feeder (0 if not present)
            drop_counts = getattr(self, 'video_feeder', None)
            drop_map = getattr(drop_counts, 'drop_counts', {}) if drop_counts else {}

            per_view_stats = {
                "view1": (view1_avg_fps, view1_avg_infer_time, view1_infer_count, view1_model, view1_mode, view1_wait, int(drop_map.get("view1", 0))),
                "view2": (view2_avg_fps, view2_avg_infer_time, view2_infer_count, view2_model, view2_mode, view2_wait, int(drop_map.get("view2", 0))),
                "view3": (view3_avg_fps, view3_avg_infer_time, view3_infer_count, view3_model, view3_mode, view3_wait, int(drop_map.get("view3", 0))),
                "view4": (view4_avg_fps, view4_avg_infer_time, view4_infer_count, view4_model, view4_mode, view4_wait, int(drop_map.get("view4", 0))),
            }

            # Calculate total throughput for scheduled views
            total_fps = sum(per_view_stats[v][0] for v in scheduled_views)
            scheduled_count = len(scheduled_views)
            total_avg_fps = total_fps / scheduled_count if scheduled_count > 0 else 0.0

            # Prepare throughput data including all scheduled views (even if 0 inferences)
            throughput_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "combination": self.current_combination,
                "models": {},
                "total": {
                    "total_throughput_fps": round(total_fps, 2),
                    "avg_throughput_fps": round(total_avg_fps, 2)
                }
            }
            
            # Add all scheduled views to the models dictionary (include zeros if no inferences)
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
                    if os.path.exists("result_throughput.json"):
                        with open("result_throughput.json", "r", encoding="utf-8") as rf:
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
                print("[Save Throughput] First schedule detected. Clearing previous contents of result_throughput.json.")

            results.append(throughput_data)

            with open("result_throughput.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
            print("[Shutdown] Throughput data appended to result_throughput.json")
        except Exception as e:
            print(f"[Shutdown ERROR] Failed to save throughput data: {e}")

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