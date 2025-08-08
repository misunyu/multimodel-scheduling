"""
Main UnifiedViewer class for the multimodel scheduling application.
"""
import os
import json
import signal
import yaml
from datetime import datetime
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import uic
from multiprocessing import Process, Queue, Event

# Import local modules
from utils import get_cpu_metrics
from view_handlers import ModelSignals, YoloViewHandler, ResNetViewHandler, VideoFeeder
from model_processors import (
    video_reader_process,
    run_yolo_cpu_process,
    run_yolo_npu_process,
    run_resnet_cpu_process,
    run_resnet_npu_process
)

class UnifiedViewer(QMainWindow):
    """Main viewer class for the multimodel scheduling application."""
    
    def __init__(self, schedule_file='model_schedules.yaml'):
        """Initialize the UnifiedViewer.
        
        Args:
            schedule_file (str): Path to the model scheduling information file.
        """
        super().__init__()
        uic.loadUi("schedule_executor_display.ui", self)
        
        # Set up signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Store the schedule file path
        self.schedule_file = schedule_file

        # Initialize model settings and views
        self.initialize_model_settings()
        self.initialize_ui_components()
        self.initialize_state_variables()
        self.initialize_processes()
        self.initialize_threads()
        
        # CPU/NPU monitoring
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)
    
    def initialize_model_settings(self):
        """Initialize model settings from YAML configuration."""
        self.model_settings = {}
        self.views_without_model = set()  # Track views without specified models
        self.current_combination = "combination_1"  # Default combination
        
        try:
            # Load configuration from the specified schedule file
            with open(self.schedule_file, "r") as f:
                config = yaml.safe_load(f)
                
            # Create a mapping from views to model configurations based on the display field
            view_to_model_map = {}
            # Use combination1 configuration
            if self.current_combination in config:
                for model_config_name, model_config in config[self.current_combination].items():
                    if "display" in model_config:
                        view_name = model_config["display"]
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
                    
            print(f"[UnifiedViewer] Loaded model settings from {self.schedule_file}")
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
        self.model_performance_label = self.findChild(QLabel, "model_performance_label")
        self.cpu_info_label = self.findChild(QLabel, "cpu_info_label")
        self.npu_info_label = self.findChild(QLabel, "npu_info_label")

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
                    args=(frame_queue, output_queue, shutdown_event, npu_id),
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU")
                process = Process(
                    target=run_yolo_cpu_process,
                    args=(frame_queue, output_queue, shutdown_event),
                )
        else:
            # ResNet model
            if execution == "npu0" or execution == "npu1":
                npu_id = 0 if execution == "npu0" else 1
                print(f"[UnifiedViewer] Starting {view_name} with {model} NPU{npu_id}")
                process = Process(
                    target=run_resnet_npu_process,
                    args=("./imagenet-sample-images", output_queue, shutdown_event, npu_id),
                )
            else:
                print(f"[UnifiedViewer] Starting {view_name} with {model} CPU")
                process = Process(
                    target=run_resnet_cpu_process,
                    args=("./imagenet-sample-images", output_queue, shutdown_event),
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
        
        # Start video feeder thread
        self.video_feeder = VideoFeeder(
            self.video_frame_queue,
            view_frame_queues,
            self.yolo_views,
            self.shutdown_flag
        )
        self.video_feeder.start_feed_thread()
        
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
        event.accept()
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
                
        # Call the shutdown method to clean up resources
        self.shutdown_all()
    
    def shutdown_all(self):
        """Clean up resources and shut down."""
        # Save throughput data before shutting down
        try:
            self.save_throughput_data()
        except Exception as e:
            print(f"[Shutdown] Error saving throughput data: {e}")
            
        # Terminate all processes
        process_names = ['view1_process', 'view2_process', 'view3_process', 'view4_process', 'video_reader_proc']
        processes = [getattr(self, name, None) for name in process_names if hasattr(self, name) and getattr(self, name, None)]
        
        for p in processes:
            if p and p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=0.5)
                except Exception as e:
                    print(f"[Shutdown] Process termination error: {e}")

        print("[Shutdown] Forcing exit")
        os._exit(0)
    
    # Monitoring and statistics methods
    def update_cpu_npu_usage(self):
        """Update CPU and NPU usage information."""
        current = get_cpu_metrics(interval=0)
        prev = self.prev_cpu_stats
        delta_ctx = current["Context_Switches"] - prev["Context_Switches"]
        delta_int = current["Interrupts"] - prev["Interrupts"]
        load1, load5, load15 = current["Load_Average"]
        
        # Get performance statistics from view handlers
        view1_avg_fps = self.view1_handler.avg_fps
        view1_avg_infer_time = self.view1_handler.avg_infer_time
        
        view2_avg_fps = self.view2_handler.avg_fps
        view2_avg_infer_time = self.view2_handler.avg_infer_time
        
        view3_avg_fps = self.view3_handler.avg_fps
        view3_avg_infer_time = self.view3_handler.avg_infer_time
        
        view4_avg_fps = self.view4_handler.avg_fps
        view4_avg_infer_time = self.view4_handler.avg_infer_time
        
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
        
        self.model_performance_label.setText(
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

        self.cpu_info_label.setText(
            f"<b><span style='color: blue;'>CPU</span></b><br>"
            f"Usage: {current['CPU_Usage_percent']:.1f} %<br>"
            f"LoadAvg: {load1:.2f} / {load5:.2f} / {load15:.2f}<br>"
            f"CtxSwitches/sec: {delta_ctx} | Int/sec: {delta_int}"
        )
        
        self.npu_info_label.setText(
            f"<b><span style='color: green;'>NPU</span></b><br>"
            f"Usage: 42.0 %<br>"
            f"LoadAvg: 0.12 / 0.10 / 0.08<br>"
            f"CtxSwitches/sec: 12 | Int/sec: 3"
        )
        
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
            
            # Identify views with allocated models (inference_count > 0)
            active_views = []
            active_fps_values = []
            
            if view1_infer_count > 0:
                active_views.append("view1")
                active_fps_values.append(view1_avg_fps)
            
            if view2_infer_count > 0:
                active_views.append("view2")
                active_fps_values.append(view2_avg_fps)
            
            if view3_infer_count > 0:
                active_views.append("view3")
                active_fps_values.append(view3_avg_fps)
            
            if view4_infer_count > 0:
                active_views.append("view4")
                active_fps_values.append(view4_avg_fps)
            
            # Calculate total throughput only for active views
            total_fps = sum(active_fps_values)
            active_view_count = len(active_views)
            total_avg_fps = total_fps / active_view_count if active_view_count > 0 else 0.0
            
            # Prepare throughput data with only active views
            throughput_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "combination": self.current_combination,
                "models": {},
                "total": {
                    "total_throughput_fps": round(total_fps, 2),
                    "avg_throughput_fps": round(total_avg_fps, 2)
                }
            }
            
            # Add only active views to the models dictionary
            if "view1" in active_views:
                throughput_data["models"]["view1"] = {
                    "model": view1_model,
                    "execution": view1_mode,
                    "throughput_fps": round(view1_avg_fps, 2),
                    "avg_inference_time_ms": round(view1_avg_infer_time, 2),
                    "inference_count": view1_infer_count
                }
            
            if "view2" in active_views:
                throughput_data["models"]["view2"] = {
                    "model": view2_model,
                    "execution": view2_mode,
                    "throughput_fps": round(view2_avg_fps, 2),
                    "avg_inference_time_ms": round(view2_avg_infer_time, 2),
                    "inference_count": view2_infer_count
                }
            
            if "view3" in active_views:
                throughput_data["models"]["view3"] = {
                    "model": view3_model,
                    "execution": view3_mode,
                    "throughput_fps": round(view3_avg_fps, 2),
                    "avg_inference_time_ms": round(view3_avg_infer_time, 2),
                    "inference_count": view3_infer_count
                }
            
            if "view4" in active_views:
                throughput_data["models"]["view4"] = {
                    "model": view4_model,
                    "execution": view4_mode,
                    "throughput_fps": round(view4_avg_fps, 2),
                    "avg_inference_time_ms": round(view4_avg_infer_time, 2),
                    "inference_count": view4_infer_count
                }
            
            # Save to JSON file
            with open("result_throughput.json", "w", encoding="utf-8") as f:
                json.dump(throughput_data, f, indent=4, ensure_ascii=False)
                
            print("[Shutdown] Throughput data saved to result_throughput.json")
        except Exception as e:
            print(f"[Shutdown ERROR] Failed to save throughput data: {e}")