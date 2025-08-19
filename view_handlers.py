"""
View handling components for the multimodel scheduling application.
"""
import threading
import queue
import time
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPixmap

# Import local modules
from utils import create_x_image, convert_cv_to_qt, async_log

class ModelSignals(QObject):
    """Signal class for updating model views."""
    update_view1_display = pyqtSignal(QPixmap)
    update_view2_display = pyqtSignal(QPixmap)
    update_view3_display = pyqtSignal(QPixmap)
    update_view4_display = pyqtSignal(QPixmap)

class ViewHandler:
    """Base class for handling model views."""
    
    def __init__(self, view_name, model_settings, frame_queue, result_queue, 
                 shutdown_flag, model_signals, views_without_model=None):
        """
        Initialize the view handler.
        
        Args:
            view_name: Name of the view (view1, view2, etc.)
            model_settings: Dictionary of model settings
            frame_queue: Queue for input frames
            result_queue: Queue for output results
            shutdown_flag: Flag to signal shutdown
            model_signals: ModelSignals object for updating displays
            views_without_model: Set of views without specified models
        """
        self.view_name = view_name
        self.model_settings = model_settings
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.shutdown_flag = shutdown_flag
        self.model_signals = model_signals
        self.views_without_model = views_without_model or set()
        
        # Statistics
        self.total_infer_time = 0.0
        self.infer_count = 0
        self.avg_infer_time = 0.0
        self.avg_fps = 0.0
        
        # Waiting time stats (time from enqueue to start of preprocessing)
        self.total_wait_ms = 0.0
        self.wait_count = 0
        self.avg_wait_ms = 0.0
        
        # Get the signal method for this view
        signal_method_name = f"update_{view_name}_display"
        self.update_signal = getattr(model_signals, signal_method_name)
        
        # Get model type and execution mode
        self.model_type = model_settings.get(view_name, {}).get("model", "")
        self.execution_mode = model_settings.get(view_name, {}).get("execution", "cpu")
        
    def start_display_thread(self):
        """Start the display thread for this view."""
        thread = threading.Thread(target=self.display_frames, daemon=True)
        thread.start()
        return thread
        
    def display_frames(self):
        """Display frames from the result queue. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement display_frames")
        
    def update_stats(self, model_name, infer_time, log_enabled=0):
        """
        Update performance statistics for this view.
        
        Args:
            model_name: Name of the model
            infer_time: Inference time in milliseconds
            log_enabled: Flag to control whether logging is enabled
        """
        self.total_infer_time += infer_time
        self.infer_count += 1
        self.avg_infer_time = self.total_infer_time / self.infer_count
        self.avg_fps = 1000.0 / self.avg_infer_time if self.avg_infer_time > 0 else 0.0
        
        # Log performance data
        async_log(model_name, infer_time, self.avg_fps, log_enabled)
        
        return self.avg_fps, self.avg_infer_time

class YoloViewHandler(ViewHandler):
    """Handler for YOLO model views."""
    
    def display_frames(self):
        """Display frames from the YOLO model."""
        global_exit_flag = False  # This should be passed from the main application
        
        # Check if this view has a specified model
        if self.view_name in self.views_without_model:
            # Display X image for this view
            x_image = create_x_image()
            pixmap = convert_cv_to_qt(x_image)
            if not pixmap.isNull():
                self.update_signal.emit(pixmap)
            return
            
        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                # For YOLO models, the result queue contains (frame, infer_time[, wait_ms])
                item = self.result_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 3:
                    frame, infer_time, wait_ms = item
                else:
                    frame, infer_time = item
                    wait_ms = 0.0
            except queue.Empty:
                continue
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[{self.view_name} Queue ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except Exception as e:
                print(f"[{self.view_name} ERROR] {e}")
                continue
                
            try:
                pixmap = convert_cv_to_qt(frame)
                if not pixmap.isNull():
                    self.update_signal.emit(pixmap)
                    self.update_stats(self.model_type, infer_time)
                    # Update wait statistics if available
                    if wait_ms is not None:
                        self.total_wait_ms += float(wait_ms)
                        self.wait_count += 1
                        self.avg_wait_ms = self.total_wait_ms / self.wait_count if self.wait_count > 0 else 0.0
                else:
                    print(f"[{self.view_name}] Pixmap is null")
            except Exception as e:
                print(f"[{self.view_name} Display ERROR] {e}")

class ResNetViewHandler(ViewHandler):
    """Handler for ResNet model views."""
    
    def display_frames(self):
        """Display frames from the ResNet model."""
        global_exit_flag = False  # This should be passed from the main application
        
        # Check if this view has a specified model
        if self.view_name in self.views_without_model:
            # Display X image for this view
            x_image = create_x_image()
            pixmap = convert_cv_to_qt(x_image)
            if not pixmap.isNull():
                self.update_signal.emit(pixmap)
            return
            
        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                # For ResNet models, the result queue contains (frame, class_name, infer_time)
                frame, class_name, infer_time = self.result_queue.get(timeout=1)
            except queue.Empty:
                continue
            except (EOFError, BrokenPipeError, OSError) as e:
                print(f"[{self.view_name} Queue ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except Exception as e:
                print(f"[{self.view_name} ERROR] {e}")
                continue
                
            try:
                pixmap = convert_cv_to_qt(frame)
                if not pixmap.isNull():
                    self.update_signal.emit(pixmap)
                    self.update_stats(self.model_type, infer_time)
                else:
                    print(f"[{self.view_name}] Pixmap is null")
            except Exception as e:
                print(f"[{self.view_name} Display ERROR] {e}")

class VideoFeeder:
    """Class for feeding video frames to model queues."""
    
    def __init__(self, video_queue, view_frame_queues, yolo_views, shutdown_flag):
        """
        Initialize the video feeder.
        
        Args:
            video_queue: Queue containing video frames
            view_frame_queues: Dictionary mapping view names to frame queues
            yolo_views: Set of views running YOLO models
            shutdown_flag: Flag to signal shutdown
        """
        self.video_queue = video_queue
        self.view_frame_queues = view_frame_queues
        self.yolo_views = yolo_views
        self.shutdown_flag = shutdown_flag
        # Track dropped frames per view due to full queue
        self.drop_counts = {v: 0 for v in view_frame_queues.keys()}
        
    def start_feed_thread(self):
        """Start the thread for feeding video frames to model queues."""
        thread = threading.Thread(target=self.feed_queues, daemon=True)
        thread.start()
        return thread
        
    def feed_queues(self):
        """Feed video frames to model queues."""
        global_exit_flag = False  # This should be passed from the main application
        fps = 30.0
        
        try:
            import cv2
            cap = cv2.VideoCapture("./stockholm_1280x720.mp4")
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            if fps_read > 1.0:
                fps = fps_read
            cap.release()
        except Exception as e:
            print(f"[feed_queues] Failed to read FPS, using default 30.0: {e}")

        frame_delay = 1.0 / fps

        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                frame = self.video_queue.get(timeout=1)
                
                # Feed frames to all views that are running YOLO models
                for view_name in self.yolo_views:
                    if view_name in self.view_frame_queues:
                        frame_q = self.view_frame_queues[view_name]
                        try:
                            # Use non-blocking put with enqueue timestamp
                            frame_q.put_nowait((frame.copy(), time.time()))
                        except queue.Full:
                            # Drop frame if the queue is full
                            try:
                                self.drop_counts[view_name] += 1
                            except Exception:
                                pass
                        except (EOFError, BrokenPipeError, OSError):
                            # Queue might be closed during shutdown
                            pass
                
                time.sleep(frame_delay)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[feed_queues ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break