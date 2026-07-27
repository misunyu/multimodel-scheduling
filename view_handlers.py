"""
View handling components for the multimodel scheduling application.
"""
import os
import threading
import queue
import time
from collections import deque

# C3: when FSRR_RATE_REPLICATE=1, feeders replicate frames to hit the target
# arrival rate lambda (= infps) even above the source/loop rate, so lambda is
# actually controllable. Each replicated frame is a real inference (the worker
# runs a full forward pass per copy), so lambda_actual == target with real load.
# Default (unset) keeps the prior throttle-only behavior.
_RATE_REPLICATE = os.environ.get("FSRR_RATE_REPLICATE", "0") == "1"

# C3/B2: per-frame end-to-end latency logging (measurement only). When
# FSRR_PERFRAME_LOG is a path, each completed frame appends one row
# (view, model, infer, wait, e2e=infer+wait, deadline=L_SLO, late=e2e>deadline)
# so miss-rate late-component and p99/p999 tail can be computed. Drops are counted
# separately by the feeders. Default (unset) = no per-frame logging.
_PERFRAME_PATH = os.environ.get("FSRR_PERFRAME_LOG", "")
_perframe_lock = threading.Lock()
_perframe_fh = None


def _perframe_log(view, model, infer_ms, wait_ms, deadline_ms):
    if not _PERFRAME_PATH:
        return
    try:
        e2e = float(infer_ms) + float(wait_ms or 0.0)
        late = 1 if (deadline_ms and e2e > float(deadline_ms)) else 0
        ts = time.time()  # epoch seconds, for offline Delta-window recomputation
        global _perframe_fh
        with _perframe_lock:
            if _perframe_fh is None:
                _perframe_fh = open(_PERFRAME_PATH, "a", buffering=1)
                _perframe_fh.write("ts,view,model,infer_ms,wait_ms,e2e_ms,deadline_ms,late\n")
            _perframe_fh.write(f"{ts:.4f},{view},{model},{float(infer_ms):.2f},{float(wait_ms or 0.0):.2f},"
                               f"{e2e:.2f},{float(deadline_ms or 0.0):.2f},{late}\n")
    except Exception:
        pass


def _deadline_ms(model_settings, view_name):
    """L_SLO for a view: slo_ms if set, else 1000/infps."""
    try:
        s = (model_settings or {}).get(view_name, {}) or {}
        if s.get("slo_ms") is not None:
            return float(s["slo_ms"])
        infps = float(s.get("infps", 10.0) or 10.0)
        return 1000.0 / infps if infps > 0 else 100.0
    except Exception:
        return 100.0
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

        # Sliding Delta-window of recent per-frame end-to-end latencies
        # (monotonic_ts, e2e_ms). This is the paper's ell_i(t): the mean e2e over
        # [t-Delta, t]. The lifetime avg_infer/avg_wait above are kept for
        # throughput reporting but MUST NOT be used for v(t) (they never reset on
        # hot-swap and contaminate V(t) with the previous phase).
        self._lat_win = deque()

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
        
    def record_latency(self, e2e_ms):
        """Append one completed frame's end-to-end latency to the Delta window."""
        try:
            self._lat_win.append((time.monotonic(), float(e2e_ms)))
        except Exception:
            pass

    def windowed_latency(self, delta_s):
        """Mean end-to-end latency over the last `delta_s` seconds (paper ell_i).
        Returns None if no frame landed in the window (caller skips the view so
        it is not diluted toward zero)."""
        try:
            now = time.monotonic()
            cutoff = now - float(delta_s)
            w = self._lat_win
            while w and w[0][0] < cutoff:
                w.popleft()
            if not w:
                return None
            return sum(e for _, e in w) / len(w)
        except Exception:
            return None

    def reset_stats(self):
        """Reset all collected statistics (used to start measurement after warmup)."""
        self.total_infer_time = 0.0
        self.infer_count = 0
        self.avg_infer_time = 0.0
        self.avg_fps = 0.0
        self.total_wait_ms = 0.0
        self.wait_count = 0
        try:
            self._lat_win.clear()
        except Exception:
            pass
        self.avg_wait_ms = 0.0
        
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
        first_display = True
        
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
                    if first_display:
                        first_display = False
                    self.update_stats(self.model_type, infer_time)
                    _perframe_log(self.view_name, self.model_type, infer_time, wait_ms,
                                  _deadline_ms(self.model_settings, self.view_name))
                    # Record per-frame end-to-end latency into the Delta window
                    # (paper ell_i). Lifetime avg_wait/avg_infer stay for reporting.
                    self.record_latency(float(infer_time) + float(wait_ms or 0.0))
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
        first_display = True
        
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
                # For ResNet models, the result queue contains
                # (frame, class_name, infer_time[, wait_ms]). wait_ms was added
                # so v(t) can include queue-wait latency.
                item = self.result_queue.get(timeout=1)
                if isinstance(item, tuple) and len(item) == 4:
                    frame, class_name, infer_time, wait_ms = item
                else:
                    frame, class_name, infer_time = item
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
                    if first_display:
                        first_display = False
                    self.update_stats(self.model_type, infer_time)
                    _perframe_log(self.view_name, self.model_type, infer_time, wait_ms,
                                  _deadline_ms(self.model_settings, self.view_name))
                    # Record per-frame end-to-end latency into the Delta window
                    # (paper ell_i). Lifetime avg_wait/avg_infer stay for reporting.
                    self.record_latency(float(infer_time) + float(wait_ms or 0.0))
                    # Update wait statistics if available
                    if wait_ms is not None:
                        self.total_wait_ms += float(wait_ms)
                        self.wait_count += 1
                        self.avg_wait_ms = self.total_wait_ms / self.wait_count if self.wait_count > 0 else 0.0
                else:
                    print(f"[{self.view_name}] Pixmap is null")
            except Exception as e:
                print(f"[{self.view_name} Display ERROR] {e}")

class VideoFeeder:
    """Class for feeding video frames to model queues, honoring per-view input FPS (infps)."""

    def __init__(self, video_queue, view_frame_queues, yolo_views, shutdown_flag, model_settings=None):
        """
        Initialize the video feeder.

        Args:
            video_queue: Queue containing video frames
            view_frame_queues: Dictionary mapping view names to frame queues
            yolo_views: Set of views running YOLO models
            shutdown_flag: Flag to signal shutdown
            model_settings: Dict view_name -> {model, execution, infps}
        """
        self.video_queue = video_queue
        self.view_frame_queues = view_frame_queues
        self._queue_lock = threading.Lock()  # Protects view_frame_queues for adaptive deploy swaps
        self.yolo_views = set(yolo_views or [])
        self.shutdown_flag = shutdown_flag
        self.model_settings = model_settings or {}
        # Track dropped frames per view due to full queue
        self.drop_counts = {v: 0 for v in view_frame_queues.keys()}
        # Track successfully enqueued frames per view (for the transition
        # frame-accounting audit: enqueued == completed + dropped + residual).
        # Pure counter, no behavior change.
        self.enqueue_counts = {v: 0 for v in view_frame_queues.keys()}
        # Compute per-view enqueue intervals from infps
        self.view_intervals = {}
        for v in self.yolo_views:
            try:
                infps = self.model_settings.get(v, {}).get("infps", None)
                if infps is None:
                    interval = None
                else:
                    inf = float(infps)
                    interval = (1.0 / inf) if inf > 0 else None
                self.view_intervals[v] = interval
            except Exception:
                self.view_intervals[v] = None
        # Last enqueue timestamps per view
        self.last_enqueue_ts = {v: 0.0 for v in self.yolo_views}
        
    def update_intervals(self, model_settings):
        """Recalculate per-view enqueue intervals from updated model_settings at runtime."""
        self.model_settings = model_settings or self.model_settings
        for v in self.yolo_views:
            try:
                infps = self.model_settings.get(v, {}).get("infps", None)
                if infps is None:
                    interval = None
                else:
                    inf = float(infps)
                    interval = (1.0 / inf) if inf > 0 else None
                self.view_intervals[v] = interval
            except Exception:
                self.view_intervals[v] = None

    def swap_view_queue(self, view_name, new_queue):
        """Thread-safe swap of the frame queue for a view (used by adaptive deploy)."""
        with self._queue_lock:
            self.view_frame_queues[view_name] = new_queue
            if view_name not in self.drop_counts:
                self.drop_counts[view_name] = 0
            print(f"[VideoFeeder] Swapped queue for {view_name}")

    def start_feed_thread(self):
        """Start the thread for feeding video frames to model queues."""
        thread = threading.Thread(target=self.feed_queues, daemon=True)
        thread.start()
        return thread

    def reset_counters(self):
        """Reset drop counters and any per-view timestamps (for warmup handling)."""
        try:
            for v in list(self.drop_counts.keys()):
                self.drop_counts[v] = 0
        except Exception:
            pass
        # We intentionally do not reset last_enqueue_ts to preserve pacing; only metrics reset is needed.

    def feed_queues(self):
        """Feed video frames to model queues, enforcing per-view infps intervals when provided."""
        global_exit_flag = False  # This should be passed from the main application
        # Source video FPS only limits max rate; per-view infps throttles enqueueing
        source_fps = 30.0
        try:
            import cv2
            cap = cv2.VideoCapture("./stockholm_1280x720.mp4")
            fps_read = cap.get(cv2.CAP_PROP_FPS)
            if fps_read > 1.0:
                source_fps = fps_read
            cap.release()
        except Exception as e:
            print(f"[feed_queues] Failed to read FPS, using default 30.0: {e}")
        min_sleep = max(0.001, 1.0 / (source_fps * 2.0))  # small sleep to avoid busy loop

        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                frame = self.video_queue.get(timeout=1)
                now = time.time()
                # Feed frames to all views that are running YOLO models
                for view_name in list(self.yolo_views):
                    with self._queue_lock:
                        frame_q = self.view_frame_queues.get(view_name)
                    if frame_q is None:
                        continue
                    interval = self.view_intervals.get(view_name)  # None means no throttle (enqueue every frame)
                    last_ts = self.last_enqueue_ts.get(view_name, 0.0)
                    if interval is None:
                        n_copies = 1
                        self.last_enqueue_ts[view_name] = now
                    elif _RATE_REPLICATE:
                        # Enqueue as many copies as the elapsed time earns at rate
                        # 1/interval, so the achieved rate == target infps even when
                        # infps > source fps (replication). Advance last_ts by whole
                        # intervals so no rate is lost or gained over time.
                        if last_ts <= 0.0:
                            n_copies = 1
                            self.last_enqueue_ts[view_name] = now
                        else:
                            n_copies = int((now - last_ts) / interval)
                            self.last_enqueue_ts[view_name] = last_ts + n_copies * interval
                    else:
                        n_copies = 1 if ((now - last_ts) >= interval) else 0
                        if n_copies:
                            self.last_enqueue_ts[view_name] = now
                    for _ in range(n_copies):
                        try:
                            frame_q.put_nowait((frame.copy(), now))
                            try:
                                self.enqueue_counts[view_name] += 1
                            except Exception:
                                pass
                        except queue.Full:
                            try:
                                self.drop_counts[view_name] += 1
                            except Exception:
                                pass
                        except (EOFError, BrokenPipeError, OSError):
                            pass
                time.sleep(min_sleep)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[feed_queues ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break

class ResnetImageFeeder:
    """Class for feeding image samples to ResNet model input queues honoring per-view infps."""

    def __init__(self, image_dir, view_frame_queues, resnet_views, shutdown_flag, model_settings=None, default_interval_sec=0.5):
        """
        Args:
            image_dir: Directory of sample images to cycle through
            view_frame_queues: Dict view_name -> frame_queue
            resnet_views: Set of view_names that are running ResNet
            shutdown_flag: Event/flag to stop feeding
            model_settings: Dict view_name -> {model, execution, infps}
            default_interval_sec: Fallback interval if infps not provided
        """
        self.image_dir = image_dir
        self.view_frame_queues = view_frame_queues
        self._queue_lock = threading.Lock()  # Protects view_frame_queues for adaptive deploy swaps
        self.resnet_views = set(resnet_views or [])
        self.shutdown_flag = shutdown_flag
        self.model_settings = model_settings or {}
        self.default_interval_sec = max(0.0, float(default_interval_sec) if default_interval_sec else 0.5)
        # Track dropped frames per view due to full queue (for CNN/ResNet)
        try:
            self.drop_counts = {v: 0 for v in (view_frame_queues or {}).keys()}
            self.enqueue_counts = {v: 0 for v in (view_frame_queues or {}).keys()}
        except Exception:
            self.drop_counts = {}
            self.enqueue_counts = {}
        # Compute per-view interval from infps
        self.view_intervals = {}
        for v in self.resnet_views:
            try:
                infps = self.model_settings.get(v, {}).get("infps", None)
                if infps is None:
                    interval = self.default_interval_sec
                else:
                    inf = float(infps)
                    interval = (1.0 / inf) if inf > 0 else None
                self.view_intervals[v] = interval
            except Exception:
                self.view_intervals[v] = self.default_interval_sec
        self._images = []
        self._index_map = {}
        try:
            import os
            self._images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        except Exception as e:
            print(f"[ResnetImageFeeder] Failed to list images in {image_dir}: {e}")
            self._images = []
        if not self._images:
            print("[ResnetImageFeeder] No images found. Feeder will be idle.")

    def update_intervals(self, model_settings):
        """Recalculate per-view enqueue intervals from updated model_settings at runtime."""
        self.model_settings = model_settings or self.model_settings
        for v in self.resnet_views:
            try:
                infps = self.model_settings.get(v, {}).get("infps", None)
                if infps is None:
                    interval = self.default_interval_sec
                else:
                    inf = float(infps)
                    interval = (1.0 / inf) if inf > 0 else None
                self.view_intervals[v] = interval
            except Exception:
                self.view_intervals[v] = self.default_interval_sec

    def swap_view_queue(self, view_name, new_queue):
        """Thread-safe swap of the frame queue for a view (used by adaptive deploy)."""
        with self._queue_lock:
            self.view_frame_queues[view_name] = new_queue
            if view_name not in self.drop_counts:
                self.drop_counts[view_name] = 0
            print(f"[ResnetImageFeeder] Swapped queue for {view_name}")

    def start_feed_thread(self):
        thread = threading.Thread(target=self.feed_queues, daemon=True)
        thread.start()
        return thread

    def reset_counters(self):
        """Reset drop counters (used at measurement window start)."""
        try:
            for v in list(self.drop_counts.keys()):
                self.drop_counts[v] = 0
        except Exception:
            pass

    def _next_image(self, view_name):
        if not self._images:
            return None
        idx = self._index_map.get(view_name, 0)
        if idx >= len(self._images):
            idx = 0
        path = self._images[idx]
        self._index_map[view_name] = idx + 1
        try:
            import cv2
            img = cv2.imread(path)
            return img
        except Exception as e:
            print(f"[ResnetImageFeeder] Failed to read image {path}: {e}")
            return None

    def feed_queues(self):
        global_exit_flag = False
        # Track last enqueue time per view
        last_ts = {v: 0.0 for v in self.resnet_views}
        min_sleep = 0.005
        while not self.shutdown_flag.is_set() and not global_exit_flag:
            now = time.time()
            try:
                for view_name in list(self.resnet_views):
                    with self._queue_lock:
                        q = self.view_frame_queues.get(view_name)
                    if q is None:
                        continue
                    interval = self.view_intervals.get(view_name, self.default_interval_sec)
                    if interval is None:
                        # If interval is None (infps <= 0), skip feeding this view
                        continue
                    _lt = last_ts.get(view_name, 0.0)
                    if _RATE_REPLICATE:
                        if _lt <= 0.0:
                            n_copies = 1
                            last_ts[view_name] = now
                        else:
                            n_copies = int((now - _lt) / interval)
                            last_ts[view_name] = _lt + n_copies * interval
                    else:
                        n_copies = 1 if ((now - _lt) >= interval) else 0
                        if n_copies:
                            last_ts[view_name] = now
                    for _ in range(n_copies):
                        img = self._next_image(view_name)
                        if img is None:
                            break
                        try:
                            q.put_nowait((img, now))
                            try:
                                self.enqueue_counts[view_name] = int(self.enqueue_counts.get(view_name, 0)) + 1
                            except Exception:
                                pass
                        except queue.Full:
                            try:
                                self.drop_counts[view_name] = int(self.drop_counts.get(view_name, 0)) + 1
                            except Exception:
                                pass
                        except (EOFError, BrokenPipeError, OSError):
                            pass
            except Exception as e:
                print(f"[ResnetImageFeeder ERROR] {e}")
            time.sleep(min_sleep)