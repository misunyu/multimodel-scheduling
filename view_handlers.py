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
import demo_render as dr

class ModelSignals(QObject):
    """Signal class for updating model views."""
    update_view1_display = pyqtSignal(QPixmap)
    update_view2_display = pyqtSignal(QPixmap)
    update_view3_display = pyqtSignal(QPixmap)
    update_view4_display = pyqtSignal(QPixmap)

class ViewHandler:
    """Base class for handling model views."""
    
    def __init__(self, view_name, model_settings, frame_queue, result_queue,
                 shutdown_flag, model_signals, views_without_model=None,
                 display_slot=None):
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
        
        # Display signal for this view. `display_slot` is the UI tile this view was
        # assigned (view1..view4), or None when the model is not visualizable or no
        # tile was left. A view without a slot runs headless: it still executes and
        # still contributes to the statistics, it is just not drawn.
        self.display_slot = display_slot
        self.update_signal = (getattr(model_signals, f"update_{display_slot}_display", None)
                              if display_slot else None)
        self.headless = self.update_signal is None
        
        # Get model type and execution mode
        self.model_type = model_settings.get(view_name, {}).get("model", "")
        self.execution_mode = model_settings.get(view_name, {}).get("execution", "cpu")

        # Per-view request deadline (period-based): deadline_ms = 1000/infps * factor.
        # A request whose end-to-end latency exceeds this is a deadline miss.
        cfg = model_settings.get(view_name, {}) or {}
        try:
            from model_registry import DEADLINE_FACTOR as _DF
        except Exception:
            _DF = 1.0
        if cfg.get("deadline_ms"):
            self.deadline_ms = float(cfg["deadline_ms"])
        else:
            infps = cfg.get("infps", None)
            try:
                infps = float(infps) if infps else 0.0
            except Exception:
                infps = 0.0
            self.deadline_ms = (1000.0 / infps) * float(_DF) if infps > 0 else float("inf")
        # Count of completed requests whose latency exceeded the deadline.
        self.late_count = 0

    def start_display_thread(self):
        """Start the display thread for this view."""
        thread = threading.Thread(target=self.display_frames, daemon=True)
        thread.start()
        return thread
        
    def display_frames(self):
        """Display frames from the result queue. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement display_frames")
        
    def reset_stats(self):
        """Reset all collected statistics (used to start measurement after warmup)."""
        self.total_infer_time = 0.0
        self.infer_count = 0
        self.avg_infer_time = 0.0
        self.avg_fps = 0.0
        self.total_wait_ms = 0.0
        self.wait_count = 0
        self.avg_wait_ms = 0.0
        self.late_count = 0

    def note_latency(self, latency_ms):
        """Record a completed request's end-to-end latency; count deadline misses."""
        try:
            if latency_ms is not None and float(latency_ms) > self.deadline_ms:
                self.late_count += 1
        except Exception:
            pass

    # ---- demo rendering helpers -------------------------------------------------
    def emit_image(self, bgr):
        """Push a BGR frame to this view's QLabel. Safe to call when headless."""
        if self.update_signal is None or bgr is None:
            return
        try:
            pixmap = convert_cv_to_qt(bgr)
            if not pixmap.isNull():
                self.update_signal.emit(pixmap)
        except Exception as e:
            print(f"[{self.view_name} render ERROR] {e}")

    def metric_text(self):
        """What the header shows on the right: FPS for vision views."""
        return f"{self.avg_fps:5.1f} FPS" if self.avg_fps else "--"

    def show_waiting(self, message="Waiting for input..."):
        """Paint a labelled placeholder until the first real frame lands.

        Model load takes seconds; an empty tile during it is indistinguishable from
        a crashed view. Once frames flow this becomes a no-op.
        """
        if self.headless or getattr(self, '_got_frame', False):
            return
        now = time.time()
        if now - getattr(self, '_last_wait_paint', 0.0) < 1.0:
            return
        self._last_wait_paint = now
        self.emit_image(dr.placeholder(self.model_type, self.execution_mode, message))

    def show_error(self, message):
        """A failed view still shows its model, device and the reason -- not black."""
        print(f"[{self.view_name}] view error: {message}")
        self.last_error = str(message)
        self.emit_image(dr.placeholder(self.model_type, self.execution_mode,
                                       "Load failed" if "FATAL" in str(message) or
                                       "Error" in str(message) else str(message)[:40]))

    def handle_control(self, item):
        """True if `item` was a control message (error) rather than a result.

        Test the tag with isinstance first: a result tuple starts with a numpy frame,
        and `frame == "error"` is an elementwise compare whose truth value raises.
        """
        if (isinstance(item, tuple) and item and isinstance(item[0], str)
                and item[0] == "error"):
            self.show_error(item[1] if len(item) > 1 else "unknown error")
            return True
        return False

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
            if self.update_signal is not None:
                pixmap = convert_cv_to_qt(create_x_image())
                if not pixmap.isNull():
                    self.update_signal.emit(pixmap)
            return
            
        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                # (frame, infer_time, wait_ms, latency_ms)
                item = self.result_queue.get(timeout=1)
                if self.handle_control(item):
                    continue
                latency_ms = None
                if isinstance(item, tuple) and len(item) == 4:
                    frame, infer_time, wait_ms, latency_ms = item
                elif isinstance(item, tuple) and len(item) == 3:
                    frame, infer_time, wait_ms = item
                else:
                    frame, infer_time = item
                    wait_ms = 0.0
            except queue.Empty:
                self.show_waiting()
                continue
            except BrokenPipeError:
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except (EOFError, OSError) as e:
                print(f"[{self.view_name} Queue ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except Exception as e:
                print(f"[{self.view_name} ERROR] {e}")
                continue
                
            try:
                self.update_stats(self.model_type, infer_time)
                self.note_latency(latency_ms)
                if wait_ms is not None:
                    self.total_wait_ms += float(wait_ms)
                    self.wait_count += 1
                    self.avg_wait_ms = self.total_wait_ms / self.wait_count if self.wait_count > 0 else 0.0
                if self.headless:
                    continue
                # The worker already drew the boxes; add the model/device/FPS header.
                self._got_frame = True
                self.emit_image(dr.draw_header(frame, self.model_type,
                                               self.execution_mode, self.metric_text()))
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
            if self.update_signal is not None:
                pixmap = convert_cv_to_qt(create_x_image())
                if not pixmap.isNull():
                    self.update_signal.emit(pixmap)
            return
            
        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                # (frame, top5, infer_time, latency_ms) -- top5: [(label, prob), ...]
                item = self.result_queue.get(timeout=1)
                if self.handle_control(item):
                    continue
                latency_ms = None
                if isinstance(item, tuple) and len(item) == 4:
                    frame, top5, infer_time, latency_ms = item
                else:
                    frame, top5, infer_time = item
                if isinstance(top5, str):     # older workers sent just the top-1 name
                    top5 = [(top5, 1.0)]
            except queue.Empty:
                self.show_waiting()
                continue
            except BrokenPipeError:
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except (EOFError, OSError) as e:
                print(f"[{self.view_name} Queue ERROR] {e}")
                if self.shutdown_flag.is_set() or global_exit_flag:
                    break
                continue
            except Exception as e:
                print(f"[{self.view_name} ERROR] {e}")
                continue
                
            try:
                self.update_stats(self.model_type, infer_time)
                self.note_latency(latency_ms)
                if self.headless:
                    continue
                self._got_frame = True
                self.emit_image(dr.top5_panel(frame, top5, self.model_type,
                                              self.execution_mode, self.metric_text()))
            except Exception as e:
                print(f"[{self.view_name} Display ERROR] {e}")

class LLMViewHandler(ViewHandler):
    """Handler for LLM / VLM views: renders the generation as it streams.

    The worker sends ("start", frame, prompt) / ("token", piece) / ("done", ...).
    Tokens are appended to a buffer and the card is repainted at a capped rate, so
    the text types itself out (a finished answer arriving in one frame looks frozen)
    without repainting once per token and starving the other views.

    VLM cards show the input image beside the text; LLM cards use the full width.
    Stats are as before: `avg_fps` is generations/sec, `avg_tokens_per_s` the token
    throughput.
    """

    _REPAINT_HZ = 12.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens_per_s = 0.0
        self.tok_count = 0
        self.avg_tokens_per_s = 0.0
        self.text = ""
        self.prompt = None
        self.image = None
        self.generating = False
        self._last_paint = 0.0

    def reset_stats(self):
        super().reset_stats()
        self.total_tokens_per_s = 0.0
        self.tok_count = 0
        self.avg_tokens_per_s = 0.0

    def metric_text(self):
        return f"{self.avg_tokens_per_s:5.1f} tok/s" if self.avg_tokens_per_s else "generating..."

    def _paint(self, force=False):
        if self.headless:
            return
        now = time.time()
        if not force and (now - self._last_paint) < (1.0 / self._REPAINT_HZ):
            return
        self._last_paint = now
        self.emit_image(dr.text_card(self.text, self.model_type, self.execution_mode,
                                     self.metric_text(), image=self.image,
                                     generating=self.generating, prompt=self.prompt))

    def display_frames(self):
        if self.view_name in self.views_without_model:
            return
        # Say something before the first token: model load can take seconds and an
        # empty tile during it looks like a failure.
        self.generating = True
        self._paint(force=True)

        while not self.shutdown_flag.is_set():
            try:
                item = self.result_queue.get(timeout=1)
            except queue.Empty:
                self._paint()          # keep the cursor blinking while we wait
                continue
            except (BrokenPipeError, EOFError, OSError):
                if self.shutdown_flag.is_set():
                    break
                continue
            except Exception as e:
                print(f"[{self.view_name} LLM ERROR] {e}")
                continue
            try:
                if self.handle_control(item):
                    continue
                tag = (item[0] if isinstance(item, tuple) and item
                       and isinstance(item[0], str) else None)

                if tag == "start":
                    _, frame, prompt = item
                    self.text = ""
                    self.prompt = prompt
                    self.image = frame if frame is not None else None
                    self.generating = True
                    self._paint(force=True)
                elif tag == "token":
                    self.text += str(item[1])
                    self._paint()
                elif tag == "done":
                    _, gen_ms, tokens_per_s, latency_ms = item
                    self.generating = False
                    self.update_stats(self.model_type, gen_ms)
                    self.note_latency(latency_ms)
                    self.total_tokens_per_s += float(tokens_per_s or 0.0)
                    self.tok_count += 1
                    self.avg_tokens_per_s = (self.total_tokens_per_s / self.tok_count
                                             if self.tok_count else 0.0)
                    self._paint(force=True)
                else:
                    # Legacy shape: (None, gen_ms, tokens_per_s[, latency_ms])
                    latency_ms = item[3] if len(item) == 4 else None
                    _, gen_ms, tokens_per_s = item[0], item[1], item[2]
                    self.update_stats(self.model_type, gen_ms)
                    self.note_latency(latency_ms)
                    self.total_tokens_per_s += float(tokens_per_s or 0.0)
                    self.tok_count += 1
                    self.avg_tokens_per_s = (self.total_tokens_per_s / self.tok_count
                                             if self.tok_count else 0.0)
            except Exception as e:
                print(f"[{self.view_name} LLM stat ERROR] {e}")


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
        self.yolo_views = set(yolo_views or [])
        self.shutdown_flag = shutdown_flag
        self.model_settings = model_settings or {}
        # Track dropped frames per view due to full queue (these are deadline misses:
        # a request that cannot be admitted never completes in time).
        self.drop_counts = {v: 0 for v in view_frame_queues.keys()}
        # Total requests offered per view (admitted + dropped) — denominator for miss rate.
        self.offered_counts = {v: 0 for v in view_frame_queues.keys()}
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
                self.offered_counts[v] = 0
        except Exception:
            pass
        # We intentionally do not reset last_enqueue_ts to preserve pacing; only metrics reset is needed.
        
    def feed_queues(self):
        """Generate requests at each view's infps, decoupled from source frame rate.

        Pacing is driven by the wall clock, not by frame arrival, so target rates
        above the source FPS are met by duplicating the most recent frame. Every
        due enqueue counts as an offered request; a full queue counts as a drop
        (an inadmissible request = a deadline miss).
        """
        global_exit_flag = False
        last_frame = None
        # Small tick to support high rates (e.g., 4ms tick -> up to ~250 req/s pacing).
        min_sleep = 0.002

        while not self.shutdown_flag.is_set() and not global_exit_flag:
            try:
                # Refresh the latest frame without blocking (drain any backlog).
                try:
                    while True:
                        last_frame = self.video_queue.get_nowait()
                except queue.Empty:
                    pass
                if last_frame is None:
                    try:
                        last_frame = self.video_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                now = time.time()
                for view_name in list(self.yolo_views):
                    if view_name not in self.view_frame_queues:
                        continue
                    interval = self.view_intervals.get(view_name)  # None -> every tick
                    last_ts = self.last_enqueue_ts.get(view_name, 0.0)
                    if (interval is None) or ((now - last_ts) >= interval):
                        self.last_enqueue_ts[view_name] = now
                        try:
                            self.offered_counts[view_name] += 1
                        except Exception:
                            pass
                        frame_q = self.view_frame_queues[view_name]
                        try:
                            frame_q.put_nowait((last_frame.copy(), now))
                        except queue.Full:
                            try:
                                self.drop_counts[view_name] += 1
                            except Exception:
                                pass
                        except (EOFError, BrokenPipeError, OSError):
                            pass
                time.sleep(min_sleep)
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
        self.resnet_views = set(resnet_views or [])
        self.shutdown_flag = shutdown_flag
        self.model_settings = model_settings or {}
        self.default_interval_sec = max(0.0, float(default_interval_sec) if default_interval_sec else 0.5)
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
            # Sorted so the cycle is deterministic across views and runs.
            self._images = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
        except Exception as e:
            print(f"[ResnetImageFeeder] Failed to list images in {image_dir}: {e}")
            self._images = []
        if not self._images:
            print(f"[ResnetImageFeeder] No images found in {image_dir}. Feeder will be idle.")
        else:
            print(f"[ResnetImageFeeder] Cycling {len(self._images)} images from {image_dir}")

    def start_feed_thread(self):
        thread = threading.Thread(target=self.feed_queues, daemon=True)
        thread.start()
        return thread

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
                    q = self.view_frame_queues.get(view_name)
                    if q is None:
                        continue
                    interval = self.view_intervals.get(view_name, self.default_interval_sec)
                    if interval is None:
                        # If interval is None (infps <= 0), skip feeding this view
                        continue
                    if (now - last_ts.get(view_name, 0.0)) < interval:
                        continue
                    img = self._next_image(view_name)
                    if img is None:
                        continue
                    try:
                        q.put_nowait((img, now))
                        last_ts[view_name] = now
                    except queue.Full:
                        pass
                    except (EOFError, BrokenPipeError, OSError):
                        pass
            except Exception as e:
                print(f"[ResnetImageFeeder ERROR] {e}")
            time.sleep(min_sleep)