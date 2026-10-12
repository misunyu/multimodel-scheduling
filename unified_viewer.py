"""
Main UnifiedViewer class for the multimodel scheduling application.
"""
import os
import json
import time
import signal
import yaml
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QPushButton, QFrame,
                             QGridLayout, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QEvent
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
import demo_render as dr

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
# Lead whitelist: the four models the demo is built around. Whichever of these are
# running take the 2x2 grid first; the single source of truth, so do not test model
# names for "lead-ness" anywhere else. Cells still free after that are filled with the
# largest of the other running models (model_registry.PARAMS_M), so the grid shows
# four live tiles whenever four models are running.
LEAD_WHITELIST = ("qwen2_vl", "resnet50", "yolo11s", "llama1b")

# The .ui provides four QLabel slots and four display signals: the leads. Every other
# running model is a SUPPORTING view -- it executes and is measured exactly the same,
# and is drawn (small, in the right-hand column) only in exhibition mode.
DISPLAY_SLOTS = ["view1", "view2", "view3", "view4"]

# The right-hand column is this fraction of the window width, whatever the mode or the
# number of models. The left 2x2 gets the rest, so nothing but a window resize can
# change the size of a lead tile.
RIGHT_COL_RATIO = 0.21

# Seconds discarded before measurement starts. 5s was not enough: llama1b's first
# inference lands at 13.4s in the 8-model set, so the window opened while it was still
# loading and the vision models ran uncontended -- which read back as a 15% optimistic
# deadline-miss rate. Measured in DURATION_CONVERGENCE_REPORT.md.
WARMUP_SECONDS = float(os.environ.get("WARMUP_SECONDS", "20"))
MAX_VIEWS = 8


def select_leads(view_models):
    """Pick the (at most four) lead views for a running set.

    `view_models` is [(view_name, model_name), ...] in view order. Returns
    (whitelisted, fill): the whitelisted views in view order, then -- only if fewer
    than four -- the remaining views largest-model-first until four are chosen. Ties
    keep view order. Purely a rendering decision.
    """
    n = len(DISPLAY_SLOTS)
    whitelisted = [v for v, m in view_models if m in LEAD_WHITELIST][:n]
    rest = [(v, m) for v, m in view_models if v not in whitelisted]
    rest.sort(key=lambda vm: -reg.params_m(vm[1]))
    fill = [v for v, _ in rest[:n - len(whitelisted)]]
    return whitelisted, fill


def _fmt_ms(ms):
    """Latency for a strip: milliseconds, or seconds once it stops fitting."""
    try:
        ms = float(ms)
    except Exception:
        return "--"
    return f"{ms / 1000.0:.1f} s" if ms >= 1000.0 else f"{ms:.1f} ms"


class SupportTile(QFrame):
    """A supporting model's small tile: caption (model, device badge, one figure) over
    a scaled-down live rendering."""

    CAPTION_H = 38
    BODY_ASPECT = (dr.VIEW_H - 34) / float(dr.VIEW_W)   # body height / width

    def __init__(self, view_name, model, device, parent=None):
        super().__init__(parent)
        self.view_name = view_name
        self.model = model
        self.device = str(device or "?")
        self.setObjectName(f"support_{view_name}")
        self.setStyleSheet("SupportTile { background-color: #202020; border: 1px solid #555555; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.caption = QLabel(self)
        self.caption.setTextFormat(Qt.RichText)
        self.caption.setFixedHeight(self.CAPTION_H)
        self.caption.setStyleSheet("font-size: 12px; color: #eeeeee; border: none;")
        self.image = QLabel(self)
        self.image.setScaledContents(True)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image.setStyleSheet("border: none;")
        lay.addWidget(self.caption)
        lay.addWidget(self.image, 1)
        self.set_metric("--")

    def set_metric(self, text):
        self.caption.setText(
            f"<b>{self.model}</b> "
            f"<span style='background-color:{dr.device_color_hex(self.device)}; color:#ffffff;'>"
            f"&nbsp;{self.device.upper()}&nbsp;</span><br>{text}")

    @classmethod
    def height_for_width(cls, w):
        return cls.CAPTION_H + 6 + int(round((w - 4) * cls.BODY_ASPECT))


class UnifiedViewer(QMainWindow):
    """The Schedule Executor Display: four lead tiles, Stop All, and the right column.

    Everything the old "System and Model Information" window showed lives here now:
    per-view figures in a strip on each lead tile, the overall figures in the right
    column. The right column has two modes, switchable while the run is going:

      measure (default)  the overall panel; only the four leads are drawn.
      exhibition         compact overall + a small live tile for every other model.

    The mode changes what is DRAWN and nothing else. Every scheduled model is loaded,
    runs and is measured identically in both.
    """

    def __init__(self, schedule_file='model_schedules.yaml', combination_name=None,
                 show_all=False):
        """Initialize the UnifiedViewer.

        Args:
            schedule_file (str): Path to the model scheduling information file.
            combination_name (str|None): Specific combination key to use from the YAML. If None, default logic applies.
            show_all (bool): Start in exhibition mode (render the supporting views too).
        """
        super().__init__()
        uic.loadUi(_asset("schedule_executor_display.ui"), self)

        # Set up signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)

        # Store the schedule file path and requested combination
        self.schedule_file = schedule_file
        self.requested_combination = combination_name

        # Hook the owner (the executor) sets to hear about mode changes, Stop All and
        # window moves: notify(event_name, **fields). Default: nobody is listening.
        self.notify = lambda _event, **_fields: None

        self.rendering_enabled = os.environ.get("DISABLE_VISUALIZATION", "0") != "1"
        self.show_all = bool(show_all) and self.rendering_enabled
        self._live = None                 # last overall/per-view snapshot (1 s tick)
        self._exh_accum = 0.0             # exhibition seconds inside the measured window
        self._exh_since = None
        self._run_t0 = None
        self._await_paint = None          # mode switch waiting for its first repaint
        self._await_support = None

        self.setWindowTitle("Schedule Executor Display")

        # Initialize model settings and views
        self.initialize_model_settings()
        self.initialize_ui_components()
        self.initialize_state_variables()
        # Don't start processes and threads automatically
        # They will be started when start_execution() is called

        # CPU monitoring + the once-a-second refresh of every figure on screen
        self.cpu_timer = QTimer()
        self.cpu_timer.timeout.connect(self.update_cpu_npu_usage)
        self.cpu_timer.start(1000)

        # Display test probe: a machine-readable snapshot of what is on screen.
        if os.environ.get("DISPLAY_PROBE", "0") == "1":
            self._probe_timer = QTimer(self)
            self._probe_timer.timeout.connect(self._emit_probe)
            self._probe_timer.start(500)

    def initialize_model_settings(self):
        """Initialize model settings from YAML configuration."""
        self.model_settings = {}
        self.views_without_model = set()  # Track views without specified models
        # Default combination, can be overridden by requested_combination
        self.current_combination = self.requested_combination or "combination1"

        try:
            # Load configuration from the specified schedule file
            with open(self.schedule_file, "r") as f:
                config = yaml.safe_load(f) or {}

            # Create a mapping from views to model configurations based on the display field
            view_to_model_map = {}

            # If a specific combination was not requested or not found, choose the first available
            if (self.current_combination not in config) and config:
                self.current_combination = next(iter(config))

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
        """Cast the run: which views are leads (2x2 grid), which are supporting.

        Decided once, here, when the run starts; switching the display mode never
        recasts. Leads are the whitelisted models that are running, topped up to four
        with the largest of the others. Every remaining view is supporting.

        This is purely a rendering decision. Every scheduled view still gets a worker
        process, still runs inference and still feeds the throughput / deadline-miss /
        score figures, whichever group it lands in.
        """
        self.display_slot_of = {}
        self.support_views = []

        # Data collection runs with rendering off: drawing costs CPU in the same
        # process that feeds the workers, and only some models in a set pay it -- so
        # leaving it on puts a per-model rendering cost into the training labels that
        # has nothing to do with placement. The demo keeps it on.
        if not self.rendering_enabled:
            print("[UnifiedViewer] DISABLE_VISUALIZATION=1 -- rendering off; "
                  "all views run headless (execution and measurement unchanged).")
            return

        def _model(v):
            return (self.model_settings.get(v, {}) or {}).get("model", "")

        whitelisted, fill = select_leads([(v, _model(v)) for v in self.view_names])
        for view in whitelisted + fill:
            self.display_slot_of[view] = DISPLAY_SLOTS[len(self.display_slot_of)]
        self.support_views = [v for v in self.view_names if v not in self.display_slot_of]

        print(f"[UnifiedViewer] Leads {len(self.display_slot_of)}/{len(self.view_names)}: "
              + (", ".join(f"{_model(v)} -> {slot}" for v, slot in self.display_slot_of.items())
                 or "none"))
        if fill:
            print("[UnifiedViewer] Lead cells filled by model size (largest first, "
                  "model_registry.PARAMS_M, millions of parameters): "
                  + ", ".join(f"{_model(v)}={reg.params_m(_model(v)):g}M" for v in fill)
                  + (". Passed over: " + ", ".join(
                      f"{_model(v)}={reg.params_m(_model(v)):g}M" for v in self.support_views)
                     if self.support_views else "."))
        if self.support_views:
            print(f"[UnifiedViewer] Supporting views (executed and measured; drawn only in "
                  f"exhibition mode): {', '.join(_model(v) for v in self.support_views)}")

    def initialize_ui_components(self):
        """Initialize UI components."""
        self.view1 = self.findChild(QLabel, "view1")
        self.view2 = self.findChild(QLabel, "view2")
        self.view3 = self.findChild(QLabel, "view3")
        self.view4 = self.findChild(QLabel, "view4")
        for slot in DISPLAY_SLOTS:
            getattr(self, slot).setStyleSheet("background-color: #181818; border: 1px solid #444444;")

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
        self.model_signals.update_support_display.connect(self.update_support_display)

        self._build_support_tiles()

        # Right column: the mode toggle, the exhibition badge, and the two pages.
        self.schedule_name_label.setText(str(self.current_combination))
        self.show_all_toggle.setChecked(self.show_all)
        self.show_all_toggle.setEnabled(self.rendering_enabled)
        self.show_all_toggle.toggled.connect(self._on_toggle_clicked)
        self.rightStack.setCurrentIndex(1 if self.show_all else 0)
        self.exhibition_badge.setVisible(self.show_all)
        for page in (self.measurePage, self.exhibitionPage):
            page.installEventFilter(self)
        self._refresh_panels()

    def _layout_visible_views(self):
        """Always show the .ui's fixed 2x2 grid: fill the lead tiles, cross out the rest.

        The display grid is FIXED at the four cells schedule_executor_display.ui defines
        (view1..view4). The first M cells hold the leads (M = min(4, running models))
        and receive live frames; the remaining 4-M cells get the X placeholder, so an
        unused cell reads as deliberately empty and never as a black blank. The grid is
        NEVER reflowed to M, and its cells are size-policy Ignored with equal stretch,
        so neither their content nor the display mode can resize them.
        """
        from utils import create_x_image, convert_cv_to_qt
        grid = self.findChild(QGridLayout, "videoLayout")
        if grid is None:
            print("[UnifiedViewer] videoLayout not found; leaving the .ui layout as-is.")
            return
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)

        # display_slot_of maps lead views -> DISPLAY_SLOTS[0..M-1], so the used
        # display widgets are exactly the first M of DISPLAY_SLOTS.
        used = set(DISPLAY_SLOTS[i]
                   for i in range(len(getattr(self, 'display_slot_of', {}) or {})))

        x_pixmap = None
        filled = 0
        for slot in DISPLAY_SLOTS:
            widget = getattr(self, slot, None)
            if widget is None:
                continue
            if slot in used:
                filled += 1  # live model tile; frames arrive via display_slot_of
                continue
            # Inactive cell: explicit X placeholder -- never a black blank.
            try:
                if x_pixmap is None:
                    x_pixmap = convert_cv_to_qt(create_x_image(
                        dr.VIEW_W, dr.TILE_H, label="No display"))
                if x_pixmap is not None and not x_pixmap.isNull():
                    widget.setPixmap(x_pixmap)
                    widget.setScaledContents(True)
            except Exception as e:
                print(f"[Layout] X placeholder for {slot}: {e}")

        if filled == 0 and self.rendering_enabled:
            running = ", ".join(sorted(
                (self.model_settings.get(v, {}) or {}).get('model', '?') for v in self.view_names))
            print(f"[UnifiedViewer] Nothing to show -- all four slots crossed out. "
                  f"Running: {running or 'nothing'}.")
        print(f"[UnifiedViewer] Display grid fixed 2x2: {filled} model tile(s) + "
              f"{len(DISPLAY_SLOTS) - filled} X slot(s).")

    # ---- supporting tiles and the right column ---------------------------------

    def _build_support_tiles(self):
        """Create one tile per supporting view, up front, for the whole run.

        They exist (hidden, not rendering) in measure mode, so switching to exhibition
        is a show + "start drawing", never a construction: no delay, no flicker.
        """
        self.support_tiles = {}
        self.supportGrid = QGridLayout(self.supportContainer)
        self.supportGrid.setContentsMargins(0, 0, 0, 0)
        self.supportGrid.setSpacing(6)
        self.supportGrid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._support_cols = None
        self.support_empty_label = QLabel("No supporting views — every running model is "
                                          "one of the four leads.", self.supportContainer)
        self.support_empty_label.setWordWrap(True)
        self.support_empty_label.setStyleSheet("color: #666666; font-size: 13px;")
        self.support_empty_label.setVisible(not self.support_views)
        if not self.support_views:
            self.supportGrid.addWidget(self.support_empty_label, 0, 0)
        from utils import convert_cv_to_qt
        for view in self.support_views:
            cfg = self.model_settings.get(view, {}) or {}
            tile = SupportTile(view, cfg.get("model", "?"), cfg.get("execution", "?"),
                               self.supportContainer)
            try:
                # Something to look at before the first frame: never a black tile.
                tile.image.setPixmap(convert_cv_to_qt(dr.support_thumb(dr.placeholder(
                    cfg.get("model", "-"), cfg.get("execution", "-"), "Starting..."))))
            except Exception as e:
                print(f"[Layout] placeholder for {view}: {e}")
            self.support_tiles[view] = tile

    def _reflow_support(self):
        """Lay the supporting tiles out in the right column: one column if they all
        fit at that size, else two; if two still overflow, the tile area scrolls."""
        tiles = [self.support_tiles[v] for v in getattr(self, 'support_views', [])
                 if v in getattr(self, 'support_tiles', {})]
        if not tiles:
            return
        sp = self.supportGrid.spacing()
        width = max(80, self.rightPanel.width() - 6)
        # Height left for tiles: the column minus what sits above the tile area.
        above = (self.show_all_toggle.sizeHint().height() + self.exhibition_badge.sizeHint().height()
                 + 2 * self.compact_summary_label.sizeHint().height() + 48)
        avail = max(120, self.rightPanel.height() - above)
        choice = None
        for cols in (1, 2):
            tw = (width - sp * (cols - 1)) // cols
            th = SupportTile.height_for_width(tw)
            rows = -(-len(tiles) // cols)
            if rows * th + sp * (rows - 1) <= avail:
                choice = (cols, tw, th)
                break
        if choice is None:                      # overflow: two columns, scrolling
            sb = self.supportScroll.verticalScrollBar().sizeHint().width()
            tw = (width - sb - sp) // 2
            choice = (2, tw, SupportTile.height_for_width(tw))
        cols, tw, th = choice
        for tile in tiles:
            tile.setFixedSize(tw, th)
        if cols != self._support_cols:
            for tile in tiles:
                self.supportGrid.removeWidget(tile)
            for i, tile in enumerate(tiles):
                self.supportGrid.addWidget(tile, i // cols, i % cols)
            self._support_cols = cols

    def _apply_right_width(self):
        """Pin the right column to RIGHT_COL_RATIO of the window. The left 2x2 gets
        what remains, so its size is a function of the window size and nothing else."""
        try:
            cw = self.centralWidget().width()
            right = max(1, int(round(cw * RIGHT_COL_RATIO)))
            # Give the grid an even width, so its two columns come out pixel-identical.
            m = self.mainLayout.contentsMargins()
            grid_w = (cw - m.left() - m.right() - self.mainLayout.spacing() - right
                      - self.videoLayout.spacing())
            if grid_w % 2:
                right += 1
            self.rightPanel.setFixedWidth(right)
            self._reflow_support()
        except Exception as e:
            print(f"[Layout] right column sizing failed: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_right_width()
        self._geometry_changed()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._geometry_changed()

    def _geometry_changed(self):
        """Tell the owner where the window is (debounced), so a relaunched display can
        come back in the same place at the same size."""
        if not hasattr(self, '_geom_timer'):
            self._geom_timer = QTimer(self)
            self._geom_timer.setSingleShot(True)
            self._geom_timer.timeout.connect(self._report_geometry)
        self._geom_timer.start(300)

    def _report_geometry(self):
        if self.isVisible():
            self.notify("geometry", x=self.x(), y=self.y(), w=self.width(), h=self.height())

    # ---- display mode: measure <-> exhibition, switchable mid-run -----------------

    def _on_toggle_clicked(self, checked):
        self.set_show_all(checked, source="display")

    def set_show_all(self, value, source="main", t_sent=None):
        """Switch the display mode. Safe to call at any point of a run.

        Touches only what is drawn: the supporting views' rendering switch and the
        right column's page. The leads -- widgets, handlers, threads, worker processes
        -- are not touched, nothing is loaded or unloaded, and no widget is created or
        destroyed. If anything here fails the previous mode is restored and the run
        carries on.
        """
        value = bool(value) and self.rendering_enabled
        prev = self.show_all
        now = time.time()
        try:
            if value != prev:
                self._account_exhibition(now, value)
                self.show_all = value
                # Supporting views: start (or stop) drawing. Inference never stopped.
                for view in self.support_views:
                    handler = getattr(self, f"{view}_handler", None)
                    if handler is not None:
                        handler.render_enabled = value
                self.rightStack.setCurrentIndex(1 if value else 0)
                self.exhibition_badge.setVisible(value)
                self._await_paint = {"value": value, "source": source, "t_sent": t_sent,
                                     "t_recv": now, "t_applied": time.time()}
                self._await_support = None
                if value:
                    self._reflow_support()
                    self._await_support = {"t0": now, "pending": set(
                        v for v in self.support_views
                        if getattr(self, f"{v}_handler", None) is not None), "first": None}
                    for view in self.support_views:
                        handler = getattr(self, f"{view}_handler", None)
                        if handler is not None:
                            handler.render_now()     # do not wait for its next result
                self._refresh_panels()
                print(f"[Display] Mode -> {'exhibition' if value else 'measure'} "
                      f"(from {source}); leads untouched, "
                      f"{len(self.support_views)} supporting view(s) "
                      f"{'rendering' if value else 'hidden'}.", flush=True)
        except Exception as e:
            print(f"[Display] Mode switch failed; staying in "
                  f"{'exhibition' if prev else 'measure'} mode: {type(e).__name__}: {e}", flush=True)
            try:
                if self.show_all != prev:
                    self._account_exhibition(time.time(), prev)
                self.show_all = prev
                for view in self.support_views:
                    handler = getattr(self, f"{view}_handler", None)
                    if handler is not None:
                        handler.render_enabled = prev
                self.rightStack.setCurrentIndex(1 if prev else 0)
                self.exhibition_badge.setVisible(prev)
            except Exception as e2:
                print(f"[Display] Mode rollback incomplete: {e2}", flush=True)
            self._await_paint = self._await_support = None
        finally:
            # The toggle mirrors the mode whichever side asked, without re-triggering.
            try:
                if self.show_all_toggle.isChecked() != self.show_all:
                    self.show_all_toggle.blockSignals(True)
                    self.show_all_toggle.setChecked(self.show_all)
                    self.show_all_toggle.blockSignals(False)
            except Exception:
                pass
            # Always report the mode actually in force, so the main window's checkbox
            # ends up agreeing with the display even when a switch was refused.
            self.notify("show_all", value=self.show_all, source=source,
                        t_sent=t_sent, t_recv=now, t_applied=time.time())

    def _account_exhibition(self, now, entering):
        """Keep the running total of exhibition time (see _display_mode_record)."""
        if entering:
            self._exh_since = now
        elif self._exh_since is not None:
            self._exh_accum += max(0.0, now - self._exh_since)
            self._exh_since = None

    def _display_mode_record(self):
        """How much of the saved window was spent in exhibition mode.

        Exhibition mode draws the supporting views too, which costs CPU in this
        process, so figures gathered with it on are not comparable to the profiled
        ones. A run can switch modes part-way; this ratio is the marker that says how
        much of the window the saved figures were contaminated for.
        """
        t0 = getattr(self, '_measure_t0', None) or self._run_t0
        end = getattr(self, '_measure_t_end', None) or time.time()
        window = max(0.0, end - t0) if t0 else 0.0
        seconds = self._exh_accum
        if self._exh_since is not None:
            seconds += max(0.0, end - self._exh_since)
        seconds = min(seconds, window)
        return {
            "mode_at_end": "exhibition" if self.show_all else "measure",
            "exhibition_seconds": round(seconds, 2),
            "window_seconds": round(window, 2),
            "exhibition_ratio": round(seconds / window, 4) if window > 0 else 0.0,
            "note": ("exhibition mode renders the supporting views as well; figures from a "
                     "window with exhibition_ratio > 0 are for reference, not measurements"),
        }

    def eventFilter(self, obj, event):
        """Time a mode switch to the first repaint of the page it switched to."""
        try:
            pending = self._await_paint
            if (pending is not None and event.type() == QEvent.Paint
                    and obj is self.rightStack.currentWidget()):
                self._await_paint = None
                self.notify("mode_painted", t_painted=time.time(), **pending)
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def update_support_display(self, view_name, pixmap):
        """A supporting view's frame: show it, unless its tile is hidden or frozen."""
        tile = self.support_tiles.get(view_name)
        if tile is None or getattr(self, '_frozen', False) or not self.show_all:
            return
        tile.image.setPixmap(pixmap)
        pending = self._await_support
        if pending is not None and view_name in pending["pending"]:
            now = time.time()
            pending["pending"].discard(view_name)
            pending["first"] = pending["first"] or now
            if not pending["pending"]:
                self._await_support = None
                self.notify("support_rendering", t_first=pending["first"], t_all=now,
                            t_recv=pending["t0"], n=len(self.support_views))

    def _refresh_panels(self):
        """Write the last snapshot into the strips, the overall panel, the compact line
        and (in exhibition mode) the supporting captions. Display only: every number
        comes out of `self._live`, computed by update_cpu_npu_usage."""
        live = self._live or {}
        per_view = live.get("views", {})

        for view in getattr(self, 'display_slot_of', {}) or {}:
            handler = getattr(self, f"{view}_handler", None)
            st = per_view.get(view)
            if handler is None or st is None:
                continue
            if not st["count"] and not st["offered"]:
                handler.strip_segments = [("lat --", None), ("miss --", None), ("wait --", None)]
            else:
                handler.strip_segments = [
                    (f"lat {_fmt_ms(st['infer_ms'])}", None),
                    (f"miss {st['miss_rate'] * 100:.1f}%", dr.miss_color(st['miss_rate'])),
                    (f"wait {_fmt_ms(st['wait_ms'])}", None),
                ]

        self.schedule_name_label.setText(str(self.current_combination))
        if not live:
            self.score_value_label.setText("--")
            self.aggregate_label.setText("waiting for the first results…")
            self.compact_summary_label.setText("Score -- / -- FPS")
            return
        miss = live["miss_rate"]
        miss_css = ("#c62828" if miss >= dr.MISS_ALERT else
                    "#ef6c00" if miss >= dr.MISS_WARN else "inherit")
        score = f"{live['score']:.2f}".replace("-", "−")
        self.score_value_label.setText(score)
        self.score_value_label.setStyleSheet(
            "font-size: 54px; font-weight: bold; color: %s;"
            % ("#c62828" if live["score"] < 0 else "#1a7f37"))
        rows = [
            ("Throughput", f"{live['total_fps']:.1f} FPS", "inherit"),
            ("Avg", f"{live['avg_fps']:.1f} FPS", "inherit"),
            ("Miss", f"{miss * 100:.1f}%", miss_css),
            ("q95", f"{live['q95_ms']:.0f} ms", "inherit"),
            ("CPU", f"{live['cpu_percent']:.0f}%", "inherit"),
        ]
        self.aggregate_label.setText(
            "<table cellspacing='0' cellpadding='3'>" + "".join(
                f"<tr><td style='color:#666666;'>{k}</td>"
                f"<td style='font-weight:bold; color:{c}; padding-left:14px;'>{v}</td></tr>"
                for k, v, c in rows) + "</table>")
        self.compact_summary_label.setText(
            f"Score {live['score']:.1f} / {live['total_fps']:.1f} FPS".replace("-", "−"))

        if self.show_all:
            for view, tile in self.support_tiles.items():
                handler = getattr(self, f"{view}_handler", None)
                if handler is not None:
                    tile.set_metric(handler.metric_text())

    def _emit_probe(self):
        """One line of machine-readable display state (DISPLAY_PROBE=1 only)."""
        try:
            def geom(w):
                g = w.geometry()
                p = w.mapTo(self, w.rect().topLeft())
                return [p.x(), p.y(), g.width(), g.height()]

            def view_info(view):
                h = getattr(self, f"{view}_handler", None)
                proc = getattr(self, f"{view}_process", None)
                cfg = self.model_settings.get(view, {}) or {}
                th = getattr(h, 'thread', None)
                return {
                    "view": view, "model": cfg.get("model"), "device": str(cfg.get("execution")),
                    "handler_id": id(h) if h is not None else None,
                    "thread": th.ident if th is not None else None,
                    "thread_alive": bool(th.is_alive()) if th is not None else False,
                    "pid": getattr(proc, 'pid', None),
                    "alive": bool(proc.is_alive()) if proc is not None else False,
                    "count": int(getattr(h, 'infer_count', 0) or 0),
                    "msgs": int(getattr(h, 'msg_count', 0) or 0),
                    "frames": int(getattr(h, 'frames_emitted', 0) or 0),
                    "render_enabled": bool(getattr(h, 'render_enabled', False)),
                    "error": getattr(h, 'last_error', None),
                }

            leads = []
            for view, slot in (getattr(self, 'display_slot_of', {}) or {}).items():
                info = view_info(view)
                widget = getattr(self, slot)
                h = getattr(self, f"{view}_handler", None)
                pm = widget.pixmap()
                info.update({
                    "slot": slot, "widget_id": id(widget), "geom": geom(widget),
                    "strip": " | ".join(t for t, _ in (getattr(h, 'strip_segments', None) or [])),
                    "pixmap_key": int(pm.cacheKey()) if pm is not None else None,
                })
                leads.append(info)
            support = []
            for view in getattr(self, 'support_views', []):
                info = view_info(view)
                tile = self.support_tiles.get(view)
                pm = tile.image.pixmap() if tile is not None else None
                info.update({
                    "tile_id": id(tile), "visible": bool(tile is not None and tile.isVisible()),
                    "geom": geom(tile) if tile is not None else None,
                    "caption": tile.caption.text() if tile is not None else None,
                    "pixmap_key": int(pm.cacheKey()) if pm is not None else None,
                })
                support.append(info)
            status = {
                "t": time.time(), "pid": os.getpid(), "combination": self.current_combination,
                "mode": "exhibition" if self.show_all else "measure",
                "measuring": bool(getattr(self, '_measure_t0', None)),
                "stopped": bool(getattr(self, '_already_stopped', False)),
                "window": [self.x(), self.y(), self.width(), self.height()],
                "left": geom(self.leftPanel), "right_w": self.rightPanel.width(),
                "central_w": self.centralWidget().width(),
                "x_slots": len(DISPLAY_SLOTS) - len(leads),
                "leads": leads, "support": support,
                "panel": {
                    "page": self.rightStack.currentIndex(),
                    "toggle": self.show_all_toggle.isChecked(),
                    "badge": self.exhibition_badge.isVisible(),
                    "schedule": self.schedule_name_label.text(),
                    "score": self.score_value_label.text(),
                    "compact": self.compact_summary_label.text(),
                    "live": {k: v for k, v in (self._live or {}).items() if k != "views"},
                },
                "video_reader_alive": bool(getattr(self, 'video_reader_proc', None) is not None
                                           and self.video_reader_proc.is_alive()),
            }
            print("@@STATUS " + json.dumps(status), flush=True)
        except Exception as e:
            print(f"[Probe] {type(e).__name__}: {e}", flush=True)

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
                support=view_name in self.support_views,
            )
            if handler.is_support:
                handler.render_enabled = self.show_all   # drawn in exhibition mode only
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
        # Stopped means stopped: the owner must not start the next cycle behind a
        # frozen screen the presenter is still talking over.
        alive = [v for v in self.view_names
                 if getattr(getattr(self, f"{v}_process", None), 'is_alive', lambda: False)()]
        self.notify("stopped", workers_alive=len(alive))

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
        print(f"Starting execution with duration: {duration} seconds "
              f"(warmup {WARMUP_SECONDS}s excluded)")
        # Warmup is excluded from the measurement window.
        try:
            self.window_duration_sec = max(0.0, float(duration) - WARMUP_SECONDS)
        except Exception:
            self.window_duration_sec = None
        
        # Reset idempotent stop flag for new run
        self._already_stopped = False

        # Exhibition-time bookkeeping starts with the run (restarted at warmup end).
        self._run_t0 = time.time()
        self._exh_accum = 0.0
        self._exh_since = self._run_t0 if self.show_all else None

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
        
        # Reset all metrics once the warmup has elapsed; measurement starts there.
        try:
            QTimer.singleShot(int(WARMUP_SECONDS * 1000), self._begin_measurement_window)
        except Exception:
            pass
            
        # Schedule stopping execution after the specified duration (includes warmup)
        QTimer.singleShot(duration * 1000, self.timed_shutdown)
    
    def timed_shutdown(self):
        """Stop execution after the scheduled duration without closing the application."""
        print("Execution duration completed, stopping model execution...")
        self.stop_execution()
        
    def measured_window_sec(self):
        """Wall-clock length of the measurement window so far (0 before warmup ends)."""
        t0 = getattr(self, '_measure_t0', None)
        if not t0:
            return 0.0
        end = getattr(self, '_measure_t_end', None) or time.time()
        return max(0.0, end - t0)

    def _begin_measurement_window(self):
        """Reset all per-view and feeder counters after warmup to start measurement."""
        import time as _t
        self._measure_t0 = _t.time()
        self._measure_t_end = None
        # The saved window starts now, so the exhibition share is counted from now.
        self._exh_accum = 0.0
        self._exh_since = self._measure_t0 if self.show_all else None
        print(f"[UnifiedViewer] Warmup over ({WARMUP_SECONDS}s); measurement window open.")
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

        # Freeze the measurement clock here: everything below (y1's completions, y2's
        # offered count) must divide by the same window.
        if getattr(self, '_measure_t0', None) and not getattr(self, '_measure_t_end', None):
            self._measure_t_end = time.time()

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

        def _close_writer(q):
            """Drop OUR copy of the write end of a queue only a worker writes to.

            A worker that had to be terminated can die half-way through writing a
            result, leaving a truncated message in the pipe. Reading it blocks until
            the rest arrives -- and while this process still holds the write end open
            the kernel never reports EOF, so `get_nowait()` below would wait forever
            (Stop All froze the window exactly here). With every writer gone the
            truncated read ends in EOFError instead, which the drain treats as empty.
            """
            try:
                writer = getattr(q, '_writer', None)
                if writer is not None:
                    writer.close()
            except Exception as e:
                print(f"[Cleanup] writer close failed (continuing): {e}")
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
                # Written by a worker process, read here: the video reader's frames
                # and every view's results. (The per-view frame queues go the other way.)
                if q and (name == 'video_frame_queue' or name.endswith('_result_queue')):
                    _close_writer(q)
                _drain(q)
                _close(q)
            finally:
                try:
                    setattr(self, name, None)
                except Exception:
                    pass
        
    def update_cpu_npu_usage(self):
        """The once-a-second tick: snapshot every figure, then refresh the display.

        All figures are the ones the per-view handlers and the feeder already keep
        (avg_fps, avg_infer_time, infer_count, late_count, avg_wait_ms, drop/offered
        counts); nothing is measured here. The snapshot covers EVERY scheduled view,
        leads and supporting alike, in either display mode.
        """
        current = get_cpu_metrics(interval=0)

        views = list(self.view_names)
        feeder = getattr(self, 'video_feeder', None)
        fdrop = getattr(feeder, 'drop_counts', {}) or {}
        foffered = getattr(feeder, 'offered_counts', {}) or {}
        stats = {}
        miss = offered = 0
        for v in views:
            cfg = self.model_settings.get(v, {}) or {}
            h = getattr(self, f"{v}_handler", None)
            count = int(getattr(h, 'infer_count', 0) or 0)
            # Deadline misses: late completions + queue-full drops, over what was offered.
            late = int(getattr(h, 'late_count', 0) or 0)
            dropped = int(fdrop.get(v, 0) or 0)
            off = int(foffered.get(v, 0) or 0) or (count + dropped)
            stats[v] = {
                "fps": getattr(h, 'avg_fps', 0.0),
                "infer_ms": getattr(h, 'avg_infer_time', 0.0),
                "count": count,
                "model": cfg.get("model", ""),
                "mode": str(cfg.get("execution", "cpu")).upper(),
                "wait_ms": getattr(h, 'avg_wait_ms', 0.0),
                "dropped": dropped,
                "offered": off,
                "miss_rate": ((late + dropped) / off) if off > 0 else 0.0,
            }
            miss += late + dropped
            offered += off

        # Total / average FPS (total throughput)
        total_fps = sum(st["fps"] for st in stats.values())
        total_avg_fps = total_fps / len(views) if views else 0.0
        miss_rate = (miss / offered) if offered > 0 else 0.0
        score = total_fps - 100.0 * miss_rate

        # q95: max queue_wait_ms_p95 across the devices in use
        max_q_wait = 0.0
        try:
            # per_view_stats fields aligned with _get_device_metrics_default expectations
            per_view_stats = {
                v: (stats[v]["fps"], stats[v]["infer_ms"], stats[v]["count"],
                    stats[v]["model"], stats[v]["mode"], stats[v]["wait_ms"],
                    stats[v]["dropped"])
                for v in views
            }
            devices_used = set(st["mode"] for st in stats.values())
            dev_metrics = self._get_device_metrics_default(devices_used, per_view_stats, views)
            q_waits = []
            for val in (dev_metrics or {}).values():
                try:
                    q_waits.append(float(val.get('queue_wait_ms_p95', 0.0) or 0.0))
                except Exception:
                    pass
            max_q_wait = max(q_waits) if q_waits else 0.0
        except Exception:
            # Don't let a metrics hiccup take the display refresh down with it.
            pass

        self._live = {
            "total_fps": total_fps, "avg_fps": total_avg_fps, "miss_rate": miss_rate,
            "score": score, "q95_ms": max_q_wait,
            "cpu_percent": current['CPU_Usage_percent'], "views": stats,
        }
        try:
            self._refresh_panels()
        except Exception as e:
            print(f"[Display] panel refresh failed (continuing): {e}")

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
            # The ACTUAL measured window, not the nominal one: y1 counts completions
            # over this same clock, so y2's "offered" must use it too or the two
            # targets are normalised against different windows.
            wsec = float(self.measured_window_sec() or self.window_duration_sec or 1.0)
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

            # Mark how much of this window was spent in exhibition mode (extra rendering
            # load): figures from such a window are for reference, not measurements.
            try:
                throughput_data["display_mode"] = self._display_mode_record()
            except Exception as e:
                print(f"[Save Throughput] Warning: display mode record failed: {e}")

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