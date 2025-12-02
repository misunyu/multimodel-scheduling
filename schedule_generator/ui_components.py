import os
from typing import Dict, List, Tuple, Any, Optional, Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, 
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QFileDialog, QLineEdit
)
from PyQt5.QtGui import QColor, QBrush, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
import yaml
from datetime import datetime

class UIComponents:
    """
    Class for handling UI components and visualization.
    Provides methods for table initialization, result display, and chart creation.
    """
    
    def __init__(self, log_callback=None):
        """
        Initialize the UI components handler.
        
        Args:
            log_callback: Function to call for logging messages
        """
        self.log_callback = log_callback
        
    def log(self, message):
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(message)
    
    def init_table(self, table):
        """
        Initialize a table for displaying results.
        
        Args:
            table: QTableWidget to initialize
        """
        table.clear()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Model", "Load (ms)", "Inference (ms)", "Tokens/s"])
        table.setRowCount(0)
    
    def insert_result_row(self, table, model_file, load_ms, infer_ms, tokens_per_s: Optional[float] = None):
        """
        Insert a result row into a table.
        
        Args:
            table: QTableWidget to insert into
            model_file: Model file name or path
            load_ms: Load time in milliseconds
            infer_ms: Inference time in milliseconds
        """
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(model_file))
        table.setItem(row, 1, QTableWidgetItem(f"{load_ms:.1f}"))
        table.setItem(row, 2, QTableWidgetItem(f"{infer_ms:.1f}"))
        tok_text = "-" if tokens_per_s is None else f"{tokens_per_s:.2f}"
        table.setItem(row, 3, QTableWidgetItem(tok_text))
    
    def initialize_total_table(self, total_table):
        """
        Initialize the total results table.
        
        Args:
            total_table: QTableWidget for total results
        """
        total_table.clear()
        # Result tab: remove NPU2 column; rename NPU1 to GPU (+ CPU Offloading)
        # Add separate Tokens/s columns for CPU and GPU
        total_table.setColumnCount(5)
        total_table.setHorizontalHeaderLabels([
            "Model",
            "CPU Inf. (ms)",
            "GPU + CPU Offloading (ms)",
            "CPU Tokens/s",
            "GPU Tokens/s"
        ])
        total_table.setRowCount(0)

        header = total_table.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
    
    def populate_total_table(self, total_table, all_models, valid_model_onnx, 
                            npu1_load, npu1_infer, npu2_load, npu2_infer, 
                            cpu_infer_per_partition):
        """
        Populate the total results table with data.
        
        Args:
            total_table: QTableWidget for total results
            all_models: Set of all model names
            valid_model_onnx: Dictionary of valid ONNX models
            npu1_load, npu1_infer: NPU1 load and inference times
            npu2_load, npu2_infer: NPU2 load and inference times
            cpu_infer_per_partition: CPU inference times per partition
        """
        # Add rows for each model
        def is_llm(name: str) -> bool:
            n = (name or "").lower()
            return ("gpt2" in n) or ("tiny-llama" in n)

        # Common test sentence for LLM throughput (tokens/s)
        # Widely used pangram
        test_sentence = "The quick brown fox jumps over the lazy dog."
        test_tokens = max(1, len(test_sentence.strip().split()))

        for model in sorted(all_models):
            row = total_table.rowCount()
            total_table.insertRow(row)
            
            # Model name (ensure tiny-llama shows folder + filename when needed)
            disp = model
            try:
                ml = (model or "").lower()
                if ("tiny-llama" in ml) and ("/" not in model) and ("\\" not in model):
                    disp = "tiny-llama-chat-onnx/model.onnx"
            except Exception:
                pass
            total_table.setItem(row, 0, QTableWidgetItem(disp))
            
            # CPU inference time: cpu_tab의 값(모든 항목 합계)을 사용
            cpu_infer = 0.0
            # 우선 원래 key(model)로 조회하고, 실패 시 표시용 이름(disp)으로 보조 조회
            if model in cpu_infer_per_partition:
                cpu_infer = sum(cpu_infer_per_partition[model])
            elif disp in cpu_infer_per_partition:
                cpu_infer = sum(cpu_infer_per_partition[disp])
            total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_infer:.1f}"))
            
            # GPU inference time (previously NPU1; Load columns removed)
            npu1_infer_time = npu1_infer.get(model, 0.0)
            # 모델명이 표기용으로 정규화(disp)된 경우 보조 조회
            if (not isinstance(npu1_infer_time, (int, float)) or npu1_infer_time <= 0) and disp != model:
                npu1_infer_time = npu1_infer.get(disp, npu1_infer_time)
            total_table.setItem(row, 2, QTableWidgetItem(f"{npu1_infer_time:.1f}"))

            # Tokens/s (LLM only): compute separately for CPU and GPU
            cpu_tps_item = QTableWidgetItem("-")
            gpu_tps_item = QTableWidgetItem("-")
            # LLM 감지는 표시용 이름(disp)을 기준으로 수행해야 tiny-llama 정규화 케이스를 놓치지 않음
            if is_llm(disp):
                # CPU tokens/s
                if isinstance(cpu_infer, (int, float)) and cpu_infer > 0:
                    cpu_tps = test_tokens / (cpu_infer / 1000.0)
                    cpu_tps_item = QTableWidgetItem(f"{cpu_tps:.2f}")
                # GPU tokens/s
                if isinstance(npu1_infer_time, (int, float)) and npu1_infer_time > 0:
                    gpu_tps = test_tokens / (npu1_infer_time / 1000.0)
                    gpu_tps_item = QTableWidgetItem(f"{gpu_tps:.2f}")
            total_table.setItem(row, 3, cpu_tps_item)
            total_table.setItem(row, 4, gpu_tps_item)
    
    def add_total_row(self, total_table, cpu_infer_total, 
                     gpu_infer_total):
        """
        Add a total row to the total results table.
        
        Args:
            total_table: QTableWidget for total results
            cpu_infer_total: Total CPU inference time
            gpu_infer_total: Total GPU inference time
        """
        row = total_table.rowCount()
        total_table.insertRow(row)
        
        # Set bold font for total row
        bold_font = QFont()
        bold_font.setBold(True)
        
        # Total label
        total_item = QTableWidgetItem("Total")
        total_item.setFont(bold_font)
        total_table.setItem(row, 0, total_item)
        
        # CPU total
        cpu_item = QTableWidgetItem(f"{cpu_infer_total:.1f}")
        cpu_item.setFont(bold_font)
        total_table.setItem(row, 1, cpu_item)
        
        # GPU inference total
        gpu_infer_item = QTableWidgetItem(f"{gpu_infer_total:.1f}")
        gpu_infer_item.setFont(bold_font)
        total_table.setItem(row, 2, gpu_infer_item)
        # Tokens/s columns for Total: not applicable
        total_table.setItem(row, 3, QTableWidgetItem("-"))
        total_table.setItem(row, 4, QTableWidgetItem("-"))
    
    def highlight_deploy_results(self, total_table, times, models, device_settings=None):
        """
        Compute and highlight a deployment schedule that (approximately) minimizes
        the overall completion time (makespan), considering device concurrency limits.
        Limits come from device_settings if provided (fallback to 1 each):
          - CPU parallel slots: devices.cpu.count (>=1)
          - GPU device count:  devices.gpu.count or devices.npu.count (>=1)

        Scheduling model:
          - CPU side: up to K parallel slots; makespan is computed via LPT list scheduling.
          - GPU side: up to D devices in parallel; makespan is computed via LPT list scheduling.

        The function also saves the chosen assignments to static_best_schedule.json.
        
        Args:
            total_table: QTableWidget for total results
            times: Output list, will be populated with the chosen per-model device time
            models: Output list, will be populated with tuples (model_name, device_label)
        """
        if not total_table or total_table.rowCount() == 0:
            return

        # Clear any previous highlighting
        for row in range(total_table.rowCount()):
            for col in range(total_table.columnCount()):
                item = total_table.item(row, col)
                if item:
                    item.setBackground(QBrush(QColor(255, 255, 255)))

        # Define colors for different devices
        cpu_color = QColor(204, 230, 255)   # Light blue for CPU
        gpu_color = QColor(255, 255, 204)   # Light yellow for GPU

        # Collect per-model timing options from the table
        # Columns: 0=Model, 1=CPU Inf, 2=GPU(+CPU Offloading)
        models_data = []  # list of dicts: {name, cpu, gpu_total, gpu_inf}
        last_row_index = total_table.rowCount() - 1  # last row is the Total row
        for row in range(max(0, last_row_index)):
            name_item = total_table.item(row, 0)
            if not name_item:
                continue
            model_name = name_item.text()
            try:
                cpu_time = float(total_table.item(row, 1).text()) if total_table.item(row, 1) else 0.0
            except Exception:
                cpu_time = 0.0
            try:
                gpu_infer = float(total_table.item(row, 2).text()) if total_table.item(row, 2) else 0.0
            except Exception:
                gpu_infer = 0.0

            # Treat non-positive or 0 times as unavailable
            def norm(x):
                return x if (isinstance(x, (int, float)) and x > 0) else float('inf')

            models_data.append({
                "name": model_name,
                "cpu": norm(cpu_time),
                # For scheduling on GPU we consider inference time occupying the GPU (Load removed)
                "gpu_total": norm(gpu_infer),
                # Keep pure inference times for JSON reporting
                "gpu_inf": norm(gpu_infer),
            })

        if not models_data:
            return

        # Resolve concurrency limits from settings
        def _get_int(d, *path, default=1):
            try:
                cur = d
                for k in path:
                    if cur is None:
                        return default
                    cur = cur.get(k)
                if isinstance(cur, int) and cur >= 1:
                    return cur
                # If nested dict with 'count'
                if isinstance(cur, dict):
                    c = cur.get('count')
                    if isinstance(c, int) and c >= 1:
                        return c
            except Exception:
                pass
            return default

        cpu_parallel = 1
        gpu_devices = 1
        if isinstance(device_settings, dict):
            devs = device_settings.get('devices') or {}
            # support both lowercase and uppercase keys and legacy 'npu'
            # CPU
            cpu_parallel = _get_int(device_settings, 'devices', 'cpu', default=1)
            if cpu_parallel == 1:
                cpu_parallel = _get_int(device_settings, 'devices', 'CPU', default=1)
            # GPU from either 'gpu' or legacy 'npu'
            gpu_devices = _get_int(device_settings, 'devices', 'gpu', default=1)
            if gpu_devices == 1:
                gpu_devices = _get_int(device_settings, 'devices', 'GPU', default=1)
            if gpu_devices == 1:
                gpu_devices = _get_int(device_settings, 'devices', 'npu', default=1)
            if gpu_devices == 1:
                gpu_devices = _get_int(device_settings, 'devices', 'NPU', default=1)

        # Safety clamps
        cpu_parallel = max(1, int(cpu_parallel))
        gpu_devices = max(1, int(gpu_devices))

        # Simplest rule: per model, choose the faster device by comparing CPU vs GPU(+CPU) column.
        # If equal (within epsilon) or CPU is not faster, choose GPU. If one side is unavailable (inf), choose the other.
        equal_eps = 1e-6
        n = len(models_data)
        best_assignment = [None] * n  # 0=CPU, 1=GPU
        for i, m in enumerate(models_data):
            cpu_t = m["cpu"]
            gpu_t = m["gpu_total"]
            if cpu_t == float('inf') and gpu_t == float('inf'):
                # No feasible device for this model: skip highlighting for this row
                best_assignment[i] = None
            elif gpu_t == float('inf'):
                best_assignment[i] = 0
            elif cpu_t == float('inf'):
                best_assignment[i] = 1
            else:
                # Prefer CPU only if strictly faster; ties default to GPU
                best_assignment[i] = 0 if (cpu_t + equal_eps < gpu_t) else 1

        # Clear output containers and fill with best assignment
        times.clear()
        models.clear()

        # Apply highlighting according to the best assignment
        for i, d in enumerate(best_assignment):
            row = i
            # Safety check: skip if beyond table (shouldn't happen)
            if row >= last_row_index:
                continue
            # Determine color and time for outputs
            if d == 0:
                color = cpu_color
                chosen_label = "CPU"
                chosen_time = models_data[i]["cpu"] if models_data[i]["cpu"] < float('inf') else 0.0
            elif d == 1:
                color = gpu_color
                chosen_label = "GPU"
                # For times list, keep inference part (for external expectations)
                chosen_time = models_data[i]["gpu_inf"] if models_data[i]["gpu_inf"] < float('inf') else 0.0
            else:
                # No feasible device; leave row uncolored and skip outputs
                continue

            # Highlight the entire row
            for col in range(total_table.columnCount()):
                item = total_table.item(row, col)
                if item:
                    item.setBackground(QBrush(color))

            # Append results
            times.append(chosen_time)
            models.append((models_data[i]["name"], chosen_label))

            # Optional trace logging if callback provided
            try:
                cpu_t = models_data[i]["cpu"]
                gpu_t = models_data[i]["gpu_total"]
                if hasattr(self, 'log') and callable(getattr(self, 'log')):
                    self.log(f"[Schedule] {models_data[i]['name']}: CPU={cpu_t if cpu_t < float('inf') else 'NA'} ms, "
                             f"GPU={gpu_t if gpu_t < float('inf') else 'NA'} ms -> {chosen_label}")
            except Exception:
                pass

        # After highlighting, save best schedule to static_best_schedule.json
        try:
            # Map UI device labels to execution names for JSON output
            def to_execution_label(label: str) -> str:
                up = label.upper()
                if up == "CPU":
                    return "CPU"
                if up == "GPU":
                    return "GPU"
                return up

            # Build a single-entry results object
            models_dict = {}
            total_fps_sum = 0.0
            for idx, (model_name, device_label) in enumerate(models, start=1):
                view_key = f"view{idx}"

                # Determine avg_inference_time_ms (use inference time only for reporting)
                avg_time_ms = 0.0
                exec_label = to_execution_label(device_label)
                # Locate row to read time columns again
                for row in range(last_row_index):
                    item = total_table.item(row, 0)
                    if item and item.text() == model_name:
                        cpu_item = total_table.item(row, 1)
                        gpu_inf_item = total_table.item(row, 2)
                        if exec_label == "CPU":
                            ref = cpu_item
                        else:  # GPU
                            ref = gpu_inf_item
                        try:
                            avg_time_ms = float(ref.text()) if ref and ref.text() else 0.0
                        except Exception:
                            avg_time_ms = 0.0
                        break

                throughput_fps = round(1000.0 / avg_time_ms, 2) if avg_time_ms > 0 else 0.0
                total_fps_sum += throughput_fps

                models_dict[view_key] = {
                    "model": model_name,
                    "execution": exec_label,
                    "throughput_fps": throughput_fps,
                    "avg_inference_time_ms": round(avg_time_ms, 2),
                    "inference_count": 0
                }

            active_views = len(models_dict)
            result_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "combination": "combination_1",
                "models": models_dict,
                "total": {
                    "total_throughput_fps": round(total_fps_sum, 2),
                    "avg_throughput_fps": round((total_fps_sum / active_views) if active_views else 0.0, 2)
                }
            }

            final_obj = {
                "best deployment": "combination_1",
                "data": [result_entry]
            }

            with open("static_best_schedule.json", "w", encoding="utf-8") as f:
                json.dump(final_obj, f, indent=4, ensure_ascii=False)
        except Exception as e:
            try:
                self.log(f"[Error] Failed to write static_best_schedule.json: {e}")
            except Exception:
                pass
    
    def create_inference_bar_chart(self, parent_widget, models, cpu_table, npu1_table, npu2_table):
        """
        Create a bar chart comparing inference times across devices.
        
        Args:
            parent_widget: Parent widget to add the chart to
            models: List of model names
            cpu_table, npu1_table, npu2_table: Tables with profiling data
            
        Returns:
            FigureCanvas with the created chart
        """
        # Create figure and canvas
        figure = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(figure)
        
        # Add subplot
        ax = figure.add_subplot(111)
        
        # Prepare data for plotting (모델 분할 미사용: 파일 이름 기준으로 그룹핑)
        model_names = []
        cpu_times = []
        gpu_times = []
        
        for model in models:
            model_names.append(model)
            
            # Get CPU inference time
            cpu_time = 0.0
            for row in range(cpu_table.rowCount()):
                path_item = cpu_table.item(row, 0)
                if path_item and model in os.path.splitext(os.path.basename(path_item.text()))[0]:
                    infer_item = cpu_table.item(row, 2)
                    if infer_item:
                        cpu_time += float(infer_item.text())
            cpu_times.append(cpu_time)
            
            # Get GPU(NPU1) inference time
            npu1_time = 0.0
            for row in range(npu1_table.rowCount()):
                path_item = npu1_table.item(row, 0)
                if path_item and model in os.path.splitext(os.path.basename(path_item.text()))[0]:
                    infer_item = npu1_table.item(row, 2)
                    if infer_item:
                        npu1_time += float(infer_item.text())
            gpu_times.append(npu1_time)
        
        # Set up bar positions
        x = range(len(model_names))
        width = 0.35
        
        # Create bars
        ax.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='skyblue')
        ax.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='gold')
        
        # Add labels and legend
        ax.set_xlabel('Models')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Comparison (CPU vs GPU)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # Adjust layout
        figure.tight_layout()
        
        return canvas
    
    def show_partition_assignment_dialog(self, parent, assignments):
        """
        Show a dialog displaying model-to-device assignments.
        
        Args:
            parent: Parent widget
            assignments: List of (model_name, device) tuples
            
        Returns:
            QDialog instance
        """
        dialog = QDialog(parent)
        dialog.setWindowTitle("Model Assignments")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # Create table for assignments
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Model", "Device"])
        
        # Add assignments to table
        for i, (model, device) in enumerate(assignments):
            table.insertRow(i)
            table.setItem(i, 0, QTableWidgetItem(model))
            table.setItem(i, 1, QTableWidgetItem(device))
        
        # Set table properties
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        # Add table to layout
        layout.addWidget(table)
        
        # Add close button
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        return dialog
    
    def show_settings_dialog(self, parent, device_settings_file, load_callback):
        """
        Show a dialog for configuring device settings.
        
        Args:
            parent: Parent widget
            device_settings_file: Path to device settings file
            load_callback: Callback function to load settings
            
        Returns:
            QDialog instance
        """
        dialog = QDialog(parent)
        dialog.setWindowTitle("Device Settings")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Device Settings File:"))
        file_input = QLineEdit(device_settings_file)
        file_layout.addWidget(file_input)
        
        browse_button = QPushButton("Browse")
        file_layout.addWidget(browse_button)
        
        layout.addLayout(file_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Set dialog layout
        dialog.setLayout(layout)
        
        # Define browse function
        def browse_file():
            file_path, _ = QFileDialog.getOpenFileName(
                dialog, "Select Device Settings File", "", "YAML Files (*.yaml);;All Files (*)"
            )
            if file_path:
                file_input.setText(file_path)
        
        # Define accept function
        def accept():
            nonlocal device_settings_file
            device_settings_file = file_input.text()
            load_callback()
            dialog.accept()
        
        # Connect signals
        browse_button.clicked.connect(browse_file)
        ok_button.clicked.connect(accept)
        cancel_button.clicked.connect(dialog.reject)
        
        return dialog, device_settings_file