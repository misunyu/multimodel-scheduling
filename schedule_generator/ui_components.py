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
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model", "Load (ms)", "Inference (ms)"])
        table.setRowCount(0)
    
    def insert_result_row(self, table, model_file, load_ms, infer_ms):
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
    
    def initialize_total_table(self, total_table):
        """
        Initialize the total results table.
        
        Args:
            total_table: QTableWidget for total results
        """
        total_table.clear()
        total_table.setColumnCount(6)
        total_table.setHorizontalHeaderLabels([
            "Model",
            "CPU Inf. (ms)",
            "NPU1 Load (ms)",
            "NPU1 + CPU Inf. (ms)",
            "NPU2 Load (ms)",
            "NPU2 + CPU Inf. (ms)"
        ])
        total_table.setRowCount(0)

        header = total_table.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(6):
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
        for model in sorted(all_models):
            row = total_table.rowCount()
            total_table.insertRow(row)
            
            # Model name
            total_table.setItem(row, 0, QTableWidgetItem(model))
            
            # CPU inference time
            cpu_infer = 0.0
            if model in valid_model_onnx:
                cpu_infer = valid_model_onnx[model][1]
            elif model in cpu_infer_per_partition:
                cpu_infer = sum(cpu_infer_per_partition[model])
            total_table.setItem(row, 1, QTableWidgetItem(f"{cpu_infer:.1f}"))
            
            # NPU1 load and inference times
            npu1_load_time = npu1_load.get(model, 0.0)
            npu1_infer_time = npu1_infer.get(model, 0.0)
            total_table.setItem(row, 2, QTableWidgetItem(f"{npu1_load_time:.1f}"))
            total_table.setItem(row, 3, QTableWidgetItem(f"{npu1_infer_time:.1f}"))
            
            # NPU2 load and inference times
            npu2_load_time = npu2_load.get(model, 0.0)
            npu2_infer_time = npu2_infer.get(model, 0.0)
            total_table.setItem(row, 4, QTableWidgetItem(f"{npu2_load_time:.1f}"))
            total_table.setItem(row, 5, QTableWidgetItem(f"{npu2_infer_time:.1f}"))
    
    def add_total_row(self, total_table, cpu_infer_total, npu1_load_total, 
                     npu1_infer_total, npu2_load_total, npu2_infer_total):
        """
        Add a total row to the total results table.
        
        Args:
            total_table: QTableWidget for total results
            cpu_infer_total: Total CPU inference time
            npu1_load_total, npu1_infer_total: Total NPU1 load and inference times
            npu2_load_total, npu2_infer_total: Total NPU2 load and inference times
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
        
        # NPU1 load total
        npu1_load_item = QTableWidgetItem(f"{npu1_load_total:.1f}")
        npu1_load_item.setFont(bold_font)
        total_table.setItem(row, 2, npu1_load_item)
        
        # NPU1 inference total
        npu1_infer_item = QTableWidgetItem(f"{npu1_infer_total:.1f}")
        npu1_infer_item.setFont(bold_font)
        total_table.setItem(row, 3, npu1_infer_item)
        
        # NPU2 load total
        npu2_load_item = QTableWidgetItem(f"{npu2_load_total:.1f}")
        npu2_load_item.setFont(bold_font)
        total_table.setItem(row, 4, npu2_load_item)
        
        # NPU2 inference total
        npu2_infer_item = QTableWidgetItem(f"{npu2_infer_total:.1f}")
        npu2_infer_item.setFont(bold_font)
        total_table.setItem(row, 5, npu2_infer_item)
    
    def highlight_deploy_results(self, total_table, times, models):
        """
        Compute and highlight a globally optimal deployment schedule that minimizes
        the overall completion time (makespan), considering that:
        - CPU can execute multiple models/partitions concurrently (parallel), so
          its total time is the maximum of assigned CPU model times.
        - Each NPU device is exclusive: it can run only one model/partition at a time,
          so multiple assignments to the same NPU serialize (sum of their times).
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
        npu1_color = QColor(255, 255, 204)  # Light yellow for NPU1 (UI label)
        npu2_color = QColor(255, 214, 153)  # Light orange for NPU2 (UI label)

        # Collect per-model timing options from the table
        # Columns: 0=Model, 1=CPU Inf, 2=NPU1 Load, 3=NPU1+CPU Inf, 4=NPU2 Load, 5=NPU2+CPU Inf
        models_data = []  # list of dicts: {name, cpu, npu1_total, npu2_total, npu1_inf, npu2_inf}
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
                npu1_load = float(total_table.item(row, 2).text()) if total_table.item(row, 2) else 0.0
            except Exception:
                npu1_load = 0.0
            try:
                npu1_infer = float(total_table.item(row, 3).text()) if total_table.item(row, 3) else 0.0
            except Exception:
                npu1_infer = 0.0
            try:
                npu2_load = float(total_table.item(row, 4).text()) if total_table.item(row, 4) else 0.0
            except Exception:
                npu2_load = 0.0
            try:
                npu2_infer = float(total_table.item(row, 5).text()) if total_table.item(row, 5) else 0.0
            except Exception:
                npu2_infer = 0.0

            # Treat non-positive or 0 times as unavailable
            def norm(x):
                return x if (isinstance(x, (int, float)) and x > 0) else float('inf')

            models_data.append({
                "name": model_name,
                "cpu": norm(cpu_time),
                # For scheduling on NPU we consider load + inference time occupying the NPU
                "npu1_total": norm(npu1_load) + norm(npu1_infer) if norm(npu1_load) < float('inf') and norm(npu1_infer) < float('inf') else float('inf'),
                "npu2_total": norm(npu2_load) + norm(npu2_infer) if norm(npu2_load) < float('inf') and norm(npu2_infer) < float('inf') else float('inf'),
                # Keep pure inference times for JSON reporting (historical format)
                "npu1_inf": norm(npu1_infer),
                "npu2_inf": norm(npu2_infer),
            })

        if not models_data:
            return

        # Brute-force search over assignments to minimize makespan
        # Device indices: 0=CPU, 1=NPU1, 2=NPU2
        n = len(models_data)
        best_assignment = None
        best_makespan = float('inf')

        # Early exit: if n is large, we could add heuristics, but typical n is small
        from itertools import product
        for choices in product((0, 1, 2), repeat=n):
            cpu_bucket = 0.0
            npu1_bucket = 0.0
            npu2_bucket = 0.0
            feasible = True
            for i, d in enumerate(choices):
                m = models_data[i]
                if d == 0:
                    t = m["cpu"]
                    if t == float('inf'):
                        feasible = False
                        break
                    # CPU runs in parallel: bucket is max
                    cpu_bucket = max(cpu_bucket, t)
                elif d == 1:
                    t = m["npu1_total"]
                    if t == float('inf'):
                        feasible = False
                        break
                    npu1_bucket += t
                else:  # d == 2
                    t = m["npu2_total"]
                    if t == float('inf'):
                        feasible = False
                        break
                    npu2_bucket += t
            if not feasible:
                continue
            makespan = max(cpu_bucket, npu1_bucket, npu2_bucket)
            if makespan < best_makespan:
                best_makespan = makespan
                best_assignment = choices

        if best_assignment is None:
            # Fallback: highlight nothing if no feasible assignment
            return

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
                color = npu1_color
                chosen_label = "NPU1"  # UI label; JSON will map to NPU0
                # For times list, keep inference part (for external expectations)
                chosen_time = models_data[i]["npu1_inf"] if models_data[i]["npu1_inf"] < float('inf') else 0.0
            else:
                color = npu2_color
                chosen_label = "NPU2"  # UI label; JSON will map to NPU1
                chosen_time = models_data[i]["npu2_inf"] if models_data[i]["npu2_inf"] < float('inf') else 0.0

            # Highlight the entire row
            for col in range(total_table.columnCount()):
                item = total_table.item(row, col)
                if item:
                    item.setBackground(QBrush(color))

            # Append results
            times.append(chosen_time)
            models.append((models_data[i]["name"], chosen_label))

        # After highlighting, save best schedule to static_best_schedule.json
        try:
            # Map UI device labels to execution names for JSON output
            # UI labels: CPU, NPU1, NPU2 -> JSON execution: CPU, NPU0, NPU1
            def to_execution_label(label: str) -> str:
                up = label.upper()
                if up == "CPU":
                    return "CPU"
                if up == "NPU1":
                    return "NPU0"  # Map UI NPU1 to NPU0
                if up == "NPU2":
                    return "NPU1"  # Map UI NPU2 to NPU1
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
                        npu0_inf_item = total_table.item(row, 3)
                        npu1_inf_item = total_table.item(row, 5)
                        if exec_label == "CPU":
                            ref = cpu_item
                        elif exec_label == "NPU0":
                            ref = npu0_inf_item
                        else:
                            ref = npu1_inf_item
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
        
        # Helper function to get inference time for a partition
        def get_inference_time(partition_name, primary_device):
            if primary_device == "CPU":
                table = cpu_table
            elif primary_device == "NPU1":
                table = npu1_table
            elif primary_device == "NPU2":
                table = npu2_table
            else:
                return 0.0
                
            for row in range(table.rowCount()):
                path_item = table.item(row, 0)
                if not path_item:
                    continue
                    
                path = path_item.text()
                if partition_name in path:
                    infer_item = table.item(row, 2)
                    if infer_item:
                        return float(infer_item.text())
            return 0.0
        
        # Prepare data for plotting
        model_names = []
        cpu_times = []
        npu1_times = []
        npu2_times = []
        
        for model in models:
            model_names.append(model)
            
            # Get CPU inference time
            cpu_time = 0.0
            for row in range(cpu_table.rowCount()):
                path_item = cpu_table.item(row, 0)
                if path_item and model in path_item.text():
                    infer_item = cpu_table.item(row, 2)
                    if infer_item:
                        cpu_time += float(infer_item.text())
            cpu_times.append(cpu_time)
            
            # Get NPU1 inference time
            npu1_time = 0.0
            for row in range(npu1_table.rowCount()):
                path_item = npu1_table.item(row, 0)
                if path_item and model in path_item.text():
                    infer_item = npu1_table.item(row, 2)
                    if infer_item:
                        npu1_time += float(infer_item.text())
            npu1_times.append(npu1_time)
            
            # Get NPU2 inference time
            npu2_time = 0.0
            for row in range(npu2_table.rowCount()):
                path_item = npu2_table.item(row, 0)
                if path_item and model in path_item.text():
                    infer_item = npu2_table.item(row, 2)
                    if infer_item:
                        npu2_time += float(infer_item.text())
            npu2_times.append(npu2_time)
        
        # Set up bar positions
        x = range(len(model_names))
        width = 0.25
        
        # Create bars
        ax.bar([i - width for i in x], cpu_times, width, label='CPU', color='skyblue')
        ax.bar(x, npu1_times, width, label='NPU1', color='gold')
        ax.bar([i + width for i in x], npu2_times, width, label='NPU2', color='orange')
        
        # Add labels and legend
        ax.set_xlabel('Models')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Time Comparison')
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
        dialog.setWindowTitle("Partition Assignments")
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