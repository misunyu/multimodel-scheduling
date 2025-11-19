import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

class DataProcessor:
    """
    Class for processing profiling data and results.
    Handles data collection, analysis, and preparation for visualization.
    """
    
    def __init__(self, log_callback=None):
        """
        Initialize the data processor.
        
        Args:
            log_callback: Function to call for logging messages
        """
        self.log_callback = log_callback
        
    def log(self, message):
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(message)
    
    def collect_values(self, table, col_index):
        """
        Collect values from a table column.
        
        Args:
            table: QTableWidget containing the data
            col_index: Column index to collect values from
            
        Returns:
            Dictionary mapping model names to average values
        """
        values = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if not name_item:
                continue
            # 모델 키는 파일 경로의 최상위 폴더가 아니라 파일 이름(확장자 제거)을 사용
            path_text = name_item.text()
            base = os.path.basename(path_text)
            model_name, _ = os.path.splitext(base)
            try:
                val = float(table.item(row, col_index).text())
                values.setdefault(model_name, []).append(val)
            except:
                continue
        return {k: np.mean(v) for k, v in values.items()}
    
    def collect_cpu_infer_per_partition(self, cpu_table):
        """
        Collect CPU inference times from the CPU tab, grouped by model.
        
        Args:
            cpu_table: QTableWidget containing CPU profiling data
            
        Returns:
            Dictionary mapping model keys to lists of inference times
        """
        cpu_infer_per_partition = {}
        for row in range(cpu_table.rowCount()):
            path_item = cpu_table.item(row, 0)
            infer_item = cpu_table.item(row, 2)
            if not path_item or not infer_item:
                continue
            rel_path = path_item.text()
            infer_time = float(infer_item.text())
            # 모델 키는 파일 이름(확장자 제거)
            base = os.path.basename(rel_path)
            model_key, _ = os.path.splitext(base)
            # cpu_tab에 입력된 해당 모델의 모든 항목의 추론 시간을 합산 대상으로 포함
            cpu_infer_per_partition.setdefault(model_key, []).append(infer_time)
        return cpu_infer_per_partition
    
    def collect_npu_values(self, npu1_table, npu2_table):
        """
        Collect NPU load and inference times.
        
        Args:
            npu1_table: QTableWidget containing NPU1 profiling data
            npu2_table: QTableWidget containing NPU2 profiling data
            
        Returns:
            Tuple of dictionaries (npu1_load, npu1_infer, npu2_load, npu2_infer)
        """
        npu1_load = self.collect_values(npu1_table, 1)
        npu1_infer = self.collect_values(npu1_table, 2)
        npu2_load = self.collect_values(npu2_table, 1)
        npu2_infer = self.collect_values(npu2_table, 2)
        return npu1_load, npu1_infer, npu2_load, npu2_infer
    
    def process_profiling_results(self, valid_model_onnx, cpu_table, npu1_table, npu2_table):
        """
        Process profiling results from all tables.
        
        Args:
            valid_model_onnx: Dictionary of valid ONNX models with their load and inference times
            cpu_table: QTableWidget containing CPU profiling data
            npu1_table: QTableWidget containing NPU1 profiling data
            npu2_table: QTableWidget containing NPU2 profiling data
            
        Returns:
            Tuple of (all_models, cpu_infer_per_partition, npu1_load, npu1_infer, npu2_load, npu2_infer)
        """
        # Collect CPU inference times per partition
        cpu_infer_per_partition = self.collect_cpu_infer_per_partition(cpu_table)
        
        # Collect NPU values
        npu1_load, npu1_infer, npu2_load, npu2_infer = self.collect_npu_values(npu1_table, npu2_table)
        
        # Get all unique model names (명시적 분할 미사용: 파일 이름 기준 키)
        # valid_model_onnx에 있는 키는 무시하고, 실제 테이블에서 수집한 키들만 사용
        all_models = set(cpu_infer_per_partition.keys()).union(
            npu1_load.keys(), npu1_infer.keys(), npu2_load.keys(), npu2_infer.keys()
        )
        
        return all_models, cpu_infer_per_partition, npu1_load, npu1_infer, npu2_load, npu2_infer
    
    def find_best_device_for_model(self, model, cpu_infer, npu1_infer, npu2_infer):
        """
        Find the best device for a model based on inference times.
        
        Args:
            model: Model name
            cpu_infer: Dictionary of CPU inference times
            npu1_infer: Dictionary of NPU1 inference times
            npu2_infer: Dictionary of NPU2 inference times
            
        Returns:
            Tuple of (best_device, best_time)
        """
        devices = []
        
        # Check CPU inference time
        if model in cpu_infer:
            devices.append(("CPU", cpu_infer[model]))
        
        # Check NPU1 inference time
        if model in npu1_infer:
            devices.append(("NPU1", npu1_infer[model]))
        
        # Check NPU2 inference time
        if model in npu2_infer:
            devices.append(("NPU2", npu2_infer[model]))
        
        # Find device with minimum inference time
        if devices:
            best_device, best_time = min(devices, key=lambda x: x[1])
            return best_device, best_time
        
        # Default to CPU if no data available
        return "CPU", 0.0
    
    def assign_models_to_devices(self, models, cpu_infer, npu1_infer, npu2_infer):
        """
        Assign models to devices based on best performance.
        
        Args:
            models: List of model names
            cpu_infer: Dictionary of CPU inference times
            npu1_infer: Dictionary of NPU1 inference times
            npu2_infer: Dictionary of NPU2 inference times
            
        Returns:
            List of (model_name, device) tuples
        """
        assignments = []
        
        for model in models:
            best_device, _ = self.find_best_device_for_model(model, cpu_infer, npu1_infer, npu2_infer)
            assignments.append((model, best_device))
            
        return assignments
    
    # 파티션 파일 관련 기능은 더 이상 사용하지 않음 (명시적 분할 미사용)