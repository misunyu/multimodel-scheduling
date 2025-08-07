# Model Profiler Package

This package contains modules for profiling ONNX and NPU models. It was created by refactoring the original `model_profiler_gui.py` file into a more modular structure for better maintainability and readability.

## Package Structure

The package is organized into the following modules:

- `model_profiler.py`: Core functionality for profiling ONNX and NPU models
- `data_processor.py`: Data processing and analysis functionality
- `ui_components.py`: UI components and visualization functionality
- `file_manager.py`: File and model management functionality
- `__init__.py`: Package initialization and exports

## Main Application

The main application is now split into two files:

- `model_profiler_app.py`: Contains the main application class `ONNXProfilerApp`
- `model_profiler_main.py`: Entry point script that launches the application

## Usage

To run the application, use the following command:

```bash
python model_profiler_main.py
```

## Benefits of Refactoring

The refactoring of the original monolithic file into a modular package structure provides several benefits:

1. **Improved Maintainability**: Each module has a single responsibility, making it easier to maintain and update.
2. **Better Readability**: Smaller, focused files are easier to read and understand.
3. **Enhanced Reusability**: Components can be reused in other parts of the application or in other projects.
4. **Easier Testing**: Isolated components are easier to test individually.
5. **Simplified Collaboration**: Multiple developers can work on different modules simultaneously with fewer conflicts.

## Module Details

### ModelProfiler

The `ModelProfiler` class handles the core functionality of profiling ONNX and NPU models. It provides methods for:

- Profiling ONNX models on CPU
- Profiling models on NPU devices
- Detecting custom operations in ONNX models
- Generating dummy input data for models

### DataProcessor

The `DataProcessor` class handles data processing and analysis. It provides methods for:

- Collecting and processing profiling results
- Finding the best device for each model
- Assigning models to devices based on performance

### UIComponents

The `UIComponents` class handles UI components and visualization. It provides methods for:

- Initializing and populating tables
- Highlighting deployment results
- Creating charts and dialogs

### FileManager

The `FileManager` class handles file and model management. It provides methods for:

- Collecting model files
- Getting selected paths from the UI
- Saving and loading configuration files
- Managing sample data