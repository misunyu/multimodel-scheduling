# Multimodel Scheduling Application

## Project Structure

The application has been modularized to improve readability and maintainability. The code is now organized into the following modules:

### Main Files
- `main.py`: Entry point for the application
- `multimodel_gui.py`: Legacy entry point (for backward compatibility)

### Core Components
- `unified_viewer.py`: Main viewer class that handles the UI and coordinates the different components
- `view_handlers.py`: Classes for handling different views (YOLO, ResNet) and video feeding
- `model_processors.py`: Functions for processing models on CPU and NPU
- `image_processing.py`: Functions for image preprocessing, postprocessing, and visualization
- `utils.py`: Utility functions for logging, metrics, and image conversion

## Module Descriptions

### utils.py
Contains general utility functions:
- `async_log`: Asynchronously log model performance data
- `create_x_image`: Create a placeholder image with an X
- `convert_cv_to_qt`: Convert OpenCV images to Qt pixmaps
- `get_cpu_metrics`: Get CPU performance metrics

### image_processing.py
Contains image processing functions:
- `image_preprocess`: Preprocess images for neural network input
- `yolo_preprocess_local`: Preprocess images for YOLO model
- `resnet50_preprocess_local`: Preprocess images for ResNet50 model
- `draw_detection_boxes`: Draw detection boxes on images
- `yolo_postprocess_cpu`: Post-process YOLO model output (CPU version)
- `yolo_postprocess_npu`: Post-process YOLO model output (NPU version)

### model_processors.py
Contains model processing functions:
- `video_reader_process`: Process for reading video frames
- `run_yolo_cpu_process`: Process for running YOLO model on CPU
- `run_resnet_cpu_process`: Process for running ResNet model on CPU
- `run_yolo_npu_process`: Process for running YOLO model on NPU
- `run_resnet_npu_process`: Process for running ResNet model on NPU

### view_handlers.py
Contains view handling components:
- `ModelSignals`: Signal class for updating model views
- `ViewHandler`: Base class for handling model views
- `YoloViewHandler`: Handler for YOLO model views
- `ResNetViewHandler`: Handler for ResNet model views
- `VideoFeeder`: Class for feeding video frames to model queues

### unified_viewer.py
Contains the main viewer class:
- `UnifiedViewer`: Main viewer class that handles the UI and coordinates the different components
  - Initialization methods: `initialize_model_settings`, `initialize_ui_components`, etc.
  - View update methods: `update_view1_display`, etc.
  - Signal handling and shutdown methods: `signal_handler`, `closeEvent`, `shutdown_all`
  - Monitoring and statistics methods: `update_cpu_npu_usage`, `save_throughput_data`

## Benefits of Modularization

1. **Improved Readability**: Each module has a clear purpose and contains related functionality.
2. **Better Maintainability**: Changes to one component don't affect others, making it easier to update and fix bugs.
3. **Code Reusability**: Functions and classes can be reused in other projects or components.
4. **Easier Testing**: Smaller, focused modules are easier to test.
5. **Collaboration**: Multiple developers can work on different modules simultaneously.