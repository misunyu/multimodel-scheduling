# Changes Made to Support Custom Scheduling File

## Overview
The application has been modified to accept a custom scheduling information file (e.g., model_schedules.yaml) as a command-line parameter. This allows users to specify different scheduling configurations without modifying the code.

## Changes Made

### 1. Modified `multimodel_gui.py`
- Added argument parsing to accept a scheduling file parameter
- Added `--schedule` (or `-s`) command-line option with default value of 'model_schedules.yaml'
- Modified the UnifiedViewer instantiation to pass the schedule file parameter

### 2. Modified `unified_viewer.py`
- Updated the `__init__` method to accept a `schedule_file` parameter
- Modified the `initialize_model_settings` method to use the provided schedule file
- Updated error messages to reference the actual file path being used

## Usage Examples
```bash
# Use default model_schedules.yaml
python multimodel_gui.py

# Use a custom scheduling file
python multimodel_gui.py --schedule custom_schedules.yaml

# Use short form parameter
python multimodel_gui.py -s another_schedule.yaml
```

## Testing
The changes have been tested with:
- Default behavior (no parameter provided)
- Custom schedule file using long-form parameter (--schedule)
- Custom schedule file using short-form parameter (-s)
- Error handling for non-existent files

All tests confirmed that the application correctly uses the specified scheduling file or falls back to defaults when the file cannot be loaded.