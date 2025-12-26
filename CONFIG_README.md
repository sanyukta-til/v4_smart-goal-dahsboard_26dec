# Centralized Configuration System

## Overview

This project now uses a centralized configuration system where all file references are managed through `config.py`. This means you only need to change file names in one place, and all other files will automatically use the updated references.

## How It Works

### 1. Configuration File (`config.py`)

The `config.py` file contains all the file path references:

```python
# Main data files
INPUT_FILE = "goals_input.xlsx"          # Original input file
OUTPUT_FILE = "goals_output.xlsx"        # Processed output file
AUDITED_OUTPUT_FILE = "goals_audited_output.xlsx"  # Audited output file

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "SMART Goal Quality Dashboard"
```

### 2. Using the Config in Your Code

Instead of hardcoding file names, import from config:

```python
from config import get_input_file, get_output_file, DASHBOARD_TITLE

# Use the configured file paths
df = pd.read_excel(get_input_file())
st.set_page_config(page_title=DASHBOARD_TITLE)
```

## Files Updated

The following files have been updated to use the centralized configuration:

- ✅ `smart_dashboard.py` - Uses config for file paths and settings
- ✅ `goal_audit.py` - Uses config for input/output files
- ✅ `auto_sync.py` - Uses config for file paths and backup directory

## How to Change File Names

### Before (Old Way)
You had to change file names in multiple places:
- `smart_dashboard.py` line 25: `pd.read_excel("goals_output.xlsx")`
- `goal_audit.py` line 346: `default='goals_input.xlsx'`
- `auto_sync.py` line 7: `excel_file='goals_input.xlsx'`
- And many more...

### After (New Way)
Just change the file name in `config.py`:

```python
# In config.py
INPUT_FILE = "my_new_goals_file.xlsx"  # Change this once
OUTPUT_FILE = "my_processed_goals.xlsx"  # Change this once
```

All other files automatically use the new names! 🎉

## Configuration Options

### File Paths
- `INPUT_FILE` - Original input Excel file
- `OUTPUT_FILE` - Processed output Excel file  
- `AUDITED_OUTPUT_FILE` - Audited output Excel file
- `CSV_INPUT_FILE` - CSV version of input file
- `BACKUP_DIR` - Directory for backup files

### Dashboard Settings
- `DASHBOARD_PORT` - Port number for the dashboard
- `DASHBOARD_TITLE` - Title shown in browser tab
- `CACHE_TTL` - Cache time-to-live in seconds

### SMART Framework Settings
- `SMART_COLUMNS` - List of SMART pillar column names
- `HIGH_QUALITY_THRESHOLD` - Score threshold for high quality
- `MEDIUM_QUALITY_THRESHOLD` - Score threshold for medium quality
- `LOW_QUALITY_THRESHOLD` - Score threshold for low quality

## Helper Functions

The config file provides helper functions:

```python
from config import get_input_file, get_output_file, validate_files

# Get current file paths
input_file = get_input_file()
output_file = get_output_file()

# Validate all files exist
if validate_files():
    print("All files are present!")
```

## Testing the Configuration

Run the example script to test your configuration:

```bash
python use_config_example.py
```

This will show you:
- Current configuration
- File validation status
- Usage examples

## Benefits

1. **Single Source of Truth** - All file references in one place
2. **Easy Maintenance** - Change file names once, update everywhere
3. **Consistency** - No more mismatched file names across files
4. **Flexibility** - Easy to switch between different datasets
5. **Documentation** - Clear overview of all configuration options

## Adding New Files

When creating new files, always import from config:

```python
from config import get_input_file, get_output_file, DASHBOARD_TITLE

# Your code here using the config values
```

This ensures your new files automatically benefit from the centralized configuration system.
