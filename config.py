"""
Central Configuration File for SMART Goals Dashboard
==================================================

This file contains all the file path references used throughout the project.
Change the file names here once, and all other files will automatically use the updated references.

Usage:
    from config import INPUT_FILE, OUTPUT_FILE, etc.
"""

# =============================================================================
# FILE PATH CONFIGURATIONS
# =============================================================================

# Main data files
INPUT_FILE = "goals_input.xlsx"      # Original input file
OUTPUT_FILE = "goals_audited_output.xlsx"        # Processed output file
AUDITED_OUTPUT_FILE = "goals_audited_output.xlsx"  # Audited output file

# Backup files
BACKUP_DIR = "backups/"
BACKUP_PREFIX = "goals_input_backup_"

# CSV files
CSV_INPUT_FILE = "goals_input.csv"       # CSV version of input

# =============================================================================
# DASHBOARD CONFIGURATIONS
# =============================================================================

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "SMART Goal Quality Dashboard"

# Cache settings
CACHE_TTL = 60  # Time to live for cached data in seconds

# =============================================================================
# SMART FRAMEWORK CONFIGURATIONS
# =============================================================================

# SMART pillar columns
SMART_COLUMNS = ["Specific", "Measurable", "Achievable", "Relevant", "TimeBound"]

# Quality thresholds
HIGH_QUALITY_THRESHOLD = 4
MEDIUM_QUALITY_THRESHOLD = 3
LOW_QUALITY_THRESHOLD = 2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_input_file():
    """Get the current input file path"""
    return INPUT_FILE

def get_output_file():
    """Get the current output file path"""
    return OUTPUT_FILE

def get_audited_output_file():
    """Get the current audited output file path"""
    return AUDITED_OUTPUT_FILE

def get_backup_path():
    """Get the backup directory path"""
    return BACKUP_DIR

def get_csv_input_file():
    """Get the CSV input file path"""
    return CSV_INPUT_FILE

# =============================================================================
# FILE VALIDATION
# =============================================================================

import os

def validate_files():
    """Validate that all configured files exist"""
    files_to_check = [
        INPUT_FILE,
        OUTPUT_FILE,
        CSV_INPUT_FILE
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Warning: The following files are missing: {', '.join(missing_files)}")
        return False
    
    return True

# =============================================================================
# CONFIGURATION INFO
# =============================================================================

def print_config_info():
    """Print current configuration information"""
    print("=" * 50)
    print("SMART GOALS DASHBOARD CONFIGURATION")
    print("=" * 50)
    print(f"Input File: {INPUT_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Audited Output File: {AUDITED_OUTPUT_FILE}")
    print(f"CSV Input File: {CSV_INPUT_FILE}")
    print(f"Backup Directory: {BACKUP_DIR}")
    print(f"Dashboard Port: {DASHBOARD_PORT}")
    print("=" * 50)

if __name__ == "__main__":
    print_config_info()
    validate_files()
