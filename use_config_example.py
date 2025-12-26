#!/usr/bin/env python3
"""
Example script showing how to use the centralized configuration system
=====================================================================

This demonstrates how all files now reference the config.py file for file paths.
"""

from config import (
    INPUT_FILE, OUTPUT_FILE, AUDITED_OUTPUT_FILE, 
    get_input_file, get_output_file, get_audited_output_file,
    print_config_info, validate_files
)

def main():
    print("🔧 SMART Goals Dashboard - Configuration Example")
    print("=" * 50)
    
    # Show current configuration
    print_config_info()
    
    # Validate files exist
    print("\n📁 File Validation:")
    if validate_files():
        print("✅ All configured files are present!")
    else:
        print("❌ Some files are missing. Check the configuration.")
    
    # Show how to use the config in your code
    print("\n💡 Usage Examples:")
    print(f"   Input file: {get_input_file()}")
    print(f"   Output file: {get_output_file()}")
    print(f"   Audited output: {get_audited_output_file()}")
    
    print("\n🔄 To change file names:")
    print("   1. Edit config.py")
    print("   2. Change INPUT_FILE, OUTPUT_FILE, etc.")
    print("   3. All other files will automatically use the new names!")
    
    print("\n📋 Files that use this config:")
    print("   • smart_dashboard.py")
    print("   • goal_audit.py") 
    print("   • auto_sync.py")
    print("   • Any new files you create")

if __name__ == "__main__":
    main()
