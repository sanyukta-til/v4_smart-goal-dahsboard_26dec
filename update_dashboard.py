"""
Automated Workflow Script: Update Dashboard After Input File Changes
====================================================================

This script automates the complete workflow when you update goals_input.xlsx:
1. Processes the input file through goal_audit.py
2. Generates the output file for the dashboard
3. Clears dashboard cache (if running locally)

Usage:
    python update_dashboard.py

Or with custom input file:
    python update_dashboard.py --input your_file.xlsx
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from config import get_input_file, get_output_file, get_backup_path

def print_header():
    """Print a nice header"""
    print("=" * 70)
    print("🔄 SMART Goals Dashboard Update Workflow")
    print("=" * 70)
    print()

def check_file_exists(file_path, file_type="Input"):
    """Check if a file exists"""
    if not os.path.exists(file_path):
        print(f"❌ {file_type} file not found: {file_path}")
        print(f"💡 Please ensure the file exists before running this script.")
        return False
    return True

def run_goal_audit(input_file, output_file):
    """Run the goal audit script to process the input file"""
    print(f"📊 Step 1: Processing input file...")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")
    print()
    
    try:
        # Run goal_audit.py
        result = subprocess.run(
            [sys.executable, 'goal_audit.py', input_file, '--output', output_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Goal audit completed successfully!")
        print()
        
        # Print any important output
        if result.stdout:
            # Filter out verbose output, show only important messages
            lines = result.stdout.split('\n')
            important_lines = [line for line in lines if any(keyword in line.lower() for keyword in ['✅', '❌', '📊', '📋', 'error', 'warning', 'success'])]
            if important_lines:
                print("📋 Processing Summary:")
                for line in important_lines[:10]:  # Show first 10 important lines
                    if line.strip():
                        print(f"   {line}")
                print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running goal audit: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def verify_output_file(output_file):
    """Verify that the output file was created successfully"""
    print(f"📋 Step 2: Verifying output file...")
    
    if not os.path.exists(output_file):
        print(f"❌ Output file was not created: {output_file}")
        return False
    
    file_size = os.path.getsize(output_file)
    if file_size == 0:
        print(f"⚠️  Warning: Output file is empty: {output_file}")
        return False
    
    print(f"✅ Output file verified: {output_file} ({file_size:,} bytes)")
    print()
    return True

def print_summary(input_file, output_file):
    """Print a summary of what was done"""
    print("=" * 70)
    print("✅ Update Complete!")
    print("=" * 70)
    print()
    print("📊 Files:")
    print(f"   • Input processed:  {input_file}")
    print(f"   • Output generated: {output_file}")
    print()
    print("🚀 Next Steps:")
    print("   1. If dashboard is running locally, refresh your browser")
    print("   2. If dashboard is on Streamlit Cloud, it will auto-update")
    print("   3. The dashboard cache will refresh automatically (60 seconds)")
    print()
    print("💡 Tip: The dashboard reads from the output file automatically.")
    print("   No manual intervention needed - just refresh the browser!")
    print("=" * 70)

def main():
    """Main workflow function"""
    parser = argparse.ArgumentParser(
        description='Update dashboard after changing goals_input.xlsx',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_dashboard.py
  python update_dashboard.py --input my_goals.xlsx
  python update_dashboard.py --input goals_input.xlsx --output goals_audited_output.xlsx
        """
    )
    parser.add_argument(
        '--input',
        default=get_input_file(),
        help=f'Input Excel file (default: {get_input_file()})'
    )
    parser.add_argument(
        '--output',
        default=get_output_file(),
        help=f'Output Excel file (default: {get_output_file()})'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    # Check if input file exists
    if not check_file_exists(args.input, "Input"):
        sys.exit(1)
    
    # Run goal audit
    if not run_goal_audit(args.input, args.output):
        print("❌ Workflow failed at goal audit step.")
        sys.exit(1)
    
    # Verify output
    if not verify_output_file(args.output):
        print("❌ Workflow failed at verification step.")
        sys.exit(1)
    
    # Print summary
    print_summary(args.input, args.output)
    
    print("\n🎉 All done! Your dashboard is ready with updated data.\n")

if __name__ == "__main__":
    main()

