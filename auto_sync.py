import pandas as pd
import time
import os
import subprocess
import sys
from datetime import datetime
import shutil
from config import get_csv_input_file, get_input_file, get_backup_path, get_output_file

def sync_csv_to_excel(csv_file=None, excel_file=None, backup_dir=None):
    """
    Automatically sync CSV changes to Excel file with backup functionality
    """
    
    # Use config defaults if not provided
    if csv_file is None:
        csv_file = get_csv_input_file()
    if excel_file is None:
        excel_file = get_input_file()
    if backup_dir is None:
        backup_dir = get_backup_path()
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file '{csv_file}' not found!")
        print("Please create a CSV file from your Excel file first.")
        return False
    
    try:
        # Read CSV file
        df_csv = pd.read_csv(csv_file)
        print(f"✅ CSV loaded: {len(df_csv)} rows, {len(df_csv.columns)} columns")
        
        # Create backup of current Excel file
        if os.path.exists(excel_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_dir}/goals_input_backup_{timestamp}.xlsx"
            shutil.copy2(excel_file, backup_file)
            print(f"📁 Backup created: {backup_file}")
        
        # Write to Excel file
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df_csv.to_excel(writer, sheet_name='All_Goals', index=False)
        
        print(f"✅ Excel file updated: {excel_file}")
        print(f"📊 Data synced: {len(df_csv)} rows")
        
        # Run goal audit to produce OUTPUT_FILE for dashboard
        print("\n🔄 Running goal audit to update dashboard data...")
        try:
            audit_result = run_goal_audit(excel_file)
            if audit_result:
                print("✅ Dashboard data updated successfully!")
            else:
                print("⚠️ Audit completed with warnings - dashboard may show stale data")
        except Exception as audit_error:
            print(f"❌ Audit failed: {str(audit_error)}")
            print("⚠️ Dashboard will show stale data until audit is run manually")
            print("💡 Run manually: python goal_audit.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error syncing files: {str(e)}")
        return False

def watch_and_sync(csv_file='goals_input.csv', excel_file='goals_input.xlsx', check_interval=5):
    """
    Watch CSV file for changes and auto-sync to Excel
    """
    print(f"🔍 Watching '{csv_file}' for changes...")
    print(f"📁 Auto-syncing to '{excel_file}' every {check_interval} seconds")
    print("Press Ctrl+C to stop watching")
    
    last_modified = 0
    
    try:
        while True:
            if os.path.exists(csv_file):
                current_modified = os.path.getmtime(csv_file)
                
                if current_modified > last_modified:
                    print(f"\n🔄 Changes detected at {datetime.now().strftime('%H:%M:%S')}")
                    if sync_csv_to_excel(csv_file, excel_file):
                        last_modified = current_modified
                        print("✅ Sync complete - Dashboard data is up-to-date!")
                        print("💡 Refresh dashboard (F5) to see latest data")
                    else:
                        print("❌ Sync failed, keeping previous version")
                else:
                    print(".", end="", flush=True)
            else:
                print(f"❌ CSV file '{csv_file}' not found!")
                print("Please create the CSV file first.")
                break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Auto-sync stopped by user")
        print("💡 You can still manually sync using: python auto_sync.py")

def run_goal_audit(input_file=None, output_file=None):
    """
    Run goal_audit.py to produce OUTPUT_FILE for dashboard
    Returns True if successful, False otherwise
    """
    if input_file is None:
        input_file = get_input_file()
    if output_file is None:
        output_file = get_output_file()
    
    try:
        # Run goal_audit.py as a subprocess
        result = subprocess.run(
            [sys.executable, 'goal_audit.py', input_file, '--output', output_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Audit output written to: {output_file}")
            return True
        else:
            print(f"⚠️ Audit returned code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")  # Print first 500 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Audit timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running audit: {str(e)}")
        return False

def create_csv_from_excel(excel_file='goals_input.xlsx', csv_file='goals_input.csv'):
    """
    Create CSV file from existing Excel file
    """
    try:
        if not os.path.exists(excel_file):
            print(f"❌ Excel file '{excel_file}' not found!")
            return False
        
        # Read Excel file - try different sheet names
        try:
            df = pd.read_excel(excel_file, 'All_Goals')
        except:
            # Try the first sheet if 'All_Goals' doesn't exist
            xls = pd.ExcelFile(excel_file)
            df = pd.read_excel(excel_file, xls.sheet_names[0])
        
        # Save as CSV
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV file created: {csv_file}")
        print(f"📊 Data exported: {len(df)} rows, {len(df.columns)} columns")
        print(f"💡 You can now edit '{csv_file}' in any text editor")
        return True
        
    except Exception as e:
        print(f"❌ Error creating CSV: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-sync CSV to Excel for Goals Dashboard')
    parser.add_argument('--mode', choices=['sync', 'watch', 'create-csv'], default='sync',
                       help='Mode: sync (one-time), watch (continuous), create-csv (from Excel)')
    parser.add_argument('--csv', default='goals_input.csv', help='CSV file path')
    parser.add_argument('--excel', default='goals_input.xlsx', help='Excel file path')
    parser.add_argument('--interval', type=int, default=5, help='Watch interval in seconds')
    
    args = parser.parse_args()
    
    if args.mode == 'create-csv':
        create_csv_from_excel(args.excel, args.csv)
    elif args.mode == 'watch':
        watch_and_sync(args.csv, args.excel, args.interval)
    else:
        sync_csv_to_excel(args.csv, args.excel) 