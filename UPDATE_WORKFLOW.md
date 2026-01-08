# 📋 Complete Workflow: Updating Input File and Refreshing Dashboard

## 🎯 Quick Answer

**Yes, everything will run the same!** When you update `goals_input.xlsx`, just follow these simple steps:

## ✅ Step-by-Step Workflow

### Step 1: Update Your Input File
1. Open `goals_input.xlsx`
2. **Remove rows** for people who left the organization
3. **Add rows** for new people who joined
4. Save the file

### Step 2: Process the Updated File
Run this command to process your updated file:

```bash
python update_dashboard.py
```

**What this does:**
- ✅ Processes `goals_input.xlsx` through the audit system
- ✅ Generates `goals_audited_output.xlsx` (the file the dashboard reads)
- ✅ Updates all SMART analysis and metrics
- ✅ Handles new employees automatically
- ✅ Removes data for employees who left

### Step 3: Refresh the Dashboard

**If running locally:**
- Click the **"🔄 Refresh Dashboard Data"** button in the sidebar, OR
- Simply refresh your browser (F5 or Ctrl+R)
- The dashboard cache automatically refreshes every 60 seconds

**If deployed on Streamlit Cloud:**
- The dashboard automatically detects the new file
- Just refresh your browser - no button needed
- Changes appear within 1-2 minutes

## 🔄 Alternative: Manual Workflow

If you prefer to run the steps manually:

```bash
# Step 1: Process the input file
python goal_audit.py goals_input.xlsx --output goals_audited_output.xlsx

# Step 2: Refresh dashboard (if running locally)
# Just click the refresh button or reload the page
```

## 📊 What Happens Automatically

When you update `goals_input.xlsx` and run the update script:

1. **Data Processing:**
   - Reads your updated input file
   - Analyzes all goals using SMART framework
   - Calculates scores and metrics
   - Generates output file

2. **Dashboard Updates:**
   - Automatically reads the new output file
   - Updates all charts and tables
   - Reflects new employee data
   - Removes data for employees who left

3. **No Manual Intervention:**
   - No need to restart the dashboard
   - No need to manually clear cache
   - Everything updates automatically

## 🎯 Key Points

✅ **Same Process Every Time:**
- Update `goals_input.xlsx`
- Run `python update_dashboard.py`
- Refresh browser

✅ **Automatic Handling:**
- New employees → Automatically analyzed
- Removed employees → Automatically removed from dashboard
- All metrics → Automatically recalculated

✅ **No Data Loss:**
- Original files are backed up automatically
- Previous versions saved in `backups/` folder

## 🚀 Quick Reference

| Action | Command/Step |
|--------|-------------|
| Update input file | Edit `goals_input.xlsx` |
| Process updates | `python update_dashboard.py` |
| Refresh dashboard (local) | Click refresh button or F5 |
| Refresh dashboard (cloud) | Just refresh browser |

## 💡 Pro Tips

1. **Always backup first:** The system creates automatic backups, but you can also manually backup `goals_input.xlsx` before major changes

2. **Check the output:** After running `update_dashboard.py`, verify that `goals_audited_output.xlsx` was created/updated

3. **Wait for processing:** Large files may take 30-60 seconds to process - be patient!

4. **Verify in dashboard:** After refreshing, check that:
   - New employees appear in the employee list
   - Removed employees no longer appear
   - All metrics are updated

## ❓ Troubleshooting

**Dashboard shows old data?**
- Click the "🔄 Refresh Dashboard Data" button in the sidebar
- Or wait 60 seconds for automatic cache refresh
- Or refresh your browser (F5)

**Processing fails?**
- Check that `goals_input.xlsx` exists
- Verify the file isn't open in Excel
- Check for any error messages in the console

**New employees not showing?**
- Verify they were added to `goals_input.xlsx`
- Check that `update_dashboard.py` ran successfully
- Refresh the dashboard

## 📁 File Flow

```
goals_input.xlsx (YOU UPDATE THIS)
         ↓
   [update_dashboard.py]
         ↓
goals_audited_output.xlsx (AUTO-GENERATED)
         ↓
   [smart_dashboard.py]
         ↓
   Dashboard Display
```

## ✅ Summary

**Every time you update the input file:**
1. Update `goals_input.xlsx` ✅
2. Run `python update_dashboard.py` ✅
3. Refresh dashboard ✅

**That's it!** The system handles everything else automatically. 🎉



