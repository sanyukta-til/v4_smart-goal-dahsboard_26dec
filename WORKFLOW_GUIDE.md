# Complete Workflow Guide: Adding Domain Column and Processing New Input Files

## 🎯 Overview

This guide explains the complete step-by-step workflow for processing new input files with the Domain column functionality. The system now supports Domain filtering and will work seamlessly when you update your input file.

## 📋 Complete Workflow Steps

### Step 1: Prepare Your New Input File

When you want to update the input file with new data:

1. **Replace the input file**: Update `goals_input.xlsx` with your new data
2. **Add Domain column**: Add Domain column with YOUR desired values for each person
3. **Backup is automatic**: Original files are automatically backed up

### Step 2: Process the Data (Automatic)

Run the goal audit script to process your new input file:

```bash
python goal_audit.py goals_input.xlsx --output goals_output.xlsx
```

**What happens automatically:**
- ✅ Reads the new input file
- ✅ Preserves YOUR Domain column values exactly
- ✅ Processes SMART framework analysis
- ✅ Creates output file with all analytics
- ✅ Generates audited output file
- ✅ Creates backup of original files

### Step 3: Run the Dashboard

```bash
python smart_dashboard.py
```

**What you'll see:**
- ✅ Domain filter in the sidebar
- ✅ All existing functionality working
- ✅ Domain-specific analytics and visualizations
- ✅ No crashes or errors

## 🔧 Domain Column Handling

The system preserves YOUR Domain column values exactly as you provide them:

- ✅ **No automatic mapping** - Your domain values are preserved exactly
- ✅ **User-controlled** - You decide what domain each person belongs to
- ✅ **Flexible naming** - Use any domain names you prefer (e.g., 'Technology', 'Sales', 'Marketing')
- ✅ **Consistent values** - Use consistent naming for better filtering

## 🚀 What Happens When You Change Input File

### Automatic Processing Pipeline

1. **File Detection**: System detects new input file
2. **Domain Preservation**: Preserves YOUR Domain column values exactly
3. **Data Processing**: Runs SMART framework analysis
4. **Output Generation**: Creates processed output file
5. **Dashboard Update**: Dashboard automatically uses new data

### Domain Filter Features

- **Multi-select filtering**: Choose multiple domains
- **Cascading filters**: Domain filter works with Business Unit and Department
- **Real-time updates**: All charts and analytics update based on domain selection
- **Domain-specific insights**: View performance by organizational domain

## 📊 Dashboard Features with Domain Support

### Sidebar Filters
- Business Unit (existing)
- Department (existing)
- **Domain (NEW)** - Filter by organizational domain
- Manager/Lead (existing)

### Analytics by Domain
- Domain-specific goal quality metrics
- Domain performance comparisons
- Domain-wise SMART score analysis
- Domain-specific manager performance

### Visualizations
- Domain distribution charts
- Domain vs. Business Unit comparisons
- Domain-specific goal quality breakdowns
- Domain performance trends

## 🔄 Complete Workflow Example

### Scenario: You have a new goals file to process

1. **Replace input file**:
   ```bash
   # Copy your new file to goals_input.xlsx
   cp your_new_goals.xlsx goals_input.xlsx
   ```

2. **Process the data**:
   ```bash
   python goal_audit.py
   ```

3. **Run dashboard**:
   ```bash
   python smart_dashboard.py
   ```

4. **Use domain filtering**:
   - Open dashboard in browser
   - Use Domain filter in sidebar
   - Select domains to filter data
   - View domain-specific analytics

## 🛠️ Troubleshooting

### If Domain Column is Missing
The system automatically adds it based on Business Unit mapping.

### If Processing Fails
1. Check input file format
2. Ensure required columns exist
3. Check backup files in `backups/` directory

### If Dashboard Doesn't Load
1. Ensure `goals_output.xlsx` exists
2. Run `python goal_audit.py` first
3. Check for any error messages

## 📁 File Structure

```
v2_goals library/
├── goals_input.xlsx          # Your input file (update this)
├── goals_output.xlsx         # Processed output (auto-generated)
├── goals_audited_output.xlsx # Audited output (auto-generated)
├── smart_dashboard.py       # Dashboard application
├── goal_audit.py            # Processing script
├── config.py                # Configuration file
└── backups/                 # Automatic backups
    ├── goals_input_backup_*.xlsx
    └── goals_input_backup_*.csv
```

## ✅ Verification Checklist

Before using the dashboard:

- [ ] Input file updated with new data
- [ ] Domain column exists (or will be auto-added)
- [ ] `python goal_audit.py` runs successfully
- [ ] `goals_output.xlsx` is created/updated
- [ ] `python smart_dashboard.py` runs without errors
- [ ] Domain filter appears in dashboard sidebar
- [ ] Domain filtering works correctly

## 🎉 Benefits of Domain Support

- **Organizational Insights**: View goals by business domain
- **Better Analytics**: Domain-specific performance metrics
- **Improved Filtering**: More granular data analysis
- **Strategic View**: Understand goal distribution across domains
- **No Crashes**: Robust error handling and data validation

## 💡 Pro Tips

1. **Always backup**: The system creates automatic backups
2. **Check domain mapping**: Verify Business Unit to Domain mapping is correct
3. **Use cascading filters**: Combine Domain with Business Unit and Department
4. **Monitor performance**: Large files may take longer to process
5. **Test first**: Run with a small subset before processing large files

---

**The system is now fully ready to handle new input files with Domain column support!** 🚀
