# 🚀 Google Colab Setup Guide for Fantasy Football MCTS

## **Quick Start (3 Easy Steps)**

### **Step 1: Prepare Your Data Locally**
```bash
# Run this in your local terminal
cd /Users/mattwang/Documents/fantasy/fantasy-football
python models/colab_data_prep.py
```
This creates `colab_data.zip` with all your data files.

### **Step 2: Upload to Google Colab**
1. Open `models/draft_strategy_mcts.ipynb` in Google Colab
2. Upload the `colab_data.zip` file to Colab (drag & drop into file panel)
3. Run the notebook!

### **Step 3: Let the Notebook Auto-Load**
The notebook will automatically:
- Detect your ZIP file
- Extract your data
- Load your custom settings
- Run the MCTS analysis

---

## **Alternative Methods**

### **Method A: Manual File Upload**
If you don't want to use the ZIP file:
1. Open the MCTS notebook in Colab
2. When prompted, upload these files manually:
   - `draft_board.csv`
   - `FantasyPros_2025_Overall_ADP_Rankings.csv`
3. The notebook will load them directly

### **Method B: Google Drive**
1. Upload your data files to Google Drive
2. In the notebook, update the file paths:
   ```python
   draft_board_path = '/content/drive/MyDrive/your-folder/draft_board.csv'
   adp_path = '/content/drive/MyDrive/your-folder/FantasyPros_2025_Overall_ADP_Rankings.csv'
   ```
3. Run the Drive mount option

### **Method C: Sample Data**
If you just want to test the system:
- The notebook will automatically generate realistic sample data
- No uploads required - just run and go!

---

## **What Gets Loaded**

Your prepared data package includes:

### **📊 Data Files:**
- `draft_board.csv` - Your VORP calculations and player values
- `FantasyPros_2025_Overall_ADP_Rankings.csv` - ADP rankings  
- `rookie_data_clean.csv` - Cleaned rookie historical data
- `metadata.json` - Your league settings and model parameters

### **⚙️ Settings:**
- League configuration (12 teams, roster spots, etc.)
- Model parameters (risk penalty, MCTS simulations)
- Opponent modeling settings
- All your customizations preserved!

---

## **Expected Workflow**

1. **Prepare locally** → `python models/colab_data_prep.py`
2. **Upload ZIP** → Drag `colab_data.zip` to Colab
3. **Run notebook** → Everything loads automatically
4. **Get results** → Best strategy identified and saved
5. **Download model** → `best_draft_strategy_model.pkl` for live drafts

---

## **Troubleshooting**

### **"No data files found"**
- Make sure you uploaded `colab_data.zip`
- Check the file panel in Colab to confirm upload
- Try the manual upload option as backup

### **"Import errors"**
- Run the package installation cell first
- Restart runtime if needed: `Runtime → Restart Runtime`

### **"Sample data being used"**
- This is normal if no files uploaded
- Sample data still demonstrates the full system
- Results will be realistic for testing

---

## **Files You'll Get Back**

After running in Colab, download these files:

### **🏆 Main Results:**
- `best_draft_strategy_model.pkl` - Complete trained model
- `draft_strategy_report.txt` - Detailed analysis report
- `model_usage_example.py` - How to use the model

### **📊 Analysis:**
- Strategy comparison charts
- Performance metrics
- Draft pick recommendations

---

## **Next Steps**

1. **Download your model** from Colab
2. **Use for live drafts** - Load the pickle file locally
3. **Integrate with apps** - Use the model for real-time recommendations
4. **Iterate and improve** - Adjust parameters and retrain

**The MCTS notebook is completely self-contained and will work in Colab with any of these data loading methods!**
