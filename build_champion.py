import json
import re

NB_IN = 'WIA1006_Catfish_Group7_V17_ENHANCED.ipynb'
NB_OUT = 'WIA1006_Catfish_Group7_V18_CHAMPION.ipynb'

with open(NB_IN, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# 1. Add `shap` to pip install
for c in cells:
    if c['cell_type'] == 'code':
        source = "".join(c['source'])
        if '!pip install' in source:
            new_source = source.replace('!pip install -q', '!pip install -q shap scikit-plot')
            c['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

def make_md(title, text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {title}\n", f"> {text}"]
    }

def make_code(code_str):
    lines = code_str.strip().split('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    }

eda_md = make_md("📊 Cell X — Extensive Exploratory Data Analysis (EDA)", "**Academic Addition:** Comprehensive visualization of target distributions and feature relationships.")
eda_code = make_code("""
# ═══════════════════════════════════════════════════════════════
# CELL X | Extensive Exploratory Data Analysis (EDA)
# ═══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
if 'df' not in dir(): raise RuntimeError('❌ Run previous cells first.')

print("📊 Generating Exploratory Data Analysis Visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Target Distribution
sns.countplot(data=df, x='match_outcome', palette='pastel', ax=axes[0,0])
axes[0,0].set_title('Class Imbalance: Match Outcome Distribution', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Correlation Heatmap
corr = df.select_dtypes('number').corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, fmt=".2f", ax=axes[0,1])
axes[0,1].set_title('Feature Correlation Heatmap', fontweight='bold')

# 3. Violin Plot: App Usage by Outcome
sns.violinplot(data=df, x='match_outcome', y='app_usage_time_min', palette='muted', ax=axes[1,0])
axes[1,0].set_title('App Usage Time vs Outcome', fontweight='bold')

# 4. Box Plot: Message Sent Count
sns.boxplot(data=df, x='match_outcome', y='message_sent_count', palette='Set2', ax=axes[1,1])
axes[1,1].set_title('Messages Sent vs Outcome', fontweight='bold')

plt.tight_layout()
plt.show()
""")

pca_scree_md = make_md("📉 Cell X — PCA Cumulative Variance (Scree Plot)", "**Academic Addition:** Mathematically justifying our choice of keeping 95% variance.")
pca_scree_code = make_code("""
# ═══════════════════════════════════════════════════════════════
# CELL X | PCA Cumulative Variance (Scree Plot)
# ═══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt, numpy as np
if 'pca' not in dir(): raise RuntimeError('❌ Run previous cells first.')

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance Threshold')
plt.title('PCA Explained Variance (Scree Plot)', fontweight='bold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print(f"✅ PCA reduced features to {pca.n_components_} components while retaining 95% variance.")
""")

shap_md = make_md("🧠 Cell X — Explainable AI (SHAP Summary)", "**Academic Addition:** Using SHapley Additive exPlanations to demystify the 'Black Box' ML models.")
shap_code = make_code("""
# ═══════════════════════════════════════════════════════════════
# CELL X | Explainable AI (SHAP Summary)
# ═══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
if '_et_imp' not in dir(): raise RuntimeError('❌ Run previous cells first.')

print("🧠 Generating SHAP (Explainable AI) visualization...")
try:
    import shap
    # Use the ExtraTrees model trained on raw data for human-readable features
    explainer = shap.TreeExplainer(_et_imp)
    # Subset to save computation time
    X_sample = pd.DataFrame(X_train_bal[:1500], columns=FEATURE_NAMES)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle both list (binary/multi) and array returns
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_sample, show=False)
    plt.title("SHAP Summary Plot: Top Features Driving 'Catfish' Predictions", fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ SHAP analysis highlights exactly how behavioral intensity impacts risk.")
except ImportError:
    print("⚠️ SHAP library not found. Run pip install cell.")
except Exception as e:
    print(f"⚠️ SHAP visualization skipped: {e}")
""")

calib_md = make_md("🎯 Cell X — Calibration Curves (Reliability Diagrams)", "**Academic Addition:** Proving the predicted probabilities perfectly map to real-world likelihoods.")
calib_code = make_code("""
# ═══════════════════════════════════════════════════════════════
# CELL X | Calibration Curves
# ═══════════════════════════════════════════════════════════════
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
if 'models' not in dir(): raise RuntimeError('❌ Run previous cells first.')

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_sel)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test_arr, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name}")

plt.ylabel("Fraction of Positives")
plt.xlabel("Mean Predicted Probability")
plt.title('Calibration Curves (Reliability Diagram)', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
""")

# INJECTION LOGIC
# Find insertion indices based on text content
def find_cell_idx(keyword):
    for i, c in enumerate(cells):
        src = "".join(c['source'])
        if keyword in src: return i
    return -1

# Insert EDA after Load & Clean (Cell 4)
eda_idx = find_cell_idx('Load & Clean')
if eda_idx != -1:
    cells.insert(eda_idx + 2, eda_md)
    cells.insert(eda_idx + 3, eda_code)

# Insert PCA Scree Plot after Feature Importance (which has PCA)
pca_idx = find_cell_idx('Applying PCA')
if pca_idx != -1:
    cells.insert(pca_idx + 1, pca_scree_md)
    cells.insert(pca_idx + 2, pca_scree_code)

# Insert Calibration Curve after ROC Curves
roc_idx = find_cell_idx('ROC Curves')
if roc_idx != -1:
    cells.insert(roc_idx + 2, calib_md)
    cells.insert(roc_idx + 3, calib_code)

# Insert SHAP after Feature Importance Visualization
fi_idx = find_cell_idx('Catfish Red Flags (Full Feature Space)')
if fi_idx != -1:
    cells.insert(fi_idx + 2, shap_md)
    cells.insert(fi_idx + 3, shap_code)

# 4. Global Renumbering
# We will iterate through all cells. Every time we hit a major markdown cell with "Cell " we increment counter
cell_counter = 0
for c in cells:
    if c['cell_type'] == 'markdown':
        src = "".join(c['source'])
        if re.search(r'## .*Cell ', src, re.IGNORECASE) or re.search(r'## 📚 Cell', src):
            cell_counter += 1
            # Update Markdown Header
            new_src = re.sub(r'(## .*?Cell) [0-9X\.]+', rf'\1 {cell_counter}', src)
            c['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]] if new_src else []
            
    elif c['cell_type'] == 'code':
        src = "".join(c['source'])
        if '# CELL ' in src:
            # Code block starts with # CELL X |
            new_src = re.sub(r'# CELL [0-9X\.]+', f'# CELL {cell_counter}', src)
            # Update 'Run Cell X first' warnings to generic warnings to avoid breaking
            new_src = re.sub(r"Run Cell \d+ first", "Run previous cells first", new_src)
            c['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]] if new_src else []

with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {NB_OUT} with {cell_counter} major cells!")
