import json
import sys

nb_path = 'WIA1006_Catfish_Group7_Ultimate.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_lines = cell['source']
    source = ''.join(source_lines)
    
    # Update Cell 30/31 (12c): Random Forest -> GMM
    if cell['cell_type'] == 'markdown' and 'Cell 12c' in source:
        source = source.replace('Random Forest', 'Gaussian Mixture Model')
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
    
    if cell['cell_type'] == 'code' and 'CELL 12c |' in source:
        source = source.replace('Random Forest', 'Gaussian Mixture Model')
        source = source.replace("name = 'Gaussian Mixture Model'", "name = 'Gaussian Mixture Model'\n# GMM does not have feature importances")
        # Remove feature importance plotting from GMM cell
        import re
        source = re.sub(r'# --- Feature Importance.*?plt\.show\(\)', '', source, flags=re.DOTALL)
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    # Update Cell 32/33 (12d): Extra Trees -> SVM
    if cell['cell_type'] == 'markdown' and 'Cell 12d' in source:
        source = source.replace('Extra Trees', 'Support Vector Machine')
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    if cell['cell_type'] == 'code' and 'CELL 12d |' in source:
        source = source.replace('Extra Trees', 'Support Vector Machine')
        # Remove feature importance plotting from SVM cell
        import re
        source = re.sub(r'# --- Feature Importance.*?plt\.show\(\)', '', source, flags=re.DOTALL)
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    # Update Cell 34/35 (12e): XGBoost -> KMeans + PCA
    if cell['cell_type'] == 'markdown' and 'Cell 12e' in source:
        source = source.replace('XGBoost', 'KMeans + PCA')
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    if cell['cell_type'] == 'code' and 'CELL 12e |' in source:
        source = source.replace('XGBoost', 'KMeans + PCA')
        # Remove feature importance plotting from KMeans cell
        import re
        source = re.sub(r'# --- Feature Importance.*?plt\.show\(\)', '', source, flags=re.DOTALL)
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    # Update Cell 41: Learning Curves
    if cell['cell_type'] == 'code' and 'Plotting Learning Curves' in source:
        source = source.replace('Random Forest & XGBoost', 'Decision Tree & Support Vector Machine')
        source = source.replace("['Random Forest', 'XGBoost']", "['Decision Tree', 'Support Vector Machine']")
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]
        
    # Update Cell 55: Feature Importance dual plot
    if cell['cell_type'] == 'code' and 'CELL 20 | Feature Importance' in source:
        # Change title
        source = source.replace('Random Forest + XGBoost', 'Decision Tree + Logistic Regression')
        # Change loop items
        source = source.replace("[(axes[0],'Random Forest','magma'),(axes[1],'XGBoost','viridis')]", 
                                "[(axes[0],'Decision Tree','magma'),(axes[1],'Logistic Regression','viridis')]")
        
        # We need to handle Logistic Regression which uses coef_ instead of feature_importances_
        # I'll just write a custom block replacing the whole plotting logic for cell 55
        new_source = '''# ==============================================================================
# CELL 20 | Feature Importance — Decision Tree + Logistic Regression
# ==============================================================================
import matplotlib.pyplot as plt, seaborn as sns, numpy as np
if 'models' not in dir() or 'Decision Tree' not in models: raise RuntimeError('❌ Train models first.')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree Importances
dt_model = models['Decision Tree']
dt_imps = dt_model.feature_importances_ if hasattr(dt_model, 'feature_importances_') else dt_model.estimator.feature_importances_
dt_imp_dict = dict(zip(SELECTED_FEATURES, dt_imps))
dt_sorted = sorted(dt_imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
sns.barplot(x=[v for k,v in dt_sorted], y=[k for k,v in dt_sorted], palette='magma', ax=axes[0])
axes[0].set_title('Decision Tree - Top 10 Feature Importances', fontweight='bold')
axes[0].set_xlabel('Gini Importance')

# Logistic Regression Coefficients
lr_model = models['Logistic Regression']
lr_coefs = lr_model.coef_[0] if hasattr(lr_model, 'coef_') else lr_model.estimator.coef_[0]
lr_imp_dict = dict(zip(SELECTED_FEATURES, np.abs(lr_coefs)))
lr_sorted = sorted(lr_imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
sns.barplot(x=[v for k,v in lr_sorted], y=[k for k,v in lr_sorted], palette='viridis', ax=axes[1])
axes[1].set_title('Logistic Regression - Top 10 Absolute Coefficients', fontweight='bold')
axes[1].set_xlabel('Absolute Coefficient Value')

plt.tight_layout()
plt.show()

print('\\n🚩 Top 10 Red Flags (Decision Tree):')
for k,v in dt_sorted: print(f'   {k:25s}: {v:.4f}')
'''
        cell['source'] = [s + '\n' for s in new_source.split('\n')[:-1]]
        
    # Update Cell 58: SHAP
    if cell['cell_type'] == 'code' and 'shap.TreeExplainer' in source:
        source = source.replace("models['Random Forest']", "models['Decision Tree']")
        # if the model is wrapped in ModelFunction, we might need to access `.estimator`
        source = source.replace("models['Decision Tree']", "models['Decision Tree'].estimator if hasattr(models['Decision Tree'], 'estimator') else models['Decision Tree']")
        # Also fix the subtitle print
        source = source.replace("SHAP Waterfall (Random Forest)", "SHAP Waterfall (Decision Tree)")
        cell['source'] = [s + '\n' for s in source.split('\n')[:-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully applied rigorous notebook updates.')
