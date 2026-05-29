
import json

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        
        # Cell 10: Fix SELECTED_FEATURES and completely comment out/remove the PCA application
        if "SELECTED_FEATURES" in src and "Applying PCA" in src:
            # We want to replace the whole PCA block and fix the print statement
            new_src = src.replace("print(f'? Features in model: {len(SELECTED_FEATURES)} (all kept — no selector drop)')", "SELECTED_FEATURES = FEATURE_NAMES\\nprint(f'? Features in model: {len(SELECTED_FEATURES)} (all kept — no selector drop)')")
            
            # Disable PCA transform block
            pca_block_old = """# Apply PCA
print('?? Applying PCA...')
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_arr)
X_test_pca = pca.transform(X_test_arr)

# Re-balance PCA data
print('?? Balancing PCA data...')
X_train_bal, y_train_bal = smt.fit_resample(X_train_pca, y_train_arr)
X_test_arr = X_test_pca"""

            pca_block_new = """# Apply PCA (DISABLED IN V19 CHAMPION - WE USE ALL 51 FEATURES)
# print('?? Applying PCA...')
# pca = PCA(n_components=0.95, random_state=42)
# X_train_pca = pca.fit_transform(X_train_arr)
# X_test_pca = pca.transform(X_test_arr)

# Re-balance PCA data (DISABLED)
# print('?? Balancing PCA data...')
# X_train_bal, y_train_bal = smt.fit_resample(X_train_pca, y_train_arr)
# X_test_arr = X_test_pca
"""
            new_src = new_src.replace(pca_block_old, pca_block_new)
            
            cell["source"] = [line + "\\n" for line in new_src.split("\\n")[:-1]] + [new_src.split("\\n")[-1]]
            
        # In Cell 23 (Scanner test) where it used to use pca.transform
        if "pca.transform" in src:
            src = src.replace("pca.transform", "lambda x: x")
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]]
            
        # In Cell 11 Scree Plot - Since pca is disabled above, this will throw an error if we run it
        # But this is just a markdown+code cell for Scree plot. We can just add a dummy pca object 
        # so it still plots the theoretical scree plot for academic purposes.
        if "plt.plot(np.cumsum(pca.explained_variance_ratio_)" in src:
            dummy_pca = """# To demonstrate the academic scree plot without breaking the main pipeline, we instantiate a standalone PCA here:
from sklearn.decomposition import PCA
pca = PCA(random_state=42)
pca.fit(X_train_arr)
"""
            if "dummy_pca" not in src:
                src = dummy_pca + src
                cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]]
            
        # SHAP Cell
        if "shap.TreeExplainer(_et_imp)" in src:
            src = src.replace("shap.TreeExplainer(_et_imp)", "shap.TreeExplainer(models['Random Forest'])")
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]]
            
        # Ensure that no code calls X_test_sel if it has not been replaced
        if "X_test_sel" in src:
            src = src.replace("X_test_sel", "X_test_arr")
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]]

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook thoroughly patched for V19 bugs!")

