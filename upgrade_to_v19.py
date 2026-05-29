
import json
import re

with open("WIA1006_Catfish_Group7_V18_CHAMPION.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Title
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "V18_CHAMPION" in src or "V18" in src or "V15.0" in src:
            src = src.replace("V18_CHAMPION", "V19_CHAMPION").replace("V15.0", "V19.0")
            src = src.replace("V18", "V19")
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]] if src else []
            
        if "V15" in src:
            src = src.replace("V15", "V19")
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]] if src else []

# We need to update Cell 12 (Training) to NOT use PCA, but use X_train_bal (the raw features).
# In V18, training was done on X_train_sel (which was PCA). We just replace X_train_sel with X_train_bal, 
# and X_test_sel with X_test_arr everywhere downstream.

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        
        # Replace PCA usage in training and evaluation with raw balanced data
        if "X_train_sel" in src:
            src = src.replace("X_train_sel", "X_train_bal")
        if "X_test_sel" in src:
            src = src.replace("X_test_sel", "X_test_arr")
            
        # Update param grids in Cell 12 for better performance
        if "param_grids = {" in src and "RandomizedSearchCV" in src:
            # We can also just update the entire cell 12 if we want, but simple replace works.
            # Let us replace the y_train_sel with y_train_bal
            src = src.replace("y_train_sel", "y_train_bal")
        
        if "y_train_sel" in src:
            src = src.replace("y_train_sel", "y_train_bal")
            
        # Cell 21 and 23 Scanner updates
        if "pca.transform" in src:
            # Remove PCA transform from scanner input
            src = src.replace("return pca.transform(input_df.values)", "return input_df.values.astype(np.float64)")
            
        cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]] if src else []

nb["metadata"]["colab"]["name"] = "WIA1006_Catfish_Group7_V19_CHAMPION.ipynb"

with open("WIA1006_Catfish_Group7_V19_CHAMPION_Full.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("V19 Full Generated!")

