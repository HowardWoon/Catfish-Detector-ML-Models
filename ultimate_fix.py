import json

def patch_notebook():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        # 1. Patch Cell 12 (Pipeline Initialization)
        if 'def tune_model(' in source and 'base_models =' in source:
            print("Found Cell 12, patching...")
            new_source = []
            for line in cell['source']:
                # Update GMM to Pipeline
                if "'Gaussian Mixture Model': GMMClassifier(random_state=RANDOM_STATE)," in line:
                    line = "    'Gaussian Mixture Model': Pipeline([('pca', PCA(n_components=0.95, random_state=RANDOM_STATE)), ('gmm', GMMClassifier(random_state=RANDOM_STATE))]),\n"
                
                # Delete KMeans + PCA entirely
                if "'KMeans + PCA': Pipeline" in line:
                    continue
                
                # Update SVM param grid
                if '"Support Vector Machine": {"C": [10], "gamma": ["auto"]},' in line:
                    line = '    "Support Vector Machine": {"C": [1, 10, 50], "gamma": ["scale", "auto"]},\n'
                    
                # Update GMM param grid
                if '"Gaussian Mixture Model": {"n_components": [2, 3], "covariance_type": ["full", "tied"]},' in line:
                    line = '    "Gaussian Mixture Model": {"gmm__n_components": [2, 3, 4], "gmm__covariance_type": ["diag", "full"], "gmm__reg_covar": [1e-4, 1e-3]},\n'
                    
                # Update KMeans param grid
                if '"KMeans": {"n_clusters": [30, 40, 50]}' in line:
                    line = '    "KMeans": {"n_clusters": [20, 30, 40], "refine_iters": [10, 15], "temperature": [0.1, 0.25, 0.5]}\n'
                
                new_source.append(line)
            cell['source'] = new_source
            
        # 2. Patch Cell 12c (GMM)
        if "name = 'Gaussian Mixture Model'" in source:
            print("Found Cell 12c (GMM), patching...")
            new_source = []
            for line in cell['source']:
                if 'gmm = models[name]' in line:
                    new_source.append(line)
                    new_source.append("  gmm_model = gmm.named_steps['gmm']\n")
                    continue
                if 'print(f"  Genuine components: {gmm.n_components' in line:
                    line = line.replace('gmm.n_components', 'gmm_model.n_components')
                if 'print(f"  Covariance: {gmm.covariance_type' in line:
                    line = line.replace('gmm.covariance_type', 'gmm_model.covariance_type')
                    line = line.replace('gmm.reg_covar', 'gmm_model.reg_covar')
                new_source.append(line)
            cell['source'] = new_source
            
        # 3. Patch Cell 12e (KMeans)
        if "name = 'KMeans'" in source:
            print("Found Cell 12e (KMeans), patching...")
            new_source = []
            for line in cell['source']:
                if "tune_model_orig(name)" in line:
                    line = line.replace("tune_model_orig(name)", "tune_model(name)")
                new_source.append(line)
            cell['source'] = new_source
            
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    patch_notebook()
