import json
import re

def apply_patches():
    with open('executed_notebook.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        source = "".join(cell.get('source', []))
        
        # 0. Import GridSearchCV
        if "from sklearn.model_selection import RandomizedSearchCV" in source:
            new_source = []
            for line in cell['source']:
                if "from sklearn.model_selection import RandomizedSearchCV" in line:
                    line = line.replace("RandomizedSearchCV", "RandomizedSearchCV, GridSearchCV")
                new_source.append(line)
            cell['source'] = new_source

        # 1. Patch Cell 12: ML PIPELINE INITIALIZATION
        if 'base_models = {' in source and "'KMeans + PCA': Pipeline" in source:
            new_source = []
            for line in cell['source']:
                if "'Gaussian Mixture Model': GMMClassifier(random_state=42)," in line:
                    line = "    'Gaussian Mixture Model': Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('gmm', GMMClassifier(random_state=42))]),\n"
                
                if "'KMeans + PCA': Pipeline([('pca', PCA(n_components=0.95, random_state=42)), ('kmeans', KMeansClassifier(random_state=42))])," in line:
                    line = "    'KMeans + PCA': KMeansClassifier(random_state=42),\n"
                    
                if '"Gaussian Mixture Model": {"n_components"' in line:
                    line = '    "Gaussian Mixture Model": {"gmm__n_components": [2, 3, 4], "gmm__covariance_type": ["diag", "full"], "gmm__reg_covar": [1e-4, 1e-3]},\n'
                    
                if '"Support Vector Machine": {"C"' in line:
                    line = '    "Support Vector Machine": {"C": [1, 10, 50], "gamma": ["scale", "auto"]},\n'
                    
                if '"KMeans + PCA": {"kmeans__n_clusters"' in line:
                    line = '    "KMeans + PCA": {"n_clusters": [20, 30, 40], "refine_iters": [10, 15], "temperature": [0.1, 0.25, 0.5]},\n'
                    
                new_source.append(line)
            cell['source'] = new_source
            
        # 2. Patch Cell 12c: GMM
        if "name = 'Gaussian Mixture Model'" in source:
            new_source = []
            for line in cell['source']:
                if "rs = RandomizedSearchCV(" in line:
                    line = re.sub(r"rs\s*=\s*RandomizedSearchCV\((.*?),\s*n_iter=\d+,(.*?),\s*random_state=\d+(.*?)\)", r"gs = GridSearchCV(\1,\2\3)", line)
                if "rs.fit(x_fast, y_fast)" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_estimator_" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_params_" in line:
                    line = line.replace("rs.", "gs.")
                
                # Fix plotting attribute errors
                if "gmm = models[name]" in line:
                    new_source.append(line)
                    new_source.append("  gmm_model = gmm.named_steps['gmm']\n")
                    continue
                if 'gmm.n_components' in line:
                    line = line.replace('gmm.n_components', 'gmm_model.n_components')
                if 'gmm.covariance_type' in line:
                    line = line.replace('gmm.covariance_type', 'gmm_model.covariance_type')
                if 'gmm.reg_covar' in line:
                    line = line.replace('gmm.reg_covar', 'gmm_model.reg_covar')
                new_source.append(line)
            cell['source'] = new_source
            
        # 3. Patch Cell 12d: SVM
        if "name = 'Support Vector Machine'" in source:
            new_source = []
            for line in cell['source']:
                if "rs = RandomizedSearchCV(" in line:
                    line = re.sub(r"rs\s*=\s*RandomizedSearchCV\((.*?),\s*n_iter=\d+,(.*?),\s*random_state=\d+(.*?)\)", r"gs = GridSearchCV(\1,\2\3)", line)
                if "rs.fit(x_fast, y_fast)" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_estimator_" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_params_" in line:
                    line = line.replace("rs.", "gs.")
                new_source.append(line)
            cell['source'] = new_source
            
        # 4. Patch Cell 12e: KMeans
        if "name = 'KMeans + PCA'" in source:
            new_source = []
            for line in cell['source']:
                if "rs = RandomizedSearchCV(" in line:
                    line = re.sub(r"rs\s*=\s*RandomizedSearchCV\((.*?),\s*n_iter=\d+,(.*?),\s*random_state=\d+(.*?)\)", r"gs = GridSearchCV(\1,\2\3)", line)
                if "rs.fit(x_fast, y_fast)" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_estimator_" in line:
                    line = line.replace("rs.", "gs.")
                if "rs.best_params_" in line:
                    line = line.replace("rs.", "gs.")
                    
                new_source.append(line)
            cell['source'] = new_source
            
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook perfectly patched!")

if __name__ == '__main__':
    apply_patches()
