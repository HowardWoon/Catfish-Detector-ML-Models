import json
import re

def sync():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        # 1. Fix Markdown Descriptions
        if cell_type == 'markdown':
            new_source = []
            for line in source:
                # Replace 7 models with 6 models
                line = line.replace('7 Machine Learning algorithms', '6 Machine Learning algorithms')
                line = line.replace('7 models', '6 models')
                # Remove KMeans + PCA references
                if 'KMeans + PCA' in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source
            
        # 2. Fix Cell 12 Python Code (base_models and param_grids)
        if cell_type == 'code':
            src_str = "".join(source)
            if 'base_models = {' in src_str and 'param_grids = {' in src_str:
                new_source = []
                skip = False
                for line in source:
                    if line.startswith('base_models = {'):
                        skip = True
                        new_source.append("base_models = {\n")
                        new_source.append("    # UPDATED: Matches backend models for absolute best performance.\n")
                        new_source.append("    # Logistic Regression uses liblinear solver for robust small-dataset convergence.\n")
                        new_source.append("    'Logistic Regression': LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE),\n")
                        new_source.append("    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_features='sqrt', random_state=RANDOM_STATE),\n")
                        new_source.append("    # GMM has tuned n_components and covariance.\n")
                        new_source.append("    'Gaussian Mixture Model': GMMClassifier(random_state=RANDOM_STATE),\n")
                        new_source.append("    'MLP Neural Network': MLPClassifier(early_stopping=True, max_iter=200, random_state=RANDOM_STATE),\n")
                        new_source.append("    # SVM uses max_iter=5000 to completely prevent infinite hanging.\n")
                        new_source.append("    'Support Vector Machine': SVC(max_iter=5000, probability=True, class_weight='balanced', random_state=RANDOM_STATE, cache_size=2000, tol=1e-3),\n")
                        new_source.append("    # KMeans is our custom Label-Aware KMeans class.\n")
                        new_source.append("    'KMeans': KMeansClassifier(random_state=RANDOM_STATE)\n")
                        new_source.append("}\n\n")
                        
                        new_source.append("param_grids = {\n")
                        new_source.append("    # SMOTE-Trained Grids (These models benefit from 50/50 synthetic balance)\n")
                        new_source.append("    \"Logistic Regression\": {\"C\": [0.5, 1, 2], \"penalty\": [\"l2\"]},\n")
                        new_source.append("    \"Decision Tree\": {\"max_depth\": [8, 12, 16], \"min_samples_split\": [5, 10]},\n")
                        new_source.append("    \"Gaussian Mixture Model\": {\"n_components\": [3, 4, 5, 6], \"covariance_type\": [\"diag\", \"full\"], \"reg_covar\": [1e-5, 1e-4, 1e-3]},\n")
                        new_source.append("    \"MLP Neural Network\": {\"hidden_layer_sizes\": [(128, 64), (64, 32), (256, 128, 64)], \"alpha\": [0.0005, 0.001, 0.005]}\n")
                        new_source.append("}\n\n")
                        
                        new_source.append("param_grids_orig = {\n")
                        new_source.append("    # ORIGINAL-Trained Grids (These models require authentic imbalanced distribution)\n")
                        new_source.append("    \"Support Vector Machine\": {\"C\": [10], \"gamma\": [\"auto\"], \"kernel\": [\"rbf\"]},\n")
                        new_source.append("    \"KMeans\": {\"n_clusters\": [30, 40, 50], \"refine_iters\": [15, 20], \"temperature\": [0.1, 0.25, 0.5]}\n")
                        new_source.append("}\n")
                    
                    if skip:
                        if line.startswith('cv = StratifiedKFold'):
                            skip = False
                            
                    if not skip:
                        new_source.append(line)
                cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook Synced Successfully.")

if __name__ == '__main__':
    sync()
