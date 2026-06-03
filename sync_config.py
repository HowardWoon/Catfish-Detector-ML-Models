import json
path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'n_clusters' in line and '[3, 4, 5]' in line and 'KMeans' in ''.join(cell['source']):
                cell['source'][i] = line.replace('[3, 4, 5]', '[30, 40, 50]')
            if 'hidden_layer_sizes' in line and 'MLP Neural Network' in ''.join(cell['source']):
                cell['source'][i] = "  'MLP Neural Network': {'hidden_layer_sizes': [(128, 64), (64, 32), (256, 128, 64)], 'alpha': [0.0005, 0.001, 0.005]},\n"
            if 'Logistic Regression' in line and 'C' in line and 'penalty' in line and 'param_grids =' not in line:
                cell['source'][i] = "  'Logistic Regression': {'C': [0.5, 1, 2], 'penalty': ['l2']},\n"
            if 'Decision Tree' in line and 'max_depth' in line and 'param_grids =' not in line:
                cell['source'][i] = "  'Decision Tree': {'max_depth': [8, 12, 16], 'min_samples_split': [5, 10]},\n"

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
