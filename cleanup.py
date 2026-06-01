import json
import re

nb_path = 'WIA1006_Catfish_Group7_Ultimate.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            line = line.replace('xgboost', '')
            line = line.replace('RANDOM FOREST', 'GAUSSIAN MIXTURE MODEL')
            line = line.replace('XGBOOST', 'KMEANS + PCA')
            line = line.replace('EXTRA TREES', 'SUPPORT VECTOR MACHINE')
            line = line.replace('preliminary Random Forest', 'preliminary Decision Tree')
            line = line.replace('representative model (Random Forest)', 'representative models')
            line = line.replace('Similar to Random Forest', 'Similar to Decision Tree')
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cleaned Notebook')
