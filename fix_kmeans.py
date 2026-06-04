import json

def fix_kmeans():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = "".join(cell.get('source', []))
            if "class KMeansClassifier(BaseEstimator, ClassifierMixin):" in src:
                # Add _estimator_type
                fixed_src = src.replace(
                    "class KMeansClassifier(BaseEstimator, ClassifierMixin):",
                    "class KMeansClassifier(BaseEstimator, ClassifierMixin):\n  _estimator_type = \"classifier\""
                )
                # Apply changes to cell source
                # reconstruct lines
                lines = []
                for line in fixed_src.split('\n'):
                    if line == '': continue
                    lines.append(line + '\n')
                cell['source'] = lines
                
                # To be safe, remove double newlines
                cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
                print("Fixed KMeansClassifier")
                
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_kmeans()
