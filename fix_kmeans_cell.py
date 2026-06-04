import json

def fix_kmeans():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if any('Tuning KMeans' in line for line in source):
                new_source = []
                for line in source:
                    if 'tune_model_orig' in line:
                        line = line.replace('tune_model_orig', 'tune_model')
                    if 'X_test_arr' in line and not 'probs_te = ' in line: # Only replace X_test if needed, wait, tune_model doesn't use X_test inside the cell.
                        pass
                    new_source.append(line)
                cell['source'] = new_source

    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    fix_kmeans()
