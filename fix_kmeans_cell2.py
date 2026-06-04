import json

def fix_kmeans():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if "name = 'KMeans'" in source and "tune_model_orig(name)" in source:
                new_source = source.replace("tune_model_orig(name)", "tune_model(name)")
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                print("KMeans cell patched successfully!")

    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    fix_kmeans()
