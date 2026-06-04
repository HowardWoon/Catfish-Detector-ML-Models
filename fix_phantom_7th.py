import json

def patch():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                # Remove all references to KMeans + PCA
                if "'KMeans + PCA': Pipeline" in line:
                    continue
                if '"KMeans + PCA": {"kmeans__n_clusters":' in line:
                    continue
                if "'KMeans + PCA'" in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Phantom 7th Model 'KMeans + PCA' eradicated from code cells.")

if __name__ == '__main__':
    patch()
