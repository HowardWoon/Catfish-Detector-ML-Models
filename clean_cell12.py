import json

def clean_cell12():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if 'ML Pipeline Initialized.' in source and '--- CELL' in source:
                # Find where the garbage starts
                garbage_idx = source.find('--- CELL')
                if garbage_idx != -1:
                    clean_source = source[:garbage_idx]
                    
                    # Ensure plot_prob_scatter_2d is in it
                    if 'def plot_prob_scatter_2d' not in clean_source:
                        plot_fn = """
def plot_prob_scatter_2d(X, y, probs, title):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=probs, cmap="coolwarm", alpha=0.7, edgecolor="k")
    plt.colorbar(scatter, label="Predicted Probability (Catfish)")
    plt.title(f"{title} - Probability Distribution (PCA Reduced)", fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()
"""
                        clean_source += plot_fn
                        
                    cell['source'] = [line + '\n' for line in clean_source.split('\n')]
                    print("Cell 12 successfully cleaned!")

    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    clean_cell12()
