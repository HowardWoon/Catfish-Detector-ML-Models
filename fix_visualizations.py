import json

def patch():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if any('ML Pipeline Initialized.' in line for line in source):
                # This is Cell 12
                # Ensure plot_prob_scatter_2d is defined
                if not any('def plot_prob_scatter_2d' in line for line in source):
                    source.append('\n')
                    source.append('def plot_prob_scatter_2d(X, y, probs, title):\n')
                    source.append('    import matplotlib.pyplot as plt\n')
                    source.append('    from sklearn.decomposition import PCA\n')
                    source.append('    pca = PCA(n_components=2, random_state=42)\n')
                    source.append('    X_pca = pca.fit_transform(X)\n')
                    source.append('    plt.figure(figsize=(10, 6))\n')
                    source.append('    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=probs, cmap="coolwarm", alpha=0.7, edgecolor="k")\n')
                    source.append('    plt.colorbar(scatter, label="Predicted Probability (Catfish)")\n')
                    source.append('    plt.title(f"{title} - Probability Distribution (PCA Reduced)", fontweight="bold")\n')
                    source.append('    plt.xlabel("PCA Component 1")\n')
                    source.append('    plt.ylabel("PCA Component 2")\n')
                    source.append('    plt.grid(True, linestyle="--", alpha=0.3)\n')
                    source.append('    plt.show()\n')
                break

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("plot_prob_scatter_2d function appended to Cell 12.")

if __name__ == '__main__':
    patch()
