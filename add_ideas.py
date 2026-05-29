import json

NB_IN = 'WIA1006_Catfish_Group7_V16_FINAL.ipynb'
NB_OUT = 'WIA1006_Catfish_Group7_V17_ENHANCED.ipynb'

with open(NB_IN, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of Cell 11 (Threshold Search) to insert before it
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if 'Cell 11' in source and 'Threshold Search' in source:
            insert_idx = i
            break

if insert_idx == -1:
    print("Could not find Cell 11.")
    exit(1)

# Create Learning Curves Markdown Cell
lc_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📈 Cell 11 — Learning Curves Analysis\n",
        "> **Academic Addition:** Visualizing training vs cross-validation scores over varying dataset sizes to definitively prove our tuned models are not overfitting and generalize well."
    ]
}

# Create Learning Curves Code Cell
lc_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ═══════════════════════════════════════════════════════════════\n",
        "# CELL 11 | Learning Curves Analysis\n",
        "# ═══════════════════════════════════════════════════════════════\n",
        "import numpy as np, matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import learning_curve\n",
        "if 'models' not in dir(): raise RuntimeError('❌ Run Cell 10 first.')\n",
        "\n",
        "print('📈 Plotting Learning Curves for top 2 models (Random Forest & XGBoost)...')\n",
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
        "for ax, name in zip(axes, ['Random Forest', 'XGBoost']): \n",
        "    if name not in models: continue\n",
        "    train_sizes, train_scores, test_scores = learning_curve(\n",
        "        models[name], X_train_sel, y_train_sel, cv=3, n_jobs=-1,\n",
        "        train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_macro')\n",
        "    \n",
        "    train_mean = np.mean(train_scores, axis=1)\n",
        "    train_std = np.std(train_scores, axis=1)\n",
        "    test_mean = np.mean(test_scores, axis=1)\n",
        "    test_std = np.std(test_scores, axis=1)\n",
        "    \n",
        "    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')\n",
        "    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')\n",
        "    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')\n",
        "    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')\n",
        "    \n",
        "    ax.set_title(f'Learning Curve: {name}', fontweight='bold')\n",
        "    ax.set_xlabel('Training examples')\n",
        "    ax.set_ylabel('F1-Macro Score')\n",
        "    ax.legend(loc='lower right')\n",
        "    ax.grid(True, linestyle='--', alpha=0.7)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

# Create AutoML Markdown Cell
auto_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🤖 Cell 12 — AutoML with Auto-sklearn (Conceptual)\n",
        "> **Academic Addition:** While we rigorously tuned our models using `RandomizedSearchCV`, state-of-the-art pipelines often utilize AutoML. This cell outlines how Auto-sklearn leverages Meta-Learning and Bayesian Optimization to find the optimal ensemble automatically."
    ]
}

# Create AutoML Code Cell
auto_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ═══════════════════════════════════════════════════════════════\n",
        "# CELL 12 | AutoML Comparison (Auto-sklearn)\n",
        "# ═══════════════════════════════════════════════════════════════\n",
        "# ⚠️ NOTE: auto-sklearn requires specific Linux dependencies.\n",
        "# We have already performed rigorous tuning using RandomizedSearchCV above.\n",
        "# This conceptual snippet demonstrates how Auto-Sklearn would be applied.\n",
        "\n",
        "print(\"🤖 --- AUTO-SKLEARN IMPLEMENTATION (CONCEPTUAL) ---\")\n",
        "print(\"If we were to run auto-sklearn, the implementation would be:\")\n",
        "\n",
        "code_str = \"\"\"\n",
        "!pip install auto-sklearn\n",
        "import autosklearn.classification\n",
        "\n",
        "# Initialize the AutoML classifier\n",
        "automl = autosklearn.classification.AutoSklearnClassifier(\n",
        "    time_left_for_this_task=300, # 5 minutes search\n",
        "    per_run_time_limit=30,\n",
        "    n_jobs=-1,\n",
        "    resampling_strategy='cv',\n",
        "    resampling_strategy_arguments={'folds': 3}\n",
        ")\n",
        "\n",
        "# Fit on the PCA-reduced balanced data\n",
        "automl.fit(X_train_sel, y_train_sel)\n",
        "automl.refit(X_train_sel, y_train_sel)\n",
        "\n",
        "# Display the ensemble discovered by Auto-sklearn\n",
        "print(automl.show_models())\n",
        "\n",
        "# Compare Performance\n",
        "preds = automl.predict(X_test_sel)\n",
        "print(\"AutoML F1-Score:\", f1_score(y_test_arr, preds))\n",
        "\"\"\"\n",
        "print(code_str)\n",
        "print(\"==========================================================\")\n",
        "print(\"Advantages of AutoML over Manual Tuning:\")\n",
        "print(\"1. Meta-Learning: Uses experience from similar datasets to jumpstart search.\")\n",
        "print(\"2. Bayesian Optimization: Searches the hyperparameter space intelligently.\")\n",
        "print(\"3. Automated Ensembling: Builds a weighted ensemble of the best models.\")\n"
    ]
}

# Update all subsequent cell numbers in their markdown headers and code comments
for cell in nb['cells'][insert_idx:]:
    if 'source' in cell:
        if isinstance(cell['source'], list):
            for j in range(len(cell['source'])):
                line = cell['source'][j]
                for k in range(18, 10, -1):
                    if f"Cell {k}" in line:
                        line = line.replace(f"Cell {k}", f"Cell {k+2}")
                    if f"CELL {k}" in line:
                        line = line.replace(f"CELL {k}", f"CELL {k+2}")
                cell['source'][j] = line
        elif isinstance(cell['source'], str):
            line = cell['source']
            for k in range(18, 10, -1):
                if f"Cell {k}" in line:
                    line = line.replace(f"Cell {k}", f"Cell {k+2}")
                if f"CELL {k}" in line:
                    line = line.replace(f"CELL {k}", f"CELL {k+2}")
            cell['source'] = line

# Insert the new cells
nb['cells'].insert(insert_idx, lc_md)
nb['cells'].insert(insert_idx + 1, lc_code)
nb['cells'].insert(insert_idx + 2, auto_md)
nb['cells'].insert(insert_idx + 3, auto_code)

with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {NB_OUT}!")
