import sys

path = 'catfish_core.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove KMeans from base_models_smote
content = content.replace('        "Gaussian Mixture Model": GMMClassifier(random_state=42),\n        "KMeans": KMeansClassifier(random_state=42),\n    }', '        "Gaussian Mixture Model": GMMClassifier(random_state=42),\n    }')

# 2. Add KMeans to base_models_orig
content = content.replace('        "Support Vector Machine": SVC(\n            probability=True, class_weight="balanced", random_state=42,\n            max_iter=5000, tol=1e-3, cache_size=2000,\n        ),\n    }', '        "Support Vector Machine": SVC(\n            probability=True, class_weight="balanced", random_state=42,\n            max_iter=5000, tol=1e-3, cache_size=2000,\n        ),\n        "KMeans": KMeansClassifier(random_state=42),\n    }')

# 3. Remove KMeans from param_grids_smote
content = content.replace('        "Gaussian Mixture Model": {\n            "n_components": [3, 4, 5, 6],\n            "covariance_type": ["diag", "full"],\n            "reg_covar": [1e-5, 1e-4, 1e-3]\n        },\n        "KMeans": {\n            "n_clusters": [30, 40, 50],\n            "refine_iters": [15, 20],\n            "temperature": [0.1, 0.25, 0.5]\n        },\n    }', '        "Gaussian Mixture Model": {\n            "n_components": [3, 4, 5, 6],\n            "covariance_type": ["diag", "full"],\n            "reg_covar": [1e-5, 1e-4, 1e-3]\n        }\n    }')

# 4. Add KMeans to param_grids_orig
content = content.replace('        "Support Vector Machine": {"C": [0.1, 0.5, 1, 5, 10], "gamma": ["scale", "auto"]}\n    }', '        "Support Vector Machine": {"C": [0.1, 0.5, 1, 5, 10], "gamma": ["scale", "auto"]},\n        "KMeans": {"n_clusters": [30, 40, 50], "refine_iters": [15, 20], "temperature": [0.1, 0.25, 0.5]},\n    }')

# 5. Add to sample_map_orig
content = content.replace('    sample_map_orig = {\n        "Support Vector Machine": 8000,\n    }', '    sample_map_orig = {\n        "Support Vector Machine": 8000,\n        "KMeans": 12000,\n    }')

# 6. Add to n_iter_map_orig
content = content.replace('    n_iter_map_orig = {\n        "Support Vector Machine": 8,\n    }', '    n_iter_map_orig = {\n        "Support Vector Machine": 8,\n        "KMeans": 8,\n    }')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
