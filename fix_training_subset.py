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
                # Ensure training_subset is defined
                if not any('def training_subset' in line for line in source):
                    # We will append the missing helper functions
                    source.append('\n')
                    source.append('def training_subset(n):\n')
                    source.append('    idx = np.random.choice(len(X_train_bal), size=min(n, len(X_train_bal)), replace=False)\n')
                    source.append('    x_ret = X_train_bal.iloc[idx] if hasattr(X_train_bal, "iloc") else X_train_bal[idx]\n')
                    source.append('    y_ret = y_train_bal.iloc[idx] if hasattr(y_train_bal, "iloc") else y_train_bal[idx]\n')
                    source.append('    return x_ret, y_ret\n')
                    source.append('\n')
                    source.append('def training_subset_orig(n):\n')
                    source.append('    idx = np.random.choice(len(X_train_arr), size=min(n, len(X_train_arr)), replace=False)\n')
                    source.append('    x_ret = X_train_arr[idx]\n')
                    source.append('    y_ret = y_train_arr[idx]\n')
                    source.append('    return x_ret, y_ret\n')
                break

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("training_subset functions appended to Cell 12.")

if __name__ == '__main__':
    patch()
