import json
import re

def sync():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        if cell_type == 'code':
            new_source = []
            for line in source:
                # Fix outdated selector references
                if 'selector.transform()' in line:
                    line = line.replace('then selector.transform()', 'and no feature dropping is needed')
                    line = line.replace('selector.transform() sees the correct updated values.', 'All 33 features are safely passed to the models.')
                # Fix V14 FIX label
                if 'V14 FIX:' in line:
                    line = line.replace('V14 FIX:', 'ULTIMATE FIX:')
                # Add comment for 100% scanner bug
                if 'def behavioral_risk(' in line:
                    new_source.append("    # ==============================================================================\n")
                    new_source.append("    # CRITICAL FIX: Added EPS (1e-6) to all denominators to prevent Division by Zero.\n")
                    new_source.append("    # This permanently eliminates the '100% Catfish Bug' where extreme inputs produced NaNs.\n")
                    new_source.append("    # ==============================================================================\n")
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook Comments Synced Successfully.")

if __name__ == '__main__':
    sync()
