import json
import re

def fix_gmm_cluster_mapping():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = "".join(cell.get('source', []))
            if "name = 'Gaussian Mixture Model'" in src and "cluster_mapping_" in src:
                # Remove the line containing cluster_mapping_
                fixed_lines = [line for line in src.split('\n') if 'cluster_mapping_' not in line]
                fixed_src = '\n'.join(fixed_lines)
                
                cell['source'] = [line + '\n' for line in fixed_src.split('\n') if line != '']
                cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
                print("Removed invalid cluster_mapping_ reference in Cell 12c")
                
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_gmm_cluster_mapping()
