import json

def fix_syntax():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if 'Run **12' in line and not line.strip().startswith('#'):
                    line = '# ' + line
                new_source.append(line)
            cell['source'] = new_source

    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
if __name__ == '__main__':
    fix_syntax()
