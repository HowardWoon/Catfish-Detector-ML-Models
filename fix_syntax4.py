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
                # Replace literal backslash + 'n' at the very end of the line
                if line.endswith('\\n'):
                    line = line[:-2] + '\n'
                elif line.endswith('\\n\n'):
                    line = line[:-3] + '\n'
                if line.strip() == '\\':
                    continue
                if line.strip() == '\\n':
                    continue
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Syntax error fixed definitively.")

if __name__ == '__main__':
    patch()
