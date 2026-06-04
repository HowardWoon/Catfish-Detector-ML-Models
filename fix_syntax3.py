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
                # Replace literal backslash+newline with just newline
                line = line.replace('\\\n', '\n')
                # If there's an isolated backslash at the end
                if line.endswith('\\\n'):
                    line = line[:-2] + '\n'
                if line.strip() == '\\':
                    line = '\n'
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Syntax error fixed.")

if __name__ == '__main__':
    patch()
