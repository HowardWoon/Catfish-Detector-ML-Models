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
                # Remove ANY line that starts with a backslash and contains only whitespace or 'n'
                clean_line = line.strip()
                if clean_line == '\\' or clean_line == '\\n' or clean_line == '\\\n':
                    continue
                # Also strip any trailing backslash if it's the very last character of a line
                if line.endswith('\\\n'):
                    line = line[:-2] + '\n'
                new_source.append(line)
                
            # If the last line is literally just '\n' (which is valid empty line but sometimes problematic)
            while new_source and not new_source[-1].strip():
                new_source.pop()
                
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Syntax error fixed in notebook comprehensively.")

if __name__ == '__main__':
    patch()
