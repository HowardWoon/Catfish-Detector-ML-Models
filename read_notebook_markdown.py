import json

nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

output = []
for i, cell in enumerate(nb.get('cells', [])):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])
    output.append(f"--- CELL {i} ({cell_type}) ---")
    if cell_type == 'markdown':
        output.extend(source)
    elif cell_type == 'code':
        # only get comments
        for line in source:
            if line.strip().startswith('#'):
                output.append(line)
            elif not line.strip():
                continue
            else:
                # Stop at the first non-comment, non-empty line to save space
                break
    output.append("\n")

with open('notebook_text.txt', 'w', encoding='utf-8') as f:
    f.writelines(output)
print("Saved to notebook_text.txt")
