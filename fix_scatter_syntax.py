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
                # Fix repeated keyword argument
                if "linewidths=3" in line and "linewidths=2" in line:
                    line = line.replace("linewidths=3,edgecolors='black',linewidths=2", "linewidths=2,edgecolors='black'")
                    # Might have spaces
                    line = line.replace("linewidths=3, edgecolors='black', linewidths=2", "linewidths=2, edgecolors='black'")
                    line = line.replace("linewidths=3,edgecolors='black',linewidths=2", "linewidths=2,edgecolors='black'")
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Scatter plot syntax error fixed.")

if __name__ == '__main__':
    patch()
