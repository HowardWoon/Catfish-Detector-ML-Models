import json

def patch_notebook():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        if cell_type == 'code':
            new_source = []
            for line in source:
                # Fix backwards description of swipe_msg_ratio
                if '# - swipe_msg_ratio: How often they swipe compared to messages sent.' in line:
                    line = line.replace('# - swipe_msg_ratio: How often they swipe compared to messages sent.', 
                                        '# - swipe_msg_ratio: Ratio of messages sent per right-swipe.')
                new_source.append(line)
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook Cell 6 Descriptions Synced Successfully.")

if __name__ == '__main__':
    patch_notebook()
