import json
with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for line in source:
            if 'Ready to train on' in line:
                # We found the end of Cell 12 probably
                print("End of Cell 12:")
                for s in source[-5:]:
                    print(repr(s).encode('ascii', 'ignore').decode('ascii'))
