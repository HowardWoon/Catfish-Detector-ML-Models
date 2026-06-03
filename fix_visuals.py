import json

path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'ax.scatter(' in line and 'color' in line and 'ra[pi]' in line:
                cell['source'][i] = line.replace("color='#ff0000',s=150", "color='#ffff00',s=350,linewidths=3").replace('color=\"#ff0000\",s=150', 'color=\"#ffff00\",s=350,linewidths=3')
            if 'font-size:12px' in line and 'CATFISH' in line and 'GENUINE' in line:
                cell['source'][i] = line.replace('font-size:12px', 'font-size:18px').replace('font-weight:800', 'font-weight:900').replace('color:white;', 'color:white;text-shadow: 1px 1px 2px black;')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
