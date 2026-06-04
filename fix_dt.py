import json

def fix_dt():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = "".join(cell.get('source', []))
            if "name = 'Decision Tree'" in src and "dt.named_steps" in src:
                # Replace dt with models[name]
                fixed_src = src.replace(
                    "dt.named_steps.get('decisiontreeclassifier', dt) if hasattr(dt, 'named_steps') else dt",
                    "models[name].named_steps.get('decisiontreeclassifier', models[name]) if hasattr(models[name], 'named_steps') else models[name]"
                )
                cell['source'] = [line + '\n' for line in fixed_src.split('\n') if line != '']
                cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
                print("Fixed NameError for dt in Cell 12b")
                
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_dt()
