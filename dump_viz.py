import json

def dump_viz():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = "".join(cell.get('source', []))
            if "name =" in src and "except Exception" in src:
                name_line = [line for line in src.split('\n') if "name =" in line][0]
                print("=====================")
                print(name_line)
                print("=====================")
                # Print just the try/except block
                lines = src.split('\n')
                in_try = False
                for line in lines:
                    if "try:" in line:
                        in_try = True
                    if in_try:
                        print(line)

if __name__ == '__main__':
    dump_viz()
