import json

with open('executed_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('dump_output.txt', 'w', encoding='utf-8') as out:
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and '12' in ''.join(cell['source']):
            out.write('======================================================================\n')
            out.write('MARKDOWN:\n')
            out.write(''.join(cell['source']) + '\n')
            out.write('----------------------------------------------------------------------\n')
            out.write('CODE CELL FOLLOWING:\n')
            if idx+1 < len(nb['cells']):
                out.write(''.join(nb['cells'][idx+1]['source']) + '\n')
            else:
                out.write('NO CODE CELL\n')
