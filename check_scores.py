import json
import base64
import os

def check():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    img_idx = 0
    os.makedirs('graphs', exist_ok=True)
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'text' in out:
                    text = "".join(out['text'])
                    if 'Tuning' in text or 'AUC' in text:
                        print(text.encode('ascii', 'ignore').decode('ascii'))
                
                # Check for images
                if 'data' in out and 'image/png' in out['data']:
                    img_data = out['data']['image/png']
                    with open(f'graphs/graph_{img_idx}.png', 'wb') as f_img:
                        f_img.write(base64.b64decode(img_data))
                    img_idx += 1

if __name__ == '__main__':
    check()
