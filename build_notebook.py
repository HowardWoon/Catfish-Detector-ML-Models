import json
import re

PY_PATH = 'wia1006_catfish_group7_v15_fixed.py'
NB_IN = 'WIA1006_Catfish_Group7_V15_FIXED.ipynb'
NB_OUT = 'WIA1006_Catfish_Group7_V16_FINAL.ipynb'

# 1. Read the updated python script and extract code cells
with open(PY_PATH, 'r', encoding='utf-8') as f:
    py_text = f.read()

# The cells in the .py file are separated by """## 📚 Cell X ... """ blocks.
# We can find all the code blocks by looking for `# ═══════════════════════════════════════════════════════════════\n# CELL`
code_blocks = {}
parts = re.split(r'(# ═══════════════════════════════════════════════════════════════\n# CELL \d+ [^\n]+)', py_text)

# parts[0] is preamble. parts[1] is header for cell 1, parts[2] is code for cell 1, etc.
for i in range(1, len(parts), 2):
    header = parts[i]
    code = parts[i+1].split('"""##')[0].strip() # cut off the next markdown string
    
    # Extract cell number
    match = re.search(r'# CELL (\d+)', header)
    if match:
        cell_num = int(match.group(1))
        # Ensure code ends cleanly and doesn't contain the next markdown block
        code_blocks[cell_num] = header + '\n' + code + '\n'

# Replace Cell 1 with the universal file uploader/finder
code_blocks[1] = """# ═══════════════════════════════════════════════════════════════
# CELL 1 | Universal Dataset Loader — run FIRST every session
# ═══════════════════════════════════════════════════════════════
import os
import subprocess

CSV_PATH = 'dating_app_behavior_dataset.csv'

if not os.path.exists(CSV_PATH):
    print("Dataset not found locally. Searching in Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        r = subprocess.run(['find', '/content/drive/MyDrive', '-name', CSV_PATH], capture_output=True, text=True)
        found = r.stdout.strip().split('\\n')[0]
        if found:
            CSV_PATH = found
            print(f'✅ Found at: {CSV_PATH}')
        else:
            raise FileNotFoundError
    except Exception:
        print("\\n❌ Dataset not found in Drive.")
        print("Please upload dating_app_behavior_dataset.csv manually:")
        try:
            from google.colab import files
            uploaded = files.upload()
            if uploaded:
                CSV_PATH = list(uploaded.keys())[0]
        except ImportError:
            print("Not running in Colab. Please place the CSV in the current folder.")

if os.path.exists(CSV_PATH):
    print(f'\\n✅ Dataset ready: {os.path.getsize(CSV_PATH)/1e6:.1f} MB')
    print(f'   {CSV_PATH}')
"""

# 2. Read the notebook JSON
with open(NB_IN, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 3. Inject code into the matching code cells in the notebook
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        match = re.search(r'# CELL (\d+)', source)
        if match:
            cell_num = int(match.group(1))
            if cell_num in code_blocks:
                # Update the source. notebook expects a list of strings or a single string
                # We split by lines and add '\n' back to all but the last line
                lines = code_blocks[cell_num].split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
                print(f"Updated Cell {cell_num}")

# 4. Save the new notebook
with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print(f"Successfully generated {NB_OUT}!")
