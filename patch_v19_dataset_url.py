
import json

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# The first code cell is where the loading logic is. Let us find it and replace it.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "google.colab" in src and "drive.mount" in src and "dating_app_behavior_dataset.csv" in src:
            new_src = """# ---------------------------------------------------------------
# CELL 1 | Universal Dataset Loader (Seamless GitHub Fetch)
# ---------------------------------------------------------------
import os
import urllib.request

CSV_PATH = "dating_app_behavior_dataset.csv"
GITHUB_URL = "https://raw.githubusercontent.com/HowardWoon/Catfish-Detector-ML-Models/main/dating_app_behavior_dataset.csv"

if not os.path.exists(CSV_PATH):
    print("Dataset not found locally. Downloading directly from GitHub repository...")
    try:
        urllib.request.urlretrieve(GITHUB_URL, CSV_PATH)
        print("? Successfully downloaded dataset from GitHub!")
    except Exception as e:
        print(f"? Failed to download dataset: {e}")
        print("Please upload dating_app_behavior_dataset.csv manually using the folder icon on the left.")

if os.path.exists(CSV_PATH):
    print(f"\\n? Dataset ready: {os.path.getsize(CSV_PATH)/1e6:.1f} MB")
    print(f"   {CSV_PATH}")
"""
            cell["source"] = [line + "\\n" for line in new_src.split("\\n")[:-1]] + [new_src.split("\\n")[-1]]
            break

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Patched dataset loader!")

