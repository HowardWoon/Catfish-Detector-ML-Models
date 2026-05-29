
import json

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "App Usage Time vs Outcome" in src and "Messages Sent vs Outcome" in src:
            # We add rotation to axes[1,0] and axes[1,1]
            if "axes[1,0].tick_params" not in src:
                src = src.replace("axes[1,0].set_title('App Usage Time vs Outcome', fontweight='bold')", "axes[1,0].set_title('App Usage Time vs Outcome', fontweight='bold')\\naxes[1,0].tick_params(axis='x', rotation=45)")
            if "axes[1,1].tick_params" not in src:
                src = src.replace("axes[1,1].set_title('Messages Sent vs Outcome', fontweight='bold')", "axes[1,1].set_title('Messages Sent vs Outcome', fontweight='bold')\\naxes[1,1].tick_params(axis='x', rotation=45)")
            
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [src.split("\\n")[-1]]
            break

with open("WIA1006_Catfish_Group7_V19_CHAMPION.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Patched EDA rotation!")

