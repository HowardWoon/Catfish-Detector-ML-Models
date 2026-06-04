import json

def fix_cells():
    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells'][-7:]:
        source = "".join(cell['source'])
        
        # Fix Cell A
        if "# CELL A | PROFILE COMPARISON MODE" in source:
            source = source.replace(
                "sliders_a = []\nsliders_b = []\n\nfor i, f in enumerate(feature_names):\n    sa = widgets.FloatSlider(value=GENUINE_MEDIANS_RAW[i], min=0, max=max_vals[i], step=0.01 if max_vals[i]==1.0 else 1, description=f, style={'description_width': '120px'})\n    sb = widgets.FloatSlider(value=CATFISH_MEDIANS_RAW[i], min=0, max=max_vals[i], step=0.01 if max_vals[i]==1.0 else 1, description=f, style={'description_width': '120px'})",
                "sliders_a = []\nsliders_b = []\nRAW_COLS = ['app_usage_time_min', 'swipe_right_ratio', 'bio_length', 'message_sent_count', 'profile_pics_count', 'likes_received', 'mutual_matches']\ngen_vals = [GENUINE_MEDIANS_RAW.get(c, 0) for c in RAW_COLS]\ncat_vals = [CATFISH_MEDIANS_RAW.get(c, 0) for c in RAW_COLS]\n\nfor i, f in enumerate(feature_names):\n    sa = widgets.FloatSlider(value=gen_vals[i], min=0, max=max_vals[i], step=0.01 if max_vals[i]==1.0 else 1, description=f, style={'description_width': '120px'})\n    sb = widgets.FloatSlider(value=cat_vals[i], min=0, max=max_vals[i], step=0.01 if max_vals[i]==1.0 else 1, description=f, style={'description_width': '120px'})"
            )
            
        # Fix Cell C
        if "# CELL C | FEATURE IMPORTANCE RADAR CHART" in source:
            source = source.replace(
                "cat_meds = np.array(CATFISH_MEDIANS_RAW)\ncurr_user = np.array(GENUINE_MEDIANS_RAW)",
                "RAW_COLS = ['app_usage_time_min', 'swipe_right_ratio', 'bio_length', 'message_sent_count', 'profile_pics_count', 'likes_received', 'mutual_matches']\ncat_meds = np.array([CATFISH_MEDIANS_RAW.get(c, 0) for c in RAW_COLS])\ncurr_user = np.array([GENUINE_MEDIANS_RAW.get(c, 0) for c in RAW_COLS])"
            )
            
        cell['source'] = [line + '\n' for line in source.split('\n') if line != '']
        if not cell['source']:
            cell['source'] = [source]
        else:
            # Fix duplicate newlines created by split/join
            cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]

    with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Cells successfully fixed!")

if __name__ == '__main__':
    fix_cells()
