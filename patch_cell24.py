import json
import os

def patch_notebook():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', [])
        
        # Look for Cell 24
        if any('CELL 24: ASSET EXPORT' in line for line in source) or any('CELL 24 | Export' in line for line in source):
            new_source = []
            skip = False
            for line in source:
                if line.strip() == "# Build artifact bundle for website link":
                    skip = True
                if skip:
                    if line.strip() == "for name,model in models.items():":
                        skip = False
                        
                if not skip:
                    new_source.append(line)
                    if line.startswith("print('   💾 Pipeline assets saved to', EXP)"):
                        # Inject our new bundle code
                        new_source.extend([
                            "\n",
                            "# Build artifact bundle for website link\n",
                            "try:\n",
                            "    from catfish_core import DetectorArtifacts\n",
                            "    import copy, pandas as pd, shutil\n",
                            "    safe_models = copy.copy(models)\n",
                            "    arts = DetectorArtifacts(\n",
                            "        dataset_shape=df.shape,\n",
                            "        class_counts=df['Target'].value_counts().to_dict(),\n",
                            "        feature_names=FEATURE_NAMES,\n",
                            "        num_cols=NUM_COLS,\n",
                            "        train_medians_raw=X_TRAIN_MEDIANS_RAW,\n",
                            "        genuine_medians_raw=GENUINE_MEDIANS_RAW,\n",
                            "        catfish_medians_raw=CATFISH_MEDIANS_RAW,\n",
                            "        scaler=scaler,\n",
                            "        thresholds=BEST_THRESHOLDS,\n",
                            "        models=safe_models,\n",
                            "        feature_importances={},\n",
                            "        population_stats=POP,\n",
                            "        notebook_cells=[],\n",
                            "        leaderboard=pd.DataFrame(),\n",
                            "        model_metrics=pd.DataFrame(),\n",
                            "        test_profiles={}\n",
                            "    )\n",
                            "    bundle_path = os.path.join(EXP, 'detector_bundle.pkl')\n",
                            "    joblib.dump(arts, bundle_path)\n",
                            "    print('   💾 detector_bundle.pkl (Website Link Ready)')\n",
                            "    local_artifacts = os.path.join(os.getcwd(), 'artifacts')\n",
                            "    os.makedirs(local_artifacts, exist_ok=True)\n",
                            "    shutil.copy(bundle_path, os.path.join(local_artifacts, 'detector_bundle.pkl'))\n",
                            "except Exception as e:\n",
                            "    print(f'Warning: could not bundle for website: {e}')\n",
                            "\n"
                        ])
                        
            cell['source'] = new_source
            break
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook Cell 24 Patched!")

if __name__ == '__main__':
    patch_notebook()
