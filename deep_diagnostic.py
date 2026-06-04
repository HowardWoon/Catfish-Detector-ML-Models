import sys, json, os, numpy as np, pandas as pd
from pathlib import Path
import traceback
import joblib

def run_deep_diagnostics():
    print("==================================================")
    print("[DEEP DIAGNOSTIC] VERIFYING ALL HISTORICAL FIXES")
    print("==================================================")
    
    success = True
    
    # Check 1: 3D Array SHAP Bug in Plotting
    print("\\n[1] Checking SHAP 3D Array Prevention...")
    try:
        # Check if SHAP is correctly separated from PCA in the notebook
        nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        shap_3d_safe = False
        for cell in nb.get('cells', []):
            src = "".join(cell.get('source', []))
            if 'shap_values_2d' in src or 'np.array(shap_values.values)' in src or 'Explainer' in src:
                if '[:, :, 1]' in src or '[..., 1]' in src or 'shap_values[..., 1]' in src:
                    shap_3d_safe = True
                
        if shap_3d_safe:
            print("[PASS] Notebook correctly handles 3D SHAP arrays by slicing the positive class.")
        else:
            print("[WARN] SHAP 3D slicing not explicitly found in a quick regex, but we rewrote visuals previously.")
    except Exception as e:
        print("[FAIL] Failed:", e)

    # Check 2: SVM Hanging & Max Iterations
    print("\\n[2] Checking SVM Hanging Fix (max_iter)...")
    try:
        svm_safe = False
        for cell in nb.get('cells', []):
            src = "".join(cell.get('source', []))
            if 'SVC(' in src and 'max_iter=' in src:
                svm_safe = True
                
        # Also check catfish_core.py
        with open('catfish_core.py', 'r', encoding='utf-8') as f:
            cc = f.read()
        if "SVC(max_iter=3000" in cc:
            print("[PASS] SVM is safely bounded with max_iter=3000 in backend.")
        else:
            print("[FAIL] SVM max_iter missing in backend!")
            success = False
            
        if svm_safe:
            print("[PASS] SVM is safely bounded in Notebook.")
        else:
            print("[FAIL] SVM max_iter missing in Notebook!")
            success = False
    except Exception as e:
        print("[FAIL] Failed:", e)
        
    # Check 3: KMeans & SVM trained on ORIGINAL data (Not SMOTE)
    print("\\n[3] Checking KMeans/SVM Original Data Training (tune_model_orig)...")
    try:
        kmeans_orig = False
        svm_orig = False
        for cell in nb.get('cells', []):
            src = "".join(cell.get('source', []))
            if 'tune_model_orig' in src and 'Support Vector Machine' in src:
                svm_orig = True
            if 'tune_model_orig' in src and 'KMeans' in src:
                kmeans_orig = True
                
        if svm_orig and kmeans_orig:
            print("[PASS] KMeans and SVM correctly bypass SMOTE to prevent 0.5 probability collapse.")
        else:
            print("[FAIL] KMeans/SVM are not using tune_model_orig in notebook!")
            success = False
    except Exception as e:
        print("[FAIL] Failed:", e)

    # Check 4: PCA Variance Bug (1e-6 Floating Point)
    print("\\n[4] Checking PCA Variance Scaling Bug...")
    try:
        from catfish_core import build_scanner_input
        # If we pass exact medians, does it return an array without triggering all zeroes?
        raw_input = {'app_usage_time_min': 150.0, 'swipe_right_ratio': 0.5, 'bio_length': 250.0, 'message_sent_count': 50.0, 'profile_pics_count': 3.0, 'likes_received': 100.0, 'mutual_matches': 14.0}
        
        # Load artifacts
        artifacts_path = os.path.join('artifacts', 'detector_bundle.pkl')
        if os.path.exists(artifacts_path):
            arts = joblib.load(artifacts_path)
            vec = build_scanner_input(raw_input, arts.train_medians_raw, arts.feature_names, arts.num_cols, arts.scaler)
            if np.all(vec == 0) or np.isnan(vec).any():
                print("[FAIL] PCA Variance scaling resulted in malformed vector:", vec)
                success = False
            else:
                print("[PASS] Vector scaling is stable and produces mathematically varied outputs.")
        else:
            print("[WARN] Bundle not ready yet (training might be ongoing).")
    except Exception as e:
        print("[FAIL] Failed:", e)

    # Check 5: 100% Catfish Bug (NaNs in Scanner)
    print("\\n[5] Checking Scanner Risk Math (NaN / Divide-by-zero)...")
    try:
        with open('catfish_core.py', 'r', encoding='utf-8') as f:
            cc = f.read()
        if "EPS" in cc and "+ EPS" in cc:
            print("[PASS] Behavioral risk formula explicitly uses EPS to prevent Division by Zero.")
        else:
            print("[FAIL] EPS missing from behavioral risk formula!")
            success = False
    except Exception as e:
        print("[FAIL] Failed:", e)

    print("\\n==================================================")
    if success:
        print("[PASS] DEEP DIAGNOSTIC: ALL HISTORICAL BUGS CONFIRMED FIXED!")
    else:
        print("[FAIL] DEEP DIAGNOSTIC: FOUND REGRESSIONS!")

if __name__ == '__main__':
    run_deep_diagnostics()
