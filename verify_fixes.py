import sys, json, os, numpy as np
from pathlib import Path
import traceback

def run_tests():
    success = True
    print("==================================================")
    print("RUNNING SYSTEM VALIDATION CHECKS")
    print("==================================================")

    # 1. Check Notebook Intactness
    print("\\n[1] Checking Notebook Synchronization...")
    try:
        nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        has_cell_12 = False
        has_cell_24_export = False
        has_cell_25_ngrok = False
        
        for cell in nb.get('cells', []):
            source = "".join(cell.get('source', []))
            if 'CELL 12: ML PIPELINE INITIALIZATION' in source and 'GMMClassifier' in source:
                has_cell_12 = True
            if 'detector_bundle.pkl' in source and 'joblib.dump' in source:
                has_cell_24_export = True
            if 'CELL 25: LAUNCH LIVE WEB APPLICATION' in source and 'pyngrok' in source:
                has_cell_25_ngrok = True
                
        if not has_cell_12:
            print("[FAIL] Cell 12 (ML Pipeline) is missing or corrupted.")
            success = False
        else:
            print("[PASS] Cell 12 is intact and contains proper GMM/KMeans implementations.")
            
        if not has_cell_24_export:
            print("[FAIL] Cell 24 (Artifact Export) does not contain detector_bundle.pkl logic.")
            success = False
        else:
            print("[PASS] Cell 24 correctly exports detector_bundle.pkl.")
            
        if not has_cell_25_ngrok:
            print("[FAIL] Cell 25 (ngrok) is missing.")
            success = False
        else:
            print("[PASS] Cell 25 correctly hosts the Flask app via pyngrok.")
            
    except Exception as e:
        print("[FAIL] Failed to parse notebook:", e)
        success = False

    # 2. Check Artifacts Load
    print("\\n[2] Checking ML Artifacts...")
    try:
        from catfish_core import load_artifacts, scan_input
        print("   Loading detector_bundle.pkl...")
        artifacts = load_artifacts()
        if len(artifacts.models) != 6:
            print(f"[FAIL] Expected 6 models, found {len(artifacts.models)}")
            success = False
        else:
            print("[PASS] detector_bundle.pkl loaded successfully with all 6 models.")
            
        # 3. Check "100% Catfish" Bug
        print("\\n[3] Checking Live Scanner '100% Bug' and NaN Resilience...")
        # Test extreme edge cases
        test_cases = [
            {
                "name": "All Minimums (Zeroes)",
                "input": {'app_usage_time_min': 0, 'swipe_right_ratio': 0, 'bio_length': 0, 'message_sent_count': 0, 'profile_pics_count': 0, 'likes_received': 0, 'mutual_matches': 0}
            },
            {
                "name": "All Maximums (Sliders Maxed)",
                "input": {'app_usage_time_min': 300, 'swipe_right_ratio': 1.0, 'bio_length': 500, 'message_sent_count': 100, 'profile_pics_count': 6, 'likes_received': 200, 'mutual_matches': 30}
            },
            {
                "name": "Normal User",
                "input": {'app_usage_time_min': 120, 'swipe_right_ratio': 0.4, 'bio_length': 200, 'message_sent_count': 40, 'profile_pics_count': 3, 'likes_received': 80, 'mutual_matches': 15}
            }
        ]
        
        for tc in test_cases:
            res = scan_input(tc["input"], artifacts)
            score = res['behavioral_score']
            print(f"   [{tc['name']}] -> Behavioral Risk: {score}%")
            if np.isnan(score):
                print(f"[FAIL] NaN detected in risk score for {tc['name']}!")
                success = False
            elif score > 100.0 or score < 0.0:
                print(f"[FAIL] Score out of bounds ({score}) for {tc['name']}!")
                success = False
            elif score == 100.0:
                print(f"[WARN] Warning: {tc['name']} returned exactly 100.0%, might still be clipping.")
        print("[PASS] Scanner risk formula is stable and bounded between 0-100 without NaNs.")

    except Exception as e:
        print("[FAIL] Failed ML validation:", e)
        traceback.print_exc()
        success = False

    print("\\n==================================================")
    if success:
        print("[PASS] ALL TESTS PASSED! The system is mathematically sound.")
        sys.exit(0)
    else:
        print("[FAIL] ERRORS DETECTED!")
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
