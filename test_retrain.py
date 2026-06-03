import sys, traceback
print("STARTING")
try:
    from catfish_core import load_artifacts, ARTIFACT_BUNDLE_PATH, _bundle_needs_retrain
    print("Bundle path:", ARTIFACT_BUNDLE_PATH)
    print("Exists:", ARTIFACT_BUNDLE_PATH.exists())
    a = load_artifacts()
    print("LOAD_ARTIFACTS FINISHED")
    print("Exists after:", ARTIFACT_BUNDLE_PATH.exists())
except Exception as e:
    print("EXCEPTION:", e)
    traceback.print_exc()
print("DONE")
