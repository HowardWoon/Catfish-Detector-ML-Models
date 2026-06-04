import sys
sys.path.append('webapp')

try:
    import app
    # Just force the artifact load
    app.get_artifacts()
    print("SUCCESS")
except Exception as e:
    print(f"CRASH: {e}")
