import sys
sys.path.insert(0, 'webapp')

if __name__ == "__main__":
    try:
        import app
        # Just force the artifact load
        app.get_artifacts()
        print("SUCCESS")
    except Exception as e:
        print(f"CRASH: {e}")
