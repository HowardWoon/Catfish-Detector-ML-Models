import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import threading
import webbrowser
import time
import os
import urllib.request
from urllib.error import URLError

def open_browser():
    # Wait until the server actually responds before opening the browser
    max_retries = 300  # 30 seconds max wait
    for _ in range(max_retries):
        try:
            urllib.request.urlopen("http://127.0.0.1:5000/api/check", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    
    print("\n" + "="*60)
    print("🎉 CATFISH DETECTOR WEB APP IS LIVE!")
    print("🌐 Opening in your default web browser...")
    print("❌ DO NOT CLOSE THIS TERMINAL. To stop, press CTRL+C")
    print("="*60 + "\n")
    sys.stdout.flush()
    
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except:
        pass

if __name__ == "__main__":
    print("============================================================")
    print("🚀 BOOTING CATFISH DETECTOR WEB APP...")
    print("============================================================")
    print("📦 Loading/Training Machine Learning Models (this may take a few minutes on first run)...")
    sys.stdout.flush()

    # Pre-load/train the artifacts in the main thread before starting the Flask app.
    # This prevents worker processes from recursive training and makes startup output visible.
    from catfish_core import load_artifacts
    load_artifacts()

    from webapp.app import create_app
    app = create_app()

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)