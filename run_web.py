import threading
import webbrowser
import time
import sys

print("\n" + "="*60)
print("⏳  LOADING MACHINE LEARNING MODELS... PLEASE WAIT!")
print("This may take 10-20 seconds depending on your PC.")
print("="*60 + "\n")
sys.stdout.flush()

from webapp.app import create_app

app = create_app()

def open_browser():
    # Wait for the server to bind
    time.sleep(2)
    print("\n" + "="*60)
    print("🚀  CATFISH DETECTOR WEB APP IS LIVE!")
    print("🌐  Click this link to open: http://127.0.0.1:5000")
    print("❌  DO NOT CLOSE THIS TERMINAL. To stop, press CTRL+C")
    print("="*60 + "\n")
    sys.stdout.flush()
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except:
        pass

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)