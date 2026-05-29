import threading
import webbrowser
import time
from webapp.app import create_app

app = create_app()

def open_browser():
    # Wait a second for the server to start
    time.sleep(1.5)
    print("\n" + "="*60)
    print("🚀  CATFISH DETECTOR WEB APP IS LIVE!")
    print("🌐  Opening browser automatically to: http://127.0.0.1:5000")
    print("❌  DO NOT CLOSE THIS WINDOW. To stop, press CTRL+C")
    print("="*60 + "\n")
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Start the browser-opening thread
    threading.Thread(target=open_browser, daemon=True).start()
    # Start the Flask server
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)