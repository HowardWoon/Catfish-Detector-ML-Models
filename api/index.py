import sys
from pathlib import Path

# Ensure the root project directory is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from webapp.app import create_app

app = create_app()
