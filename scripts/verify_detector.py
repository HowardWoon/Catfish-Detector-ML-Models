from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.validation import run_validation_suite


if __name__ == "__main__":
    for item in run_validation_suite():
        print(f"{item['name']}: {'PASS' if item['passed'] else 'FAIL'} | {item['details']}")