from pathlib import Path
import sys

ARTIFACT = Path(__file__).resolve().parent.parent / 'artifacts' / 'detector_bundle.pkl'
if ARTIFACT.exists():
    ARTIFACT.unlink()
    print('Removed existing artifact bundle')
else:
    print('No existing bundle to remove')

print('Rebuilding artifacts... this may take several minutes')
from catfish_core import load_artifacts
art = load_artifacts()
plots = list((Path(__file__).resolve().parent.parent / 'artifacts' / 'plots').glob('*.png'))
print('Artifacts rebuilt. Plots:')
for p in plots:
    print('-', p)
print('Done')
