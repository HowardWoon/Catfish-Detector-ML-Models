import shutil, pathlib
src = pathlib.Path('artifacts/plots')
dst = pathlib.Path('webapp/static/plots')
dst.mkdir(parents=True, exist_ok=True)
count = 0
for f in src.glob('*.png'):
    shutil.copy(f, dst / f.name)
    count += 1
print('copied', count, 'files to', dst)
