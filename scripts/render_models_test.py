import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.app import create_app
from flask import render_template

app = create_app()
plot_dir = Path('webapp') / 'static' / 'plots'
images = []
if plot_dir.exists():
    images = [p.name for p in sorted(plot_dir.iterdir()) if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]

# Minimal context
artifacts = type('A', (), {})()
artifacts.leaderboard = []
artifacts.feature_names = []
artifacts.test_profiles = {}
artifacts.dataset_shape = (0, 0)
artifacts.models = []
artifacts.class_counts = {'catfished': 0}

app.test_request_context().push()
html = render_template('models.html', page_title='Models', page_description='desc', active_page='models', images=images, leaderboard=[], test_profiles={}, artifacts=artifacts)
out = Path('tmp_models_rendered.html')
out.write_text(html, encoding='utf-8')
print('Wrote', out.resolve())
