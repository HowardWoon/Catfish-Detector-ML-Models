"""Comprehensive notebook audit script."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells = [(i, c) for i, c in enumerate(cells) if c['cell_type'] == 'code']
print(f'=== NOTEBOOK: {len(code_cells)} code cells ===\n')

all_issues = []
for i, c in code_cells:
    src = ''.join(c['source'])
    issues = []
    if '_et_imp' in src: issues.append('HAS _et_imp guard')
    if 'top_indices = []' in src: issues.append('EMPTY CHART - top_indices=[]')
    if 'Extreme Gradient Boosting' in src: issues.append('WRONG DESC: says XGBoost')
    if 'estimators (trees)' in src: issues.append('WRONG DESC: says trees for GMM')
    for v in ['_et_imp', 'feature_importances_et', 'et_feat_imp']:
        if v in src and v != '_et_imp': issues.append(f'Old variable: {v}')
    
    firstline = src.splitlines()[0].strip() if src.strip() else 'EMPTY'
    if issues:
        print(f'  Cell {i} [{firstline[:50]}]:')
        for iss in issues:
            print(f'    - {iss}')
        all_issues.extend(issues)

if not all_issues:
    print('  NO ISSUES FOUND - All cells are clean')

print()
print('=== KEY CELLS ===')
# Check live scanner cell (should be cell 60)
for i in [57, 58, 60, 61, 62]:
    if i >= len(cells): continue
    src = ''.join(cells[i]['source'])
    first = src.splitlines()[0].strip()[:70] if src.strip() else 'EMPTY'
    print(f'Cell {i}: {first}')

print()
print('=== CELL 60 (LIVE SCANNER) CHECK ===')
c60 = ''.join(cells[60]['source']) if len(cells) > 60 else ''
if 'App_Usage_Time_min' in c60 or 'app_usage_time_min' in c60:
    print('  HAS slider inputs')
if 'behavioral_risk =' in c60:
    # Check if it's an assignment (hardcoded) vs function call
    for line in c60.split('\n'):
        if 'behavioral_risk' in line:
            print(f'  behavioral_risk line: {line.strip()}')
            break

print()
print('=== CELL 30 (GMM) CHECK ===')
c30 = ''.join(cells[30]['source']) if len(cells) > 30 else ''
print(f'  Length: {len(c30)} chars')
if 'plt.show' in c30:
    print('  HAS plt.show - visualization exists')
if 'PCA' in c30:
    print('  HAS PCA visualization')
if 'scatter' in c30.lower():
    print('  HAS scatter plot')

print()
print('=== CELL 32 (SVM) CHECK ===')
c32 = ''.join(cells[32]['source']) if len(cells) > 32 else ''
print(f'  Length: {len(c32)} chars')
if 'plt.show' in c32:
    print('  HAS plt.show - visualization exists')
if 'permutation' in c32.lower():
    print('  HAS permutation importance')
if 'hist' in c32.lower():
    print('  HAS histogram')

print()
print('=== CELL 34 (KMEANS) CHECK ===')
c34 = ''.join(cells[34]['source']) if len(cells) > 34 else ''
print(f'  Length: {len(c34)} chars')
if 'plt.show' in c34:
    print('  HAS plt.show - visualization exists')
if 'scatter' in c34.lower():
    print('  HAS scatter plot')

print()
print('=== CELL 57 (SHAP) CHECK ===')
c57 = ''.join(cells[57]['source']) if len(cells) > 57 else ''
print(f'  Length: {len(c57)} chars')
if '_et_imp' in c57:
    print('  ERROR: Still has _et_imp guard!')
else:
    print('  OK: No _et_imp guard')
if "'models' not in dir()" in c57 or '"models" not in dir()' in c57:
    print('  OK: Has proper models guard')

print()
print('=== VERDICT ===')
print(f'Total issues found: {len(all_issues)}')
