import sys
from catfish_core import evaluate_all
res = evaluate_all()
for r in res['model_details']:
    print(f"{r['name']}: Accuracy {r['accuracy_pct']:.2f}% | Recall {r['recall_pct']:.2f}% | Precision {r['precision_pct']:.2f}% | F1 {r['f1_score_pct']:.2f}% | ROC-AUC {r['roc_auc']:.4f}")
