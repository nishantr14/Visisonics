# Cell 1 - Imports
import os, sys, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#080D12',
    'axes.facecolor':   '#0D1520',
    'axes.edgecolor':   '#2a3a4a',
    'axes.labelcolor':  '#8A9BB0',
    'text.color':       '#ffffff',
    'xtick.color':      '#8A9BB0',
    'ytick.color':      '#8A9BB0',
    'legend.facecolor': '#0D1520',
    'legend.edgecolor': '#2a3a4a',
    'grid.color':       '#1a2a3a',
    'font.size':        11,
})
print("Imports OK")


# Cell 2 - Load dataset04
df = pd.read_csv('../data/BATADAL_dataset04.csv')
df.columns = [c.strip() for c in df.columns]
y_true = (df['ATT_FLAG'].values == 1).astype(int)
print(f"Dataset04: {len(df)} samples")
print(f"Normal: {(y_true == 0).sum()}, Attack: {(y_true == 1).sum()}")
print(f"Attack rate: {y_true.mean()*100:.1f}%")


# Cell 3 - Load model artifacts
with open('../models/feature_names_supervised.json') as f:
    FEATURE_NAMES = json.load(f)
with open('../models/supervised_meta.json') as f:
    META = json.load(f)

SENSOR_COLS = META['sensor_cols']
OPTIMAL_THRESHOLD = META['optimal_threshold']

model = joblib.load('../models/supervised_model.pkl')
scaler = joblib.load('../models/scaler_supervised.pkl')

print(f"Model: {META['best_model']}")
print(f"CV F1: {META['cv_f1']}")
print(f"Threshold: {OPTIMAL_THRESHOLD}")
print(f"Features: {META['n_features']}")

sys.path.insert(0, os.path.abspath('..'))
from train_supervised import engineer_features
print("Loaded OK")


# Cell 4 - Vectorized scoring
X_eng = engineer_features(df, SENSOR_COLS)
X_eng = X_eng[FEATURE_NAMES]
X_scaled = scaler.transform(X_eng)

y_proba = model.predict_proba(X_scaled)[:, 1]
scores = y_proba * 100
y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

print(f"Predicted attacks: {y_pred.sum()} / {len(y_pred)}")
print(f"True attacks:      {y_true.sum()}")
print(f"Score range: [{scores.min():.1f}, {scores.max():.1f}]")


# Cell 5 - Metrics
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
auc  = roc_auc_score(y_true, y_proba)

print("=" * 60)
print("OTShield - Benchmark Results on BATADAL Dataset04")
print("=" * 60)
print()
print(f"{'Model':<20} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'AUC':>6}")
print("-" * 60)
print(f"{'OTShield (ours)':<20} | {prec:>9.3f} | {rec:>6.3f} | {f1:>6.3f} | {auc:>6.3f}")
print(f"{'Zahoor et al. 2025':<20} | {'---':>9} | {'---':>6} | {'0.835':>6} | {'0.891':>6}  <- baseline")
print()
print(f"CV F1 (5-fold):  {META['cv_f1']:.4f}")
print(f"Final F1:        {f1:.4f}")
print(f"Final AUC:       {auc:.4f}")
print()
if f1 >= 0.835:
    print(f">>> OTShield EXCEEDS baseline by {f1 - 0.835:+.3f} <<<")
else:
    print(f">>> Gap to baseline: {0.835 - f1:.3f} <<<")


# Cell 6a - ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_true, y_proba)
auc_val = roc_auc_score(y_true, y_proba)
ax.plot(fpr, tpr, color='#FF1A1A', linewidth=2.5, label=f"OTShield (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], 'w--', alpha=0.2, linewidth=1)
ax.scatter([1 - 0.891], [0.891], color='#FFB800', s=150, zorder=5, marker='*', label='Zahoor et al. 2025 (AUC=0.891)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('OTShield - ROC Curve vs Baseline', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()


# Cell 6b - F1 Bar Chart
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['OTShield\n(ours)', 'Zahoor et al.\n2025']
f1_vals = [f1, 0.835]
colors = ['#FF1A1A', '#FFB800']
bars = ax.bar(labels, f1_vals, color=colors, width=0.5, edgecolor='none')
for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('OTShield vs Baseline - F1 Score', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(f1_vals) * 1.25)
ax.axhline(y=0.835, color='#FFB800', linestyle='--', alpha=0.4, label='Baseline (0.835)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


# Cell 6c - Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('OTShield - Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.show()
print(f"\nTrue Positives:  {cm[1,1]} / {y_true.sum()} attacks detected")
print(f"False Positives: {cm[0,1]} / {(y_true==0).sum()} normal flagged")
print(f"False Negatives: {cm[1,0]} / {y_true.sum()} attacks missed")


# Cell 8 - Demo simulation
sys.path.insert(0, os.path.abspath('../src'))
from supervised_scorer import get_supervised_score, reset_history

normal_idx = np.where(y_true == 0)[0][100]
attack_indices = np.where(y_true == 1)[0]

print("=" * 60)
print("OTShield - LIVE DEMO SIMULATION")
print("=" * 60)

for label, idx in [("NORMAL", normal_idx), ("ATTACK 1", attack_indices[0]), ("ATTACK 2", attack_indices[len(attack_indices)//2])]:
    reset_history()
    score, expl, feat = get_supervised_score(df.iloc[idx].to_dict())
    gt = "ATTACK" if y_true[idx] == 1 else "NORMAL"
    print(f"\n[{label}] Sample #{idx}: score={score:.1f}/100  top={feat}  truth={gt}")

print("\nDemo complete")

