"""
train_model.py
Run this SECOND after generate_dataset.py
Trains Random Forest + Isolation Forest and saves all charts
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

# Create output folders
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── LOAD DATA ──────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv('gaming_dataset.csv')
print(f"      Rows: {len(df)} | Columns: {df.shape[1]}")
print(f"      Legitimate: {(df['Label']=='Legitimate').sum()} | Cheating: {(df['Label']=='Cheating').sum()}")

# ── PREPROCESS ─────────────────────────────────────────────
print("[2/5] Preprocessing...")

FEATURE_COLS = [
    'Age', 'Gender', 'Location', 'GameGenre',
    'PlayTimeHours', 'SessionDurationMinutes', 'KillDeathRatio',
    'HeadshotPercentage', 'MovementSpeed', 'ReactionTimeMs',
    'AimAccuracy', 'ActionsPerMinute', 'NetworkLatencyMs',
    'PacketLossPercentage', 'InGamePurchases', 'AchievementRate',
    'EngagementLevel'
]

# Encode categorical columns
cat_cols = ['Gender', 'Location', 'GameGenre', 'EngagementLevel']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode label
label_enc = LabelEncoder()
df['Label_encoded'] = label_enc.fit_transform(df['Label'])

X = df[FEATURE_COLS]
y = df['Label_encoded']

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

# Split 70 / 15 / 15
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Save preprocessors
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_dict, 'models/label_encoders.pkl')
joblib.dump(label_enc, 'models/label_encoder_target.pkl')
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')

# ── RANDOM FOREST ───────────────────────────────────────────
print("[3/5] Training Random Forest (Supervised)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

rf_acc  = accuracy_score(y_test, y_pred)
rf_prec = precision_score(y_test, y_pred, zero_division=0)
rf_rec  = recall_score(y_test, y_pred, zero_division=0)
rf_f1   = f1_score(y_test, y_pred, zero_division=0)
rf_auc  = roc_auc_score(y_test, y_proba)

print(f"      Accuracy:{rf_acc:.2%}  Precision:{rf_prec:.2%}  Recall:{rf_rec:.2%}  F1:{rf_f1:.2%}  AUC:{rf_auc:.4f}")
joblib.dump(rf, 'models/random_forest_model.pkl')

# ── ISOLATION FOREST ────────────────────────────────────────
print("[4/5] Training Isolation Forest (Unsupervised)...")
X_legit = X_train[y_train == 1]   # train only on legitimate
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_legit)

iso_raw  = iso.predict(X_test)
iso_pred = (iso_raw == -1).astype(int)   # -1 = anomaly = cheat

iso_acc  = accuracy_score(y_test, iso_pred)
iso_prec = precision_score(y_test, iso_pred, zero_division=0)
iso_rec  = recall_score(y_test, iso_pred, zero_division=0)
iso_f1   = f1_score(y_test, iso_pred, zero_division=0)

print(f"      Accuracy:{iso_acc:.2%}  Precision:{iso_prec:.2%}  Recall:{iso_rec:.2%}  F1:{iso_f1:.2%}")
joblib.dump(iso, 'models/isolation_forest_model.pkl')

# ── COMBINED ────────────────────────────────────────────────
combined = ((y_pred == 1) | (iso_pred == 1)).astype(int)
comb_acc  = accuracy_score(y_test, combined)
comb_prec = precision_score(y_test, combined, zero_division=0)
comb_rec  = recall_score(y_test, combined, zero_division=0)
comb_f1   = f1_score(y_test, combined, zero_division=0)
fpr_rate  = 1 - precision_score(y_test, combined, pos_label=0, zero_division=0)

results = {
    'random_forest':    {'accuracy': rf_acc,   'precision': rf_prec,   'recall': rf_rec,   'f1': rf_f1,   'auc': rf_auc},
    'isolation_forest': {'accuracy': iso_acc,  'precision': iso_prec,  'recall': iso_rec,  'f1': iso_f1},
    'combined':         {'accuracy': comb_acc, 'precision': comb_prec, 'recall': comb_rec, 'f1': comb_f1, 'fpr': fpr_rate}
}
joblib.dump(results, 'models/results.pkl')

# ── CHARTS ──────────────────────────────────────────────────
print("[5/5] Generating charts...")

# 1. Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix', fontsize=14, fontweight='bold')
for ax, preds, title in zip(axes, [y_pred, combined], ['Random Forest', 'Combined Framework']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. ROC Curve
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr_roc, tpr_roc, color='blue', lw=2, label=f'AUC = {rf_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.fill_between(fpr_roc, tpr_roc, alpha=0.1, color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Feature Importance
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
colors = ['#d62728' if v > imp.mean() else '#1f77b4' for v in imp.values]
plt.figure(figsize=(9, 7))
imp.plot(kind='barh', color=colors)
plt.axvline(imp.mean(), color='orange', linestyle='--', label='Mean')
plt.title('Feature Importance (Red = Above Average)', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Performance Comparison
cats = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
existing = [0.82, 0.80, 0.79, 0.80]
proposed = [comb_acc, comb_prec, comb_rec, comb_f1]
x = np.arange(len(cats))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, existing, w, label='Existing System', color='#2196F3', alpha=0.85)
b2 = ax.bar(x + w/2, proposed, w, label='Proposed AI System', color='#FF9800', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, 1.15)
ax.set_title('Existing vs Proposed System Performance', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f'{b.get_height():.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Behavior Distribution
df_orig = pd.read_csv('gaming_dataset.csv')
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Player Behavior: Legitimate vs Cheating', fontsize=13, fontweight='bold')
feats = ['KillDeathRatio', 'HeadshotPercentage', 'ReactionTimeMs',
         'AimAccuracy', 'ActionsPerMinute', 'MovementSpeed']
for ax, feat in zip(axes.flatten(), feats):
    for label, color in [('Legitimate', '#2196F3'), ('Cheating', '#F44336')]:
        ax.hist(df_orig[df_orig['Label'] == label][feat], bins=30,
                alpha=0.6, color=color, label=label)
    ax.set_title(feat, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/behavior_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 50)
print("  TRAINING COMPLETE!")
print("=" * 50)
print(f"  Random Forest  → Accuracy: {rf_acc:.2%}  F1: {rf_f1:.2%}")
print(f"  Isolation Forest→ Accuracy: {iso_acc:.2%}  F1: {iso_f1:.2%}")
print(f"  Combined        → Accuracy: {comb_acc:.2%}  F1: {comb_f1:.2%}")
print(f"  False Positive Rate: {fpr_rate:.2%}")
print(f"  AUC-ROC: {rf_auc:.4f}")
print("\n  Charts saved in: outputs/")
print("  Models saved in: models/")
print("  Run realtime_detection.py next!")
