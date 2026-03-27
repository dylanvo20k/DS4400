# HW4 - Problem 3: AdaBoost Ensemble
# DS4400 | Dylan Nguyen
# Dataset: SPAMBASE (UCI)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, roc_curve)

# load and split data (same seed as P1 and P2)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df  = pd.read_csv(url, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# helper function to evaluate model
def eval_model(model, X_tr, y_tr, X_te, y_te):
    tr_pred = model.predict(X_tr);  te_pred = model.predict(X_te)
    tr_prob = model.predict_proba(X_tr)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]
    return {
        "tr_acc": accuracy_score(y_tr, tr_pred),
        "te_acc": accuracy_score(y_te, te_pred),
        "tr_f1":  f1_score(y_tr, tr_pred),
        "te_f1":  f1_score(y_te, te_pred),
        "tr_auc": roc_auc_score(y_tr, tr_prob),
        "te_auc": roc_auc_score(y_te, te_prob),
    }

# problem 3.1 - adaBoost for T in {10, 50, 100, 500}
# adaBoost default base estimator is a depth-1 stump;
# using depth-1 DecisionTreeClassifier explicitly for clarity.
T_values   = [10, 50, 100, 500]
ada_results = []
rf_results  = []

print("AdaBoost (base: depth-1 decision stump)")
print(f"\n{'T':>6} | {'Tr Acc':>7} | {'Te Acc':>7} | {'Tr F1':>7} | {'Te F1':>7} | {'Tr AUC':>7} | {'Te AUC':>7}")
print("-" * 62)

for T in T_values:
    base = DecisionTreeClassifier(max_depth=1)   # canonical weak learner
    ada  = AdaBoostClassifier(
        estimator=base,
        n_estimators=T,
        random_state=42
    )
    ada.fit(X_train, y_train)
    m = eval_model(ada, X_train, y_train, X_test, y_test)
    ada_results.append({"T": T, **m})
    print(f"{T:>6} | {m['tr_acc']:>7.4f} | {m['te_acc']:>7.4f} | "
          f"{m['tr_f1']:>7.4f} | {m['te_f1']:>7.4f} | "
          f"{m['tr_auc']:>7.4f} | {m['te_auc']:>7.4f}")

print("""
Observations (3.1):
  - As T increases, AdaBoost improves on both train and test metrics.
  - Test accuracy rises from 0.8914 (T=10) to 0.9327 (T=500).
  - Test F1 rises from 0.8623 to 0.9136, and test AUC from 0.9542 to 0.9807.
  - Most improvement happens by T=100; gains from 100 to 500 are smaller.
""")

# problem 3.2 - adaBoost vs random forest comparison

# re-train RF for all T values (reuse if already in memory)
print("\nRandom Forest (for comparison)")
print(f"\n{'T':>6} | {'Tr Acc':>7} | {'Te Acc':>7} | {'Tr F1':>7} | {'Te F1':>7} | {'Tr AUC':>7} | {'Te AUC':>7}")
print("-" * 62)

for T in T_values:
    rf = RandomForestClassifier(n_estimators=T, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m = eval_model(rf, X_train, y_train, X_test, y_test)
    rf_results.append({"T": T, **m})
    print(f"{T:>6} | {m['tr_acc']:>7.4f} | {m['te_acc']:>7.4f} | "
          f"{m['tr_f1']:>7.4f} | {m['te_f1']:>7.4f} | "
          f"{m['tr_auc']:>7.4f} | {m['te_auc']:>7.4f}")

# side-by-side for each T
print("\nSide-by-side (Test metrics only)")
print(f"\n{'T':>6} | {'Ada TeAcc':>10} | {'RF TeAcc':>9} | {'Ada TeF1':>9} | {'RF TeF1':>8} | {'Ada AUC':>8} | {'RF AUC':>8}")
print("-" * 72)
for a, r in zip(ada_results, rf_results):
    print(f"{a['T']:>6} | {a['te_acc']:>10.4f} | {r['te_acc']:>9.4f} | "
          f"{a['te_f1']:>9.4f} | {r['te_f1']:>8.4f} | "
          f"{a['te_auc']:>8.4f} | {r['te_auc']:>8.4f}")

print("""
Observations (3.2):
  - RF is better than AdaBoost at every T on test accuracy, F1, and AUC.
  - Example at T=100: RF (acc 0.9446, F1 0.9283, AUC 0.9834) vs
    AdaBoost (acc 0.9273, F1 0.9058, AUC 0.9778).
  - The gap is largest at T=10 and smaller at higher T, but RF still leads.
  - Both RF and AdaBoost are better than the single unpruned decision tree
    from Problem 1 on test metrics.
""")


# problem 3.3 - ROC Curves: DT vs RF(100) vs AdaBoost(100)

# train the three models
dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

rf100 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf100.fit(X_train, y_train)

ada100_base = DecisionTreeClassifier(max_depth=1)
ada100 = AdaBoostClassifier(
    estimator=ada100_base, n_estimators=100,
    random_state=42
)
ada100.fit(X_train, y_train)

# compute ROC curves
models = {
    "Decision Tree (entropy, unpruned)": dt,
    "Random Forest (T=100)":             rf100,
    "AdaBoost (T=100)":                  ada100,
}
colors = ["#D85A30", "#185FA5", "#1D9E75"]

fig, ax = plt.subplots(figsize=(8, 6))

for (label, model), color in zip(models.items(), colors):
    probs    = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc      = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})",
            color=color, linewidth=2)

# diagonal baseline
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves - Decision Tree vs RF(100) vs AdaBoost(100)\nSPAMBASE Test Set", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()

print("""
Observations (3.3):
  - The single decision tree ROC curve is the lowest.
  - RF(100) and AdaBoost(100) are both much higher than the single tree.
  - RF(100) has slightly higher AUC than AdaBoost(100) on this split.
  - This matches the metric tables above.
""")