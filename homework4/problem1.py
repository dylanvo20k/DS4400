import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score)

 # load and split data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1].values   # 57 word-frequency features
y = df.iloc[:, -1].values    # 1 = SPAM, 0 = NOT SPAM

# 80/20 stratified split (preserves class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# helper function to compute all required metrics
def eval_model(model, X_tr, y_tr, X_te, y_te, label=""):
    # returns dict of train/test error, accuracy, F1, AUC.
    tr_pred  = model.predict(X_tr)
    te_pred  = model.predict(X_te)
    tr_prob  = model.predict_proba(X_tr)[:, 1]
    te_prob  = model.predict_proba(X_te)[:, 1]

    metrics = {
        "train_error":    1 - accuracy_score(y_tr, tr_pred),
        "test_error":     1 - accuracy_score(y_te, te_pred),
        "train_acc":      accuracy_score(y_tr, tr_pred),
        "test_acc":       accuracy_score(y_te, te_pred),
        "train_f1":       f1_score(y_tr, tr_pred),
        "test_f1":        f1_score(y_te, te_pred),
        "train_auc":      roc_auc_score(y_tr, tr_prob),
        "test_auc":       roc_auc_score(y_te, te_prob),
    }

    if label:
        print(f"\n{'='*45}")
        print(f"  {label}")
        print(f"{'='*45}")
        print(f"  Train Error:   {metrics['train_error']:.4f}  |  Train Acc: {metrics['train_acc']:.4f}")
        print(f"  Test  Error:   {metrics['test_error']:.4f}  |  Test  Acc: {metrics['test_acc']:.4f}")
        print(f"  Train F1:      {metrics['train_f1']:.4f}")
        print(f"  Test  F1:      {metrics['test_f1']:.4f}")
        print(f"  Train AUC:     {metrics['train_auc']:.4f}")
        print(f"  Test  AUC:     {metrics['test_auc']:.4f}")

    return metrics

# problem 1.1 - information gain (entropy), no pruning
dt_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_entropy.fit(X_train, y_train)

m_entropy = eval_model(
    dt_entropy, X_train, y_train, X_test, y_test,
    label="1.1 - Information Gain (Entropy), no pruning"
)

# observations printed for writeup:
print("""
Observations (1.1):
  - Unpruned entropy tree almost perfectly fits training data
    (train acc 0.9997, train error 0.0003).
  - Test performance is lower (test acc 0.9197, test error 0.0803,
    test F1 0.8984, test AUC 0.9164), showing overfitting.
  - The train-test gap is large, so this model does not generalize as
    well as a pruned tree.
""")

# problem 1.2 - gini index, no pruning
dt_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_gini.fit(X_train, y_train)

m_gini = eval_model(
    dt_gini, X_train, y_train, X_test, y_test,
    label="1.2 - Gini Index, no pruning"
)

print("""
Observations (1.2):
  - Gini and entropy both nearly memorize the training set
    (train error 0.0003 for both), so both overfit when unpruned.
  - On test data, gini is slightly worse than entropy:
    error 0.0890 vs 0.0803, F1 0.8877 vs 0.8984,
    AUC 0.9078 vs 0.9164.
  - Difference is small, but entropy is better on this split.
""")

# problem 1.3 - depth pruning: train/test error vs max_depth
depths      = range(1, 31)          # sweep depths 1 – 30
train_errs  = []
test_errs   = []

for d in depths:
    clf = DecisionTreeClassifier(
        criterion="entropy",         # use information gain (same as 1.1)
        max_depth=d,
        random_state=42
    )
    clf.fit(X_train, y_train)
    train_errs.append(1 - accuracy_score(y_train, clf.predict(X_train)))
    test_errs.append( 1 - accuracy_score(y_test,  clf.predict(X_test)))

# plot train/test error vs max_depth
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, train_errs, "o-", label="Train Error", color="#185FA5", linewidth=1.8, markersize=5)
ax.plot(depths, test_errs,  "s-", label="Test Error",  color="#D85A30", linewidth=1.8, markersize=5)

# mark optimal depth (min test error)
opt_depth = list(depths)[np.argmin(test_errs)]
opt_err   = min(test_errs)
ax.axvline(opt_depth, linestyle="--", color="gray", linewidth=1, alpha=0.7)
ax.annotate(
    f"Optimal depth = {opt_depth}\n(test err = {opt_err:.3f})",
    xy=(opt_depth, opt_err),
    xytext=(opt_depth + 1.5, opt_err + 0.015),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray"),
    color="gray"
)

ax.set_xlabel("Max Tree Depth", fontsize=12)
ax.set_ylabel("Error Rate", fontsize=12)
ax.set_title("Train vs Test Error by Max Depth - SPAMBASE (Information Gain)", fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks(list(depths))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("depth_vs_error.png", dpi=150)
plt.show()

print(f"\nOptimal max_depth (min test error): {opt_depth}")
print(f"  Train error at opt depth: {train_errs[opt_depth-1]:.4f}")
print(f"  Test  error at opt depth: {opt_err:.4f}")

print("""
Observations (1.3):
  - Small depths underfit (high train and test error).
  - Test error decreases as depth increases, reaches its minimum at
    depth 11 (0.0695), then rises again.
  - Train error keeps decreasing with depth, while test error worsens
    after depth 11, which is overfitting.
  - Recommended max depth: 11. It improves test error over the
    unpruned entropy tree (0.0695 vs 0.0803).
""")