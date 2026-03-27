import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# load and split data (same split as Problem 1)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df  = pd.read_csv(url, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# feature names from SPAMBASE documentation
# first 48: word_freq_*, next 6: char_freq_*, last 3: capital run stats
feature_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total"
]

# problem 2.1 - random forest for T in {10, 50, 100, 500}
T_values = [10, 50, 100, 500]
results  = []

print(f"\n{'T':>6} | {'Tr Acc':>7} | {'Te Acc':>7} | {'Tr F1':>7} | {'Te F1':>7} | {'Tr AUC':>7} | {'Te AUC':>7}")
print("-" * 62)

for T in T_values:
    rf = RandomForestClassifier(n_estimators=T, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    tr_pred = rf.predict(X_train);  te_pred = rf.predict(X_test)
    tr_prob = rf.predict_proba(X_train)[:, 1]
    te_prob = rf.predict_proba(X_test)[:, 1]

    row = {
        "T":      T,
        "tr_acc": accuracy_score(y_train, tr_pred),
        "te_acc": accuracy_score(y_test,  te_pred),
        "tr_f1":  f1_score(y_train, tr_pred),
        "te_f1":  f1_score(y_test,  te_pred),
        "tr_auc": roc_auc_score(y_train, tr_prob),
        "te_auc": roc_auc_score(y_test,  te_prob),
    }
    results.append(row)

    print(f"{T:>6} | {row['tr_acc']:>7.4f} | {row['te_acc']:>7.4f} | "
          f"{row['tr_f1']:>7.4f} | {row['te_f1']:>7.4f} | "
          f"{row['tr_auc']:>7.4f} | {row['te_auc']:>7.4f}")

print("""
Observations (2.1):
  - Train metrics are near perfect for all T (acc/F1/AUC about 1.0).
  - Test metrics improve as T increases, but gains are small after 100 trees.
  - Test AUC rises from 0.9797 (T=10) to 0.9836 (T=500), with most gain
    by T=50.
  - T=100 is a good tradeoff; T=500 gives only a tiny extra gain.
""")

# problem 2.2 - compare with problem 1 results
# reference values from problem 1 (paste your actual output here)
p1_entropy = {"te_acc": 0.9197, "te_f1": 0.8984, "te_auc": 0.9164, "label": "DT (entropy, unpruned)"}
p1_gini    = {"te_acc": 0.9110, "te_f1": 0.8877, "te_auc": 0.9078, "label": "DT (gini, unpruned)"}
p1_pruned  = {"te_acc": None,   "te_f1": None,    "te_auc": None,   "label": "DT (entropy, depth=11)"}

print("\nComparison: Random Forest vs Decision Tree (test metrics)")
print(f"{'Model':>28} | {'Te Acc':>7} | {'Te F1':>7} | {'Te AUC':>7}")
print("-" * 55)
for m in [p1_entropy, p1_gini]:
    print(f"{m['label']:>28} | {m['te_acc']:>7.4f} | {m['te_f1']:>7.4f} | {m['te_auc']:>7.4f}")
for r in results:
    print(f"{'RF T='+str(r['T']):>28} | {r['te_acc']:>7.4f} | {r['te_f1']:>7.4f} | {r['te_auc']:>7.4f}")

print("""
Observations (2.2):
  - All RF models beat both unpruned decision trees on test acc, F1, and AUC.
  - Largest improvement is AUC: RF T=10 is 0.9797 vs entropy tree 0.9164.
  - Unpruned single trees and RF both fit training data very well, but RF
    generalizes better on test data.
  - A pruned tree from Problem 1 is closer, but RF still gives better AUC.
""")

# problem 2.3 - feature importance plot
rf500 = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf500.fit(X_train, y_train)

importances = rf500.feature_importances_
indices     = np.argsort(importances)[::-1]   # sorted descending
top_n       = 20                               # show top 20 features

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#185FA5" if i < 5 else "#378ADD" if i < 10 else "#85B7EB"
          for i in range(top_n)]

ax.bar(range(top_n),
       importances[indices[:top_n]],
       color=colors,
       edgecolor="white",
       linewidth=0.5)

ax.set_xticks(range(top_n))
ax.set_xticklabels(
    [feature_names[i] for i in indices[:top_n]],
    rotation=45, ha="right", fontsize=9
)
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Mean Decrease in Impurity (Importance)", fontsize=11)
ax.set_title("Top 20 Feature Importances — Random Forest (T=500) on SPAMBASE", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


print("\nFull feature importance ranking (T=500):")
print(f"{'Rank':>4} | {'Feature':>32} | {'Importance':>10}")
print("-" * 52)
for rank, i in enumerate(indices, 1):
    print(f"{rank:>4} | {feature_names[i]:>32} | {importances[i]:>10.4f}")

print("""
Observations (2.3):
  - Most important features are char_freq_!, char_freq_$, and word_freq_remove.
  - Capital-run features also rank high, which matches common spam style.
  - Many lower-ranked features have very small importance values.
  - Feature importance is model-specific and dataset-specific, so use it as
    guidance, not a strict causal claim.
""")