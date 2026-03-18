import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

# load data
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(URL, header=None)

# feature names
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

df.columns = feature_names + ["label"]
X = df[feature_names].values
y = df["label"].values

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("1a. Confusion Matrix")
print(f"{'':>20} Predicted NOT SPAM  Predicted SPAM")
print(f"  Actual NOT SPAM      {cm[0,0]:>9}      {cm[0,1]:>9}")
print(f"  Actual SPAM     {cm[1,0]:>9}      {cm[1,1]:>9}")

# accuracy & error
acc  = accuracy_score(y_test, y_pred)
err  = 1 - acc
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n1b. Accuracy & Error")
print(f"  Accuracy : {acc:.4f}")
print(f"  Error    : {err:.4f}")

print("\n1c. Precision, Recall, F1")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")

# coefficients
coefs = model.coef_[0]
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs
}).sort_values("coefficient", ascending=False)

TOP_N = 10
print("\n2. Feature Coefficients")
print(f"\n  Top {TOP_N} Positive Contributors (-> SPAM):")
print(f"  {'Feature':<40} Coefficient")
for _, row in coef_df.head(TOP_N).iterrows():
    print(f"  {row['feature']:<40} {row['coefficient']:+.4f}")

print(f"\n  Top {TOP_N} Negative Contributors (-> NOT SPAM):")
print(f"  {'Feature':<40} Coefficient")
for _, row in coef_df.tail(TOP_N).iloc[::-1].iterrows():
    print(f"  {row['feature']:<40} {row['coefficient']:+.4f}")

# vary decision threshold
thresholds = [0.25, 0.50, 0.75, 0.90]
proba = model.predict_proba(X_test)[:, 1]   # P(SPAM)

print("\n3. Metrics Across Decision Thresholds")
print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}")

for T in thresholds:
    y_t = (proba >= T).astype(int)
    a = accuracy_score(y_test, y_t)
    p = precision_score(y_test, y_t, zero_division=0)
    r = recall_score(y_test, y_t, zero_division=0)
    print(f"  {T:>10.2f}  {a:>10.4f}  {p:>10.4f}  {r:>10.4f}")

print("""
2.) The model assigns large positive coefficients to features such as
    capital_run_length_longest, char_freq_$, capital_run_length_average,
    char_freq_#, word_freq_3d, word_freq_remove, word_freq_000 and
    word_freq_free. These features are positively correlated with the
    SPAM class and contribute most to predicting a message as SPAM.
    Features with large negative coefficients include word_freq_george,
    word_freq_hp, word_freq_85, word_freq_cs, word_freq_meeting,
    word_freq_edu, word_freq_415, word_freq_lab and word_freq_hpl.
    These are negatively correlated with the SPAM class and push the
    prediction toward being NOT SPAM.

3.) For this model on the SPAMBASE test set, lowering T (e.g., to 0.25)
    increases recall (we catch more spam) at the cost of precision
    (more false positives), while raising T (e.g., to 0.75 or 0.90)
    decreases recall but increases precision. Around T = 0.50 we obtain
    a strong overall performance with high accuracy, precision and
    recall. As T increases toward 0.75 and 0.90, overall accuracy
    changes only slightly but recall drops noticeably, so more spam is
    misclassified as NOT SPAM even though most messages predicted as SPAM
    are indeed spam. In practice, a spam filter would usually choose a
    relatively low threshold to prioritize recall and avoid letting
    spam through, accepting some additional false positives.
""")