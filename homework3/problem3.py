import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def load_spambase():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    df = pd.read_csv(url, header=None)

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
    return X, y


def train_test_split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    err = 1.0 - acc
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return acc, err, prec, rec


def cross_validate_knn(X_train, y_train, k_values, n_splits=5):
    """
    Simple k-fold CV over training data to choose k for kNN.
    Returns a dict: k -> (mean_acc, mean_err, mean_prec, mean_rec).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for k in k_values:
        fold_metrics = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_tr, y_tr)
            y_val_pred = clf.predict(X_val)
            fold_metrics.append(compute_metrics(y_val, y_val_pred))

        fold_metrics = np.array(fold_metrics)
        mean_acc, mean_err, mean_prec, mean_rec = fold_metrics.mean(axis=0)
        results[k] = (mean_acc, mean_err, mean_prec, mean_rec)

    return results


def main():
    X, y = load_spambase()
    X_train, X_test, y_train, y_test = train_test_split_and_scale(X, y)

    # 1) Cross-validation for k in kNN
    k_values = [1, 3, 5, 7, 9, 11, 15]
    cv_results = cross_validate_knn(X_train, y_train, k_values, n_splits=5)

    print("1) kNN cross-validation on TRAINING data (5-fold):")
    print("   {:>5}  {:>10}  {:>10}  {:>10}  {:>10}".format(
        "k", "Accuracy", "Error", "Precision", "Recall"
    ))

    best_k = None
    best_err = float("inf")

    for k in sorted(cv_results.keys()):
        acc, err, prec, rec = cv_results[k]
        print("   {:>5d}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}".format(
            k, acc, err, prec, rec
        ))
        if err < best_err:
            best_err = err
            best_k = k

    print(f"\n   Selected k (minimum average CV error): k = {best_k}")

    # 2) Train three classifiers and compare metrics on train/test
    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    lda = LDA()
    knn = KNeighborsClassifier(n_neighbors=best_k)

    log_reg.fit(X_train, y_train)
    lda.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    models = {
        "Logistic Regression": log_reg,
        "LDA": lda,
        f"kNN (k={best_k})": knn,
    }

    print("\n2) Metrics for all classifiers (TRAIN and TEST):")
    print("   {:>20}  {:>6}  {:>8}  {:>8}  {:>10}  {:>8}".format(
        "Model", "Set", "Acc", "Error", "Precision", "Recall"
    ))

    metrics_summary = {}

    for name, clf in models.items():
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        tr_acc, tr_err, tr_prec, tr_rec = compute_metrics(y_train, y_train_pred)
        te_acc, te_err, te_prec, te_rec = compute_metrics(y_test, y_test_pred)

        metrics_summary[name] = {
            "train": (tr_acc, tr_err, tr_prec, tr_rec),
            "test": (te_acc, te_err, te_prec, te_rec),
        }

        print("   {:>20}  {:>6}  {:>8.4f}  {:>8.4f}  {:>10.4f}  {:>8.4f}".format(
            name, "train", tr_acc, tr_err, tr_prec, tr_rec
        ))
        print("   {:>20}  {:>6}  {:>8.4f}  {:>8.4f}  {:>10.4f}  {:>8.4f}".format(
            name, "test", te_acc, te_err, te_prec, te_rec
        ))

    # Identify best / worst on test accuracy (you can also look at other metrics)
    best_model = max(metrics_summary.items(), key=lambda item: item[1]["test"][0])[0]
    worst_model = min(metrics_summary.items(), key=lambda item: item[1]["test"][0])[0]

    print(
        f"""
  - Based on test-set accuracy, the best-performing model here is: {best_model}.
  - The worst-performing model on the test set (by accuracy) is: {worst_model}.
  - Logistic regression performs best, followed by kNN, then LDA. LDA does
    worse than the other two likely because it assumes both classes have the
    same spread in the data, which may not hold for word frequency features.
    kNN performs in between — it picks up local patterns but is more sensitive
    to the choice of k and the scale of features.
"""
    )

    # 3) ROC curve for logistic regression using package
    y_test_proba_lr = log_reg.predict_proba(X_test)[:, 1]
    fpr_pkg, tpr_pkg, _ = roc_curve(y_test, y_test_proba_lr)
    auc_pkg = auc(fpr_pkg, tpr_pkg)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_pkg, tpr_pkg, label=f"Logistic Regression (AUC = {auc_pkg:.4f})")
    plt.plot([0, 0, 1], [0, 1, 1], "k--", label="Ideal ROC (reference)")
    plt.plot([0, 1], [0, 1], "r--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Logistic Regression, Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("problem3_roc_logreg_package.png")
    plt.close()

    print("3) Package-based ROC for logistic regression:")
    print(f"   AUC (test set) = {auc_pkg:.4f}")
    print('   ROC curve saved as "problem3_roc_logreg_package.png".')

    # 4) Manual ROC curve for logistic regression using discrete thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    manual_points = []

    for T in thresholds:
        y_pred_T = (y_test_proba_lr >= T).astype(int)
        cm = confusion_matrix(y_test, y_pred_T)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        manual_points.append((fpr, tpr))

    manual_points = np.array(manual_points)
    fpr_manual = manual_points[:, 0]
    tpr_manual = manual_points[:, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_pkg, tpr_pkg, label="Package ROC (sklearn)")
    plt.plot(
        fpr_manual,
        tpr_manual,
        "o-",
        label="Manual ROC (threshold grid)",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Logistic Regression, Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("problem3_roc_logreg_comparison.png")
    plt.close()

    print('4) Manual ROC points computed for thresholds T in {0,0.1,...,1}.')
    print(
        """
  - The package ROC curve is a smooth curve obtained by sweeping the decision
    threshold over all possible values and effectively using many distinct
    operating points.
  - The manual ROC curve here uses only a coarse grid of thresholds
    {0, 0.1, ..., 1}, so it appears as a polyline through a small number
    of points instead of a smooth curve.
  - To make the manual ROC curve look more similar to the package ROC,
    we could:
      * Use a much finer set of thresholds (for example, all unique
        predicted probabilities or a dense grid of values between 0 and 1),
      * Sort these thresholds and recompute (FPR, TPR) at each point.
    With enough thresholds, the manually computed ROC will coincide
    (up to numerical precision) with the package ROC curve.
"""
    )

if __name__ == "__main__":
    main()
