import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_spambase():
    """Load SPAMBASE dataset and return X (features) and y (labels)."""
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


def k_fold_indices(n_samples, k, random_state=42):
    """
    Implement k-fold partitioning of indices.
    Returns a list of length k where each element is a numpy array
    containing the indices for that fold's validation set.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    return folds


def k_fold_cv_error(X, y, k, model_type="logreg"):
    """
    My own implementation of k-fold cross-validation:
      1. Split data indices into k folds.
      2. For each fold i, train on k-1 folds and validate on fold i.
      3. Record validation error (1 - accuracy) for each fold.
      4. Return the list of validation errors and their average.

    model_type: "logreg" or "lda"
    """
    n_samples = X.shape[0]
    folds = k_fold_indices(n_samples, k, random_state=42)

    val_errors = []

    for i, val_idx in enumerate(folds):
        # training indices are all others
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # fit scaler on training data only, then transform train and val
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if model_type == "logreg":
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        elif model_type == "lda":
            clf = LDA()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        clf.fit(X_train_scaled, y_train)
        y_val_pred = clf.predict(X_val_scaled)

        acc = accuracy_score(y_val, y_val_pred)
        err = 1.0 - acc
        val_errors.append(err)

        print(
            f"    Fold {i+1}/{k} ({model_type}): "
            f"validation accuracy = {acc:.4f}, error = {err:.4f}"
        )

    val_errors = np.array(val_errors)
    avg_error = val_errors.mean()
    return val_errors, avg_error


def main():
    X, y = load_spambase()

    ks = [5, 10]
    results = {}

    for k in ks:
        print(f"\n{k}-fold Cross-Validation")

        print("\n  Logistic Regression:")
        logreg_errors, logreg_avg_err = k_fold_cv_error(X, y, k=k, model_type="logreg")

        print("\n  LDA:")
        lda_errors, lda_avg_err = k_fold_cv_error(X, y, k=k, model_type="lda")

        results[(k, "logreg")] = logreg_avg_err
        results[(k, "lda")] = lda_avg_err

        print(
            f"\nSummary for k = {k}:\n"
            f"  Logistic Regression average validation error: {logreg_avg_err:.4f}\n"
            f"  LDA                average validation error: {lda_avg_err:.4f}\n"
        )

    print("Overall average validation errors across k values:")
    print("  {:>6}  {:>15}  {:>10}".format("k", "Model", "Avg Error"))
    for (k, model_name), avg_err in results.items():
        print(f"  {k:>6}  {model_name:>15}  {avg_err:>10.4f}")

    # simple comparison / observations
    best_model = min(results.items(), key=lambda item: item[1])
    (best_k, best_name), best_err = best_model

    print(
        f"""
3.) Across k = 5 and k = 10, the best-performing configuration
    (lowest average validation error) is:
      - Model: {best_name}
      - k-folds: {best_k}
      - Average validation error: {best_err:.4f}

    Logistic regression performs better than LDA under both 5-fold and
    10-fold CV, with about 3.5-4% lower average validation error. LDA
    does worse likely because it assumes both classes have the same
    spread in the data, which may not be true for SPAMBASE's word
    frequency features. Using k=10 instead of k=5 slightly improves
    logistic regression (0.0767 -> 0.0737) since each fold trains on
    more data, while LDA stays about the same (0.1130 -> 0.1141).
"""
    )


if __name__ == "__main__":
    main()

