import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_spambase():
    """Load SPAMBASE and return features, labels, feature names."""
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


def sigmoid(z):
    # Numerically stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_loss(X, y, theta):
    """
    Cross-entropy for logistic regression.
    X: (n_samples, n_features) including intercept column
    y: (n_samples,) in {0,1}
    theta: (n_features,)
    """
    n_samples = X.shape[0]
    logits = X @ theta
    p = sigmoid(logits)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss


def gradient_descent_logistic(X, y, alpha, num_iterations, record_iters):
    """
    Batch gradient descent for logistic regression.
    Returns theta and a dict of losses at the specified iteration numbers.
    """
    n_samples, n_features = X.shape
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    theta = np.zeros(n_features + 1)

    losses = {}
    record_iters_set = set(record_iters)

    for i in range(1, num_iterations + 1):
        logits = X_with_intercept @ theta
        p = sigmoid(logits)
        error = p - y  # derivative of cross-entropy wrt logits
        grad = (1.0 / n_samples) * (X_with_intercept.T @ error)
        theta -= alpha * grad

        if i in record_iters_set:
            losses[i] = cross_entropy_loss(X_with_intercept, y, theta)

    return theta, losses


def predict_proba_logistic(X, theta):
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    logits = X_with_intercept @ theta
    return sigmoid(logits)


def metrics_from_probs(y_true, p_hat, threshold=0.5):
    y_pred = (p_hat >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def main():
    X, y = load_spambase()
    X_train, X_test, y_train, y_test = train_test_split_and_scale(X, y)

    # package (scikit-learn) baseline on the same split
    pkg_model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    pkg_model.fit(X_train, y_train)
    pkg_train_proba = pkg_model.predict_proba(X_train)[:, 1]
    pkg_test_proba = pkg_model.predict_proba(X_test)[:, 1]
    pkg_train_metrics = metrics_from_probs(y_train, pkg_train_proba, threshold=0.5)
    pkg_test_metrics = metrics_from_probs(y_test, pkg_test_proba, threshold=0.5)

    print("Package (scikit-learn) Logistic Regression (T = 0.50)")
    print("  Training:  Acc = {:.4f}, Prec = {:.4f}, Rec = {:.4f}, F1 = {:.4f}".format(*pkg_train_metrics))
    print("  Testing :  Acc = {:.4f}, Prec = {:.4f}, Rec = {:.4f}, F1 = {:.4f}".format(*pkg_test_metrics))

    # gradient descent logistic regression
    learning_rates = [0.001, 0.01, 0.1]
    record_iters = [10, 50, 100]
    num_iterations = max(record_iters)

    print("\nGradient Descent Logistic Regression")
    print("Cross-entropy loss on TRAINING set at selected iterations:")
    print("  {:>8} {:>8} {:>18}".format("alpha", "iters", "loss"))

    results = []

    for alpha in learning_rates:
        theta, losses = gradient_descent_logistic(
            X_train,
            y_train,
            alpha=alpha,
            num_iterations=num_iterations,
            record_iters=record_iters,
        )

        for it in sorted(record_iters):
            loss_value = losses.get(it, np.nan)
            print("  {:>8.3g} {:>8d} {:>18.6f}".format(alpha, it, loss_value))

        # metrics at 100 iterations
        train_proba = predict_proba_logistic(X_train, theta)
        test_proba = predict_proba_logistic(X_test, theta)

        train_metrics = metrics_from_probs(y_train, train_proba, threshold=0.5)
        test_metrics = metrics_from_probs(y_test, test_proba, threshold=0.5)

        results.append(
            {
                "alpha": alpha,
                "theta": theta,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )

    print("\nAccuracy, Precision, Recall, F1 at 100 iterations (T = 0.50)")
    print("  {:>8}  {:>8}  {:>9}  {:>9}  {:>9}  {:>9}".format(
        "alpha", "set", "Acc", "Prec", "Rec", "F1"
    ))
    for r in results:
        alpha = r["alpha"]
        tr_acc, tr_prec, tr_rec, tr_f1 = r["train_metrics"]
        te_acc, te_prec, te_rec, te_f1 = r["test_metrics"]
        print("  {:>8.3g}  {:>8}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}".format(
            alpha, "train", tr_acc, tr_prec, tr_rec, tr_f1
        ))
        print("  {:>8.3g}  {:>8}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9.4f}".format(
            alpha, "test", te_acc, te_prec, te_rec, te_f1
        ))

if __name__ == "__main__":
    main()

