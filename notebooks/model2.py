# ---
# jupyter:
#   colab:
#     provenance: []
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.6
# ---

# %%
import numpy as np

# Metrics
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    average_precision_score,  # PR-AUC
    roc_auc_score
)

# Preprocessing + Model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")

print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)
print("Positive rate (train):", y_train.mean())
print("Positive rate (val):  ", y_val.mean())


# %%
# Class counts for imbalance handling
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()

print("Train positives:", n_pos)
print("Train negatives:", n_neg)

# Similar to logistic pos_weight: negatives / positives
pos_weight = n_neg / n_pos
class_weight = {0: 1.0, 1: float(pos_weight)}

print("Computed pos_weight (neg/pos):", pos_weight)
print("Using class_weight:", class_weight)


# %%
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Build a pipeline: StandardScaler -> MLPClassifier
ann_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),  # 2 hidden layers
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=30,
        alpha=1e-4,
        class_weight=class_weight,
        random_state=42
    ))
])

@ignore_warnings(category=ConvergenceWarning)
def fit_ann_model(pipe, X_train, y_train):
    return pipe.fit(X_train, y_train)

ann_model = fit_ann_model(ann_pipeline, X_train, y_train)
print("ANN model trained.")


# %%
def hit_rate_at_k(y_true, y_prob, k):
    # Measures what fraction of the top-k highest-probability cells actually had a crash.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    idx_sorted = np.argsort(y_prob)[::-1]
    top_idx = idx_sorted[:k]
    hits = y_true[top_idx].sum()
    return hits / float(k)


# Get probabilities on validation set
y_val_proba = ann_model.predict_proba(X_val)[:, 1]

brier = brier_score_loss(y_val, y_val_proba)
logloss = log_loss(y_val, y_val_proba)
pr_auc = average_precision_score(y_val, y_val_proba)
roc_auc = roc_auc_score(y_val, y_val_proba)

# Choose a k for hit-rate
k = 500
hr_k = hit_rate_at_k(y_val, y_val_proba, k)

print("=== ANN (MLP) Model – Validation Metrics ===")
print(f"Brier score:        {brier:.6f}")
print(f"Log-loss:           {logloss:.6f}")
print(f"PR-AUC (avg prec):  {pr_auc:.6f}")
print(f"ROC-AUC:            {roc_auc:.6f}")
print(f"Hit-rate@k (k={k}): {hr_k:.6f}")

