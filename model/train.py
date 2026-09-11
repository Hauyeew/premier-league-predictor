
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

from features.feature_engineering import create_features


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/matches.csv"
MODEL_FILE = "model/model.pkl"

TEST_SIZE = 0.20
N_FOLDS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} matches")


# ============================================================
# CREATE FEATURES
# ============================================================

print()
print("Creating features...")

features = create_features(df)

print(f"Created {len(features)} training examples")


# ============================================================
# PREPARE FEATURES / TARGET
# ============================================================

feature_columns = [
    col
    for col in features.columns
    if col not in ["Result", "Date", "HomeTeam", "AwayTeam"]
]

X = features[feature_columns]
y = features["Result"]


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print()
print("Classes:", label_encoder.classes_)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(X) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]

print()
print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

model_params = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


# ============================================================
# CHRONOLOGICAL CROSS-VALIDATION
# ============================================================

print()
print()
print("==============================")
print("Time-Series Cross-Validation")
print("==============================")

tscv = TimeSeriesSplit(n_splits=N_FOLDS)
cv_results = []

for fold, (train_idx, valid_idx) in enumerate(
    tscv.split(X_train), 1
):

    X_cv_train = X_train.iloc[train_idx]
    X_cv_valid = X_train.iloc[valid_idx]

    y_cv_train = y_train[train_idx]
    y_cv_valid = y_train[valid_idx]

    print()
    print(f"Fold {fold}")
    print("------------------------------")
    print(f"Train matches:      {len(X_cv_train)}")
    print(f"Validation matches: {len(X_cv_valid)}")

    model = XGBClassifier(**model_params)

    model.fit(
        X_cv_train,
        y_cv_train
    )

    probabilities = model.predict_proba(
        X_cv_valid
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    fold_accuracy = accuracy_score(
        y_cv_valid,
        predictions
    )

    fold_log_loss = log_loss(
        y_cv_valid,
        probabilities,
        labels=[0, 1, 2]
    )

    cv_results.append({
        "fold": fold,
        "accuracy": fold_accuracy,
        "log_loss": fold_log_loss,
    })

    print(
        f"Accuracy: {fold_accuracy:.4f}"
    )

    print(
        f"Log Loss: {fold_log_loss:.4f}"
    )


cv_df = pd.DataFrame(cv_results)

print()
print("==============================")
print("Cross-Validation Summary")
print("==============================")

print(
    cv_df.to_string(index=False)
)

print()
print(
    f"Average Accuracy: "
    f"{cv_df['accuracy'].mean():.4f}"
)

print(
    f"Average Log Loss: "
    f"{cv_df['log_loss'].mean():.4f}"
)

print(
    f"Accuracy Std: "
    f"{cv_df['accuracy'].std():.4f}"
)

print(
    f"Log Loss Std: "
    f"{cv_df['log_loss'].std():.4f}"
)

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print()
print("==============================")
print("Training Final Model")
print("==============================")

final_model = XGBClassifier(
    **model_params
)

final_model.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL TEST SET
# ============================================================

test_probabilities = final_model.predict_proba(
    X_test
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


# ============================================================
# FINAL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

print()
print("==============================")
print(
    f"Final Test Accuracy: "
    f"{accuracy:.4f}"
)
print("==============================")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print()

print("==============================")

print("Confusion Matrix")

print("==============================")

print(cm)




# ============================================================
# FINAL LOG LOSS
# ============================================================

loss = log_loss(
    y_test,
    test_probabilities,
    labels=[0, 1, 2]
)

print()
print("==============================")
print(
    f"Final Test Log Loss: "
    f"{loss:.4f}"
)
print("==============================")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("==============================")
print("Feature Importance")
print("==============================")

importance = pd.Series(
    final_model.feature_importances_,
    index=feature_columns
)

importance = importance.sort_values(
    ascending=False
)

print(
    importance.to_string()
)


# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================

print()
print("==============================")
print("Example Predictions")
print("==============================")

for i in range(
    min(10, len(X_test))
):

    actual = label_encoder.inverse_transform(
        [y_test[i]]
    )[0]

    predicted = label_encoder.inverse_transform(
        [test_predictions[i]]
    )[0]

    probabilities = test_probabilities[i]

    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"A: {probabilities[0]:.3f} "
        f"D: {probabilities[1]:.3f} "
        f"H: {probabilities[2]:.3f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

saved_model = {
    "model": final_model,
    "label_encoder": label_encoder,
    "feature_columns": feature_columns,
}

joblib.dump(
    saved_model,
    MODEL_FILE
)

print()
print("==============================")
print(
    f"Model saved to {MODEL_FILE}"
)
print("==============================")
