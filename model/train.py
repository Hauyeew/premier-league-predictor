
import pandas as pd
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.preprocessing import LabelEncoder

from features.feature_engineering import create_features


# ==========================================
# 1. Load data
# ==========================================

print("Creating features...")

df = pd.read_csv("data/matches.csv")

features = create_features(df)

print(f"Created {len(features)} training examples")


# ==========================================
# 2. Select features
# ==========================================

feature_columns = [
    "HomeGoalsScored",
    "HomeGoalsConceded",
    "HomeShots",
    "HomeShotsOnTarget",

    "AwayGoalsScored",
    "AwayGoalsConceded",
    "AwayShots",
    "AwayShotsOnTarget",

    "HomeForm",
    "AwayForm",
    "FormDifference",

    "HomeHomeForm",
    "AwayAwayForm",
    "HomeAwayFormDifference",

    "HomeGoalsScoredAtHome",
    "HomeGoalsConcededAtHome",

    "AwayGoalsScoredAway",
    "AwayGoalsConcededAway",

    "AttackDifference",
    "DefenseDifference",

    "HomeAwayAttackDifference",
    "HomeAwayDefenseDifference",

    "ShotsDifference",
    "ShotsOnTargetDifference"
]

X = features[feature_columns]
y = features["Result"]


# ==========================================
# 3. Convert A/D/H into numbers
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)


# ==========================================
# 4. Chronological train/test split
# ==========================================

split_index = int(len(features) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]

print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")


# ==========================================
# 5. Create XGBoost model
# ==========================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",

    random_state=42
)


# ==========================================
# 6. Train model
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. Make predictions
# ==========================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)


# ==========================================
# 8. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print()
print("==============================")
print(f"Accuracy: {accuracy:.3f}")
print("==============================")


# ==========================================
# 9. Confusion matrix
# ==========================================

print()
print("==============================")
print("Confusion Matrix")
print("==============================")

print(confusion_matrix(y_test, predictions))


# ==========================================
# 10. Log loss
# ==========================================

loss = log_loss(y_test, probabilities)

print()
print("==============================")
print("Log Loss")
print("==============================")

print(f"{loss:.4f}")


# ==========================================
# 11. Classification report
# ==========================================

print()
print("==============================")
print("Classification Report")
print("==============================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 12. Show example predictions
# ==========================================

print()
print("==============================")
print("Example Predictions")
print("==============================")

for i in range(min(10, len(X_test))):

    actual = label_encoder.inverse_transform(
        [y_test[i]]
    )[0]

    predicted = label_encoder.inverse_transform(
        [predictions[i]]
    )[0]

    probs = probabilities[i]

    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"A: {probs[0]:.3f} "
        f"D: {probs[1]:.3f} "
        f"H: {probs[2]:.3f}"
    )


# ==========================================
# 13. Save model
# ==========================================

with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

print()
print("Model saved to model/model.pkl")
