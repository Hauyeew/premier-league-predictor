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


print("Creating features...")

df = pd.read_csv("data/matches.csv")

features = create_features(df)

print(f"Created {len(features)} training examples")


# ==========================================
# FEATURES USED BY THE MODEL
# ==========================================

feature_columns = [

    # Overall home team stats
    "HomeGoalsScored",
    "HomeGoalsConceded",
    "HomeShots",
    "HomeShotsOnTarget",

    # Overall away team stats
    "AwayGoalsScored",
    "AwayGoalsConceded",
    "AwayShots",
    "AwayShotsOnTarget",

    # Overall form
    "HomeForm",
    "AwayForm",
    "FormDifference",

    # Venue-specific form
    "HomeHomeForm",
    "AwayAwayForm",
    "HomeAwayFormDifference",

    # Venue-specific goals
    "HomeGoalsScoredAtHome",
    "HomeGoalsConcededAtHome",
    "AwayGoalsScoredAway",
    "AwayGoalsConcededAway",

    "HomeXG",
    "HomeXGA",
    "AwayXG",
    "AwayXGA",
    "XGAttackDifference",
    "XGDefenseDifference",  

    # Team strength
    "HomeWinRate",
    "AwayWinRate",
    "HomePointsPerGame",
    "AwayPointsPerGame",
    "HomeWinRateDifference",
    "PointsPerGameDifference",

    # Other differences
    "AttackDifference",
    "DefenseDifference",
    "HomeAwayAttackDifference",
    "HomeAwayDefenseDifference",
    "ShotsDifference",
    "ShotsOnTargetDifference",

    # Head-to-head
    "H2HHomeWinRate",
    "H2HDrawRate",
    "H2HAwayWinRate",
    "H2HGoalsFor",
    "H2HGoalsAgainst",
    "H2HGoalDifference",
    "H2HMeetings"
]


X = features[feature_columns]
y = features["Result"]


# ==========================================
# ENCODE H / D / A
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)


# ==========================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(features) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]


print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")


# ==========================================
# XGBOOST
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

model.fit(X_train, y_train)


import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

print()
print("==============================")
print("Feature Importance")
print("==============================")
print(importance.to_string())

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print()
print("==============================")
print(f"Accuracy: {accuracy:.3f}")
print("==============================")


# ==========================================
# CONFUSION MATRIX
# ==========================================

print()
print("==============================")
print("Confusion Matrix")
print("==============================")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# LOG LOSS
# ==========================================

loss = log_loss(y_test, probabilities)

print()
print("==============================")
print("Log Loss")
print("==============================")

print(f"{loss:.4f}")


# ==========================================
# CLASSIFICATION REPORT
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
# EXAMPLE PREDICTIONS
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
# SAVE MODEL
# ==========================================

with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

print()
print("Model saved to model/model.pkl")