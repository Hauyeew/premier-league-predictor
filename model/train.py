import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

from features.feature_engineering import create_features


# Load data
df = pd.read_csv("data/matches.csv")

df["Date"] = pd.to_datetime(df["Date"])

print("Creating features...")

features = create_features(df)

print(f"Created {len(features)} training examples")


# Features used by the model
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

    "GoalsScoredDifference",
    "GoalsConcededDifference",
    "ShotsDifference",
    "ShotsOnTargetDifference"
]


X = features[feature_columns]
y = features["Result"]


# Chronological split
# First 80% = training
# Last 20% = testing

split_index = int(len(features) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")


# Create model
model = LogisticRegression(
    max_iter=2000,
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\n==============================")
print("Model Classes")
print("==============================")
print(model.classes_)

print("\n==============================")
print("Example Predictions")
print("==============================")

for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]} | "
        f"Predicted: {y_pred[i]} | "
        f"A: {y_prob[i][0]:.3f} "
        f"D: {y_prob[i][1]:.3f} "
        f"H: {y_prob[i][2]:.3f}"
    )

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print()
print("==============================")
print(f"Accuracy: {accuracy:.3f}")
print("==============================")

print()
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# Save model
with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model saved to model/model.pkl")