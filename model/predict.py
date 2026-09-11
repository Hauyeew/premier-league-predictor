import joblib
import pandas as pd

from features.feature_engineering import create_features


MATCHES_FILE = "data/matches.csv"
FIXTURES_FILE = "data/fixtures.csv"
MODEL_FILE = "model/model.pkl"


print("Loading historical matches...")
history = pd.read_csv(MATCHES_FILE)

print(f"Historical matches: {len(history)}")


print()
print("Loading fixtures...")
fixtures = pd.read_csv(FIXTURES_FILE)

print(f"Fixtures: {len(fixtures)}")


# ==========================================
# NORMALIZE TEAM NAMES
# ==========================================

team_name_map = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Hull": "Hull City",
    "Leeds": "Leeds United",
    "Brighton": "Brighton & Hove Albion",
    "Ipswich": "Ipswich Town",
    "Coventry": "Coventry City",
    "Tottenham": "Tottenham Hotspur",
}


history["HomeTeam"] = history["HomeTeam"].replace(team_name_map)
history["AwayTeam"] = history["AwayTeam"].replace(team_name_map)

fixtures["HomeTeam"] = fixtures["HomeTeam"].replace(team_name_map)
fixtures["AwayTeam"] = fixtures["AwayTeam"].replace(team_name_map)


# ==========================================
# DATES
# ==========================================

history["Date"] = pd.to_datetime(history["Date"])
fixtures["Date"] = pd.to_datetime(fixtures["Date"])


# ==========================================
# REMOVE FIXTURES THAT HAVE ALREADY BEEN PLAYED
# ==========================================

history_keys = set(
    zip(
        history["Date"].dt.date,
        history["HomeTeam"],
        history["AwayTeam"]
    )
)

fixtures["match_key"] = list(
    zip(
        fixtures["Date"].dt.date,
        fixtures["HomeTeam"],
        fixtures["AwayTeam"]
    )
)

upcoming = fixtures[
    ~fixtures["match_key"].isin(history_keys)
].copy()

upcoming = upcoming.drop(columns=["match_key"])

# Only predict the next upcoming matchweek
next_matchweek = upcoming["Matchweek"].min()

upcoming = upcoming[
    upcoming["Matchweek"] == next_matchweek
].copy()

print()
print(f"Predicting Matchweek {next_matchweek}")
print(f"Matches: {len(upcoming)}")


print()
print(f"Upcoming matches: {len(upcoming)}")


# ==========================================
# CREATE EMPTY COLUMNS NEEDED BY FEATURES
# ==========================================

upcoming["FTR"] = pd.NA

upcoming["FTHG"] = pd.NA
upcoming["FTAG"] = pd.NA

upcoming["HS"] = pd.NA
upcoming["AS"] = pd.NA

upcoming["HST"] = pd.NA
upcoming["AST"] = pd.NA


# ==========================================
# COMBINE HISTORICAL + UPCOMING
# ==========================================

combined = pd.concat(
    [
        history,
        upcoming
    ],
    ignore_index=True,
    sort=False
)


# ==========================================
# CREATE FEATURES
# ==========================================

print()
print("Creating features...")

features = create_features(combined)

print(
    f"Created {len(features)} feature rows"
)


# ==========================================
# IDENTIFY UPCOMING MATCHES
# ==========================================

upcoming_keys = set(
    zip(
        upcoming["Date"].dt.date,
        upcoming["HomeTeam"],
        upcoming["AwayTeam"]
    )
)

features["match_key"] = list(
    zip(
        features["Date"],
        features["HomeTeam"],
        features["AwayTeam"]
    )
)

prediction_features = features[
    features["match_key"].isin(upcoming_keys)
].copy()

prediction_features = prediction_features.drop(
    columns=["match_key"]
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print()
print("Loading trained model...")

saved_model = joblib.load(MODEL_FILE)

model = saved_model["model"]
label_encoder = saved_model["label_encoder"]
feature_columns = saved_model["feature_columns"]


# ==========================================
# PREPARE FEATURES
# ==========================================

X_future = prediction_features[
    feature_columns
]


# ==========================================
# PREDICT PROBABILITIES
# ==========================================

probabilities = model.predict_proba(
    X_future
)


# ==========================================
# ADD PROBABILITIES
# ==========================================

for i, class_name in enumerate(
    label_encoder.classes_
):
    prediction_features[
        f"Probability_{class_name}"
    ] = probabilities[:, i]


# ==========================================
# PREDICTED RESULT
# ==========================================

predicted_classes = model.predict(
    X_future
)

prediction_features["Prediction"] = (
    label_encoder.inverse_transform(
        predicted_classes
    )
)


# ==========================================
# DISPLAY
# ==========================================

print()
print("==============================")
print("UPCOMING PREMIER LEAGUE GAMES")
print("==============================")


for _, row in prediction_features.iterrows():

    print()
    print(
        f"{row['Date']} | "
        f"{row['HomeTeam']} vs "
        f"{row['AwayTeam']}"
    )

    print(
        f"  Home: "
        f"{row['Probability_H'] * 100:.1f}%"
    )

    print(
        f"  Draw: "
        f"{row['Probability_D'] * 100:.1f}%"
    )

    print(
        f"  Away: "
        f"{row['Probability_A'] * 100:.1f}%"
    )

    print(
        f"  Prediction: "
        f"{row['Prediction']}"
    )


# ==========================================
# SAVE PREDICTIONS
# ==========================================

prediction_features.to_csv(
    "data/predictions.csv",
    index=False
)

print()
print("==============================")
print("Predictions saved to:")
print("data/predictions.csv")
print("==============================")