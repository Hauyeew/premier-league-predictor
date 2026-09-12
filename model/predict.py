import joblib
import pandas as pd

from features.feature_engineering import create_features


MATCHES_FILE = "data/matches.csv"
FIXTURES_FILE = "data/fixtures.csv"
MODEL_FILE = "model/model.pkl"


TEAM_NAME_MAP = {
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
    "AFC Bournemouth": "Bournemouth",
}


def generate_predictions():
    # Load historical matches and fixtures
    history = pd.read_csv(MATCHES_FILE)
    fixtures = pd.read_csv(FIXTURES_FILE)

    # Normalize team names
    history["HomeTeam"] = history["HomeTeam"].replace(TEAM_NAME_MAP)
    history["AwayTeam"] = history["AwayTeam"].replace(TEAM_NAME_MAP)

    fixtures["HomeTeam"] = fixtures["HomeTeam"].replace(TEAM_NAME_MAP)
    fixtures["AwayTeam"] = fixtures["AwayTeam"].replace(TEAM_NAME_MAP)

    # Convert dates
    history["Date"] = pd.to_datetime(history["Date"])
    fixtures["Date"] = pd.to_datetime(fixtures["Date"])

    # Create unique keys for completed historical matches
    history_keys = set(
        zip(
            history["Date"].dt.date,
            history["HomeTeam"],
            history["AwayTeam"],
        )
    )

    # Create keys for fixtures
    fixtures["match_key"] = list(
        zip(
            fixtures["Date"].dt.date,
            fixtures["HomeTeam"],
            fixtures["AwayTeam"],
        )
    )

    # Remove fixtures that have already been played
    upcoming = fixtures[
        ~fixtures["match_key"].isin(history_keys)
    ].copy()

    upcoming = upcoming.drop(
        columns=["match_key"]
    )

    # Find the next upcoming Matchweek
    next_matchweek = upcoming["Matchweek"].min()

    upcoming = upcoming[
        upcoming["Matchweek"] == next_matchweek
    ].copy()

    print(
        f"Predicting Matchweek {next_matchweek}"
    )
    print(
        f"Matches: {len(upcoming)}"
    )

    # Add empty match-result/stat columns
    # so feature_engineering can process
    # upcoming fixtures.
    upcoming["FTR"] = pd.NA
    upcoming["FTHG"] = pd.NA
    upcoming["FTAG"] = pd.NA
    upcoming["HS"] = pd.NA
    upcoming["AS"] = pd.NA
    upcoming["HST"] = pd.NA
    upcoming["AST"] = pd.NA

    # Combine historical matches with
    # upcoming fixtures.
    combined = pd.concat(
        [history, upcoming],
        ignore_index=True,
        sort=False,
    )

    # Create the exact same features
    # used when training the model.
    features = create_features(combined)

    # Create keys for the upcoming matches
    upcoming_keys = set(
        zip(
            upcoming["Date"].dt.date,
            upcoming["HomeTeam"],
            upcoming["AwayTeam"],
        )
    )

    features["match_key"] = list(
        zip(
            features["Date"],
            features["HomeTeam"],
            features["AwayTeam"],
        )
    )

    # Keep only features for upcoming matches
    prediction_features = features[
        features["match_key"].isin(upcoming_keys)
    ].copy()

    prediction_features = prediction_features.drop(
        columns=["match_key"]
    )

    # Load trained model
    saved_model = joblib.load(MODEL_FILE)

    model = saved_model["model"]
    label_encoder = saved_model["label_encoder"]
    feature_columns = saved_model["feature_columns"]

    # Make sure the prediction data uses
    # the exact same columns as training.
    X_future = prediction_features[
        feature_columns
    ]

    # Get probabilities
    probabilities = model.predict_proba(
        X_future
    )

    # Add probabilities
    for i, class_name in enumerate(
        label_encoder.classes_
    ):
        prediction_features[
            f"Probability_{class_name}"
        ] = probabilities[:, i]

    # Get predicted result
    predicted_classes = model.predict(
        X_future
    )

    prediction_features["Prediction"] = (
        label_encoder.inverse_transform(
            predicted_classes
        )
    )

    # Only return the information
    # our website needs.
    results = prediction_features[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "Probability_H",
            "Probability_D",
            "Probability_A",
            "Prediction",
        ]
    ].copy()

    # Convert Date to a string so it can
    # be returned as JSON.
    results["Date"] = results["Date"].astype(str)

    return results.to_dict(
        orient="records"
    )


if __name__ == "__main__":
    predictions = generate_predictions()

    for prediction in predictions:
        print()
        print(
            f"{prediction['Date']} | "
            f"{prediction['HomeTeam']} vs "
            f"{prediction['AwayTeam']}"
        )

        print(
            f"  Home: "
            f"{prediction['Probability_H'] * 100:.1f}%"
        )

        print(
            f"  Draw: "
            f"{prediction['Probability_D'] * 100:.1f}%"
        )

        print(
            f"  Away: "
            f"{prediction['Probability_A'] * 100:.1f}%"
        )

        print(
            f"  Prediction: "
            f"{prediction['Prediction']}"
        )