import pandas as pd


def create_features(df, window=5):

    df = df.copy()

    df = df.sort_values("Date").reset_index(drop=True)

    # Statistics that we'll track for each team
    stats = {
        "goals_scored": "FTHG",
        "goals_conceded": "FTAG",
        "shots": "HS",
        "shots_on_target": "HST"
    }

    # Store each team's previous statistics
    team_history = {}

    feature_rows = []

    for _, match in df.iterrows():

        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        # Create history if team doesn't exist
        if home_team not in team_history:
            team_history[home_team] = []

        if away_team not in team_history:
            team_history[away_team] = []

        home_history = team_history[home_team]
        away_history = team_history[away_team]

        # Need previous games before making a prediction
        if len(home_history) >= window and len(away_history) >= window:

            home_recent = home_history[-window:]
            away_recent = away_history[-window:]

            features = {
                "Date": match["Date"],
                "HomeTeam": home_team,
                "AwayTeam": away_team,

                # Home team's recent stats
                "HomeGoalsScored": sum(
                    x["goals_scored"] for x in home_recent
                ) / window,

                "HomeGoalsConceded": sum(
                    x["goals_conceded"] for x in home_recent
                ) / window,

                "HomeShots": sum(
                    x["shots"] for x in home_recent
                ) / window,

                "HomeShotsOnTarget": sum(
                    x["shots_on_target"] for x in home_recent
                ) / window,

                # Away team's recent stats
                "AwayGoalsScored": sum(
                    x["goals_scored"] for x in away_recent
                ) / window,

                "AwayGoalsConceded": sum(
                    x["goals_conceded"] for x in away_recent
                ) / window,

                "AwayShots": sum(
                    x["shots"] for x in away_recent
                ) / window,

                "AwayShotsOnTarget": sum(
                    x["shots_on_target"] for x in away_recent
                ) / window,

                # Differences
                "GoalsScoredDifference": (
                    sum(x["goals_scored"] for x in home_recent) / window
                    -
                    sum(x["goals_scored"] for x in away_recent) / window
                ),

                "GoalsConcededDifference": (
                    sum(x["goals_conceded"] for x in away_recent) / window
                    -
                    sum(x["goals_conceded"] for x in home_recent) / window
                ),

                "ShotsDifference": (
                    sum(x["shots"] for x in home_recent) / window
                    -
                    sum(x["shots"] for x in away_recent) / window
                ),

                "ShotsOnTargetDifference": (
                    sum(x["shots_on_target"] for x in home_recent) / window
                    -
                    sum(x["shots_on_target"] for x in away_recent) / window
                ),

                # Target
                "Result": match["FTR"]
            }

            feature_rows.append(features)

        # After prediction features are created,
        # add THIS match to the team's history.

        home_history.append({
            "goals_scored": match["FTHG"],
            "goals_conceded": match["FTAG"],
            "shots": match["HS"],
            "shots_on_target": match["HST"]
        })

        away_history.append({
            "goals_scored": match["FTAG"],
            "goals_conceded": match["FTHG"],
            "shots": match["AS"],
            "shots_on_target": match["AST"]
        })

    return pd.DataFrame(feature_rows)