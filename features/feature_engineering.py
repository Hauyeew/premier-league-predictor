import pandas as pd


def create_features(df, window=5):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep two types of history:
    # 1. Overall history
    # 2. Home/away-specific history
    team_history = {}
    home_history = {}
    away_history = {}

    feature_rows = []

    for _, match in df.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        # Create histories if teams don't exist
        for team in [home_team, away_team]:
            if team not in team_history:
                team_history[team] = []

            if team not in home_history:
                home_history[team] = []

            if team not in away_history:
                away_history[team] = []

        home_recent = team_history[home_team][-window:]
        away_recent = team_history[away_team][-window:]

        home_home_recent = home_history[home_team][-window:]
        away_away_recent = away_history[away_team][-window:]

        # We need enough overall AND home/away matches
        if (
            len(home_recent) >= window
            and len(away_recent) >= window
            and len(home_home_recent) >= window
            and len(away_away_recent) >= window
        ):
            # Overall form
            home_form = sum(x["form"] for x in home_recent) / window
            away_form = sum(x["form"] for x in away_recent) / window

            # Home-specific form
            home_home_form = sum(
                x["form"] for x in home_home_recent
            ) / window

            # Away-specific form
            away_away_form = sum(
                x["form"] for x in away_away_recent
            ) / window

            features = {
                "Date": match["Date"],
                "HomeTeam": home_team,
                "AwayTeam": away_team,

                # Overall recent stats
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

                # Overall form
                "HomeForm": home_form,
                "AwayForm": away_form,
                "FormDifference": home_form - away_form,

                # Home/away-specific form
                "HomeHomeForm": home_home_form,
                "AwayAwayForm": away_away_form,
                "HomeAwayFormDifference": (
                    home_home_form - away_away_form
                ),

                # Home team's recent home performance
                "HomeGoalsScoredAtHome": sum(
                    x["goals_scored"] for x in home_home_recent
                ) / window,

                "HomeGoalsConcededAtHome": sum(
                    x["goals_conceded"] for x in home_home_recent
                ) / window,

                # Away team's recent away performance
                "AwayGoalsScoredAway": sum(
                    x["goals_scored"] for x in away_away_recent
                ) / window,

                "AwayGoalsConcededAway": sum(
                    x["goals_conceded"] for x in away_away_recent
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

        # Determine results AFTER creating features
        if match["FTR"] == "H":
            home_result = 3
            away_result = 0
        elif match["FTR"] == "D":
            home_result = 1
            away_result = 1
        else:
            home_result = 0
            away_result = 3

        # Add to overall history
        team_history[home_team].append({
            "goals_scored": match["FTHG"],
            "goals_conceded": match["FTAG"],
            "shots": match["HS"],
            "shots_on_target": match["HST"],
            "form": home_result
        })

        team_history[away_team].append({
            "goals_scored": match["FTAG"],
            "goals_conceded": match["FTHG"],
            "shots": match["AS"],
            "shots_on_target": match["AST"],
            "form": away_result
        })

        # Add to home-specific history
        home_history[home_team].append({
            "goals_scored": match["FTHG"],
            "goals_conceded": match["FTAG"],
            "form": home_result
        })

        # Add to away-specific history
        away_history[away_team].append({
            "goals_scored": match["FTAG"],
            "goals_conceded": match["FTHG"],
            "form": away_result
        })

    return pd.DataFrame(feature_rows)