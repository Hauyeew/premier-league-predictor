import pandas as pd


def create_features(df, window=5):
    df = df.copy()

    # Make sure dates are actual dates
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Overall team history
    team_history = {}

    # Home-only history
    home_history = {}

    # Away-only history
    away_history = {}

    feature_rows = []

    # Process one DATE at a time.
    # This prevents one match's result from affecting another
    # match played on the same day.
    for date, day_matches in df.groupby("Date", sort=True):

        day_features = []

        # --------------------------------------------------
        # CREATE FEATURES USING ONLY HISTORY BEFORE THIS DATE
        # --------------------------------------------------

        for _, match in day_matches.iterrows():

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            # Initialize histories
            for team in [home_team, away_team]:

                if team not in team_history:
                    team_history[team] = []

                if team not in home_history:
                    home_history[team] = []

                if team not in away_history:
                    away_history[team] = []

            # Last 5 overall matches
            home_recent = team_history[home_team][-window:]
            away_recent = team_history[away_team][-window:]

            # Last 5 home matches
            home_home_recent = home_history[home_team][-window:]

            # Last 5 away matches
            away_away_recent = away_history[away_team][-window:]

            # We only require 5 overall matches.
            # Home/away history can have fewer than 5.
            if len(home_recent) >= window and len(away_recent) >= window:

                # -------------------------
                # Overall statistics
                # -------------------------

                home_goals_scored = (
                    sum(x["goals_scored"] for x in home_recent)
                    / len(home_recent)
                )

                home_goals_conceded = (
                    sum(x["goals_conceded"] for x in home_recent)
                    / len(home_recent)
                )

                away_goals_scored = (
                    sum(x["goals_scored"] for x in away_recent)
                    / len(away_recent)
                )

                away_goals_conceded = (
                    sum(x["goals_conceded"] for x in away_recent)
                    / len(away_recent)
                )

                home_shots = (
                    sum(x["shots"] for x in home_recent)
                    / len(home_recent)
                )

                away_shots = (
                    sum(x["shots"] for x in away_recent)
                    / len(away_recent)
                )

                home_shots_on_target = (
                    sum(x["shots_on_target"] for x in home_recent)
                    / len(home_recent)
                )

                away_shots_on_target = (
                    sum(x["shots_on_target"] for x in away_recent)
                    / len(away_recent)
                )

                # -------------------------
                # Recent form
                # -------------------------

                home_form = (
                    sum(x["form"] for x in home_recent)
                    / len(home_recent)
                )

                away_form = (
                    sum(x["form"] for x in away_recent)
                    / len(away_recent)
                )

                # -------------------------
                # Home / away specific stats
                # -------------------------

                if len(home_home_recent) > 0:

                    home_home_form = (
                        sum(x["form"] for x in home_home_recent)
                        / len(home_home_recent)
                    )

                    home_goals_scored_at_home = (
                        sum(
                            x["goals_scored"]
                            for x in home_home_recent
                        )
                        / len(home_home_recent)
                    )

                    home_goals_conceded_at_home = (
                        sum(
                            x["goals_conceded"]
                            for x in home_home_recent
                        )
                        / len(home_home_recent)
                    )

                else:

                    home_home_form = home_form
                    home_goals_scored_at_home = home_goals_scored
                    home_goals_conceded_at_home = home_goals_conceded

                if len(away_away_recent) > 0:

                    away_away_form = (
                        sum(x["form"] for x in away_away_recent)
                        / len(away_away_recent)
                    )

                    away_goals_scored_away = (
                        sum(
                            x["goals_scored"]
                            for x in away_away_recent
                        )
                        / len(away_away_recent)
                    )

                    away_goals_conceded_away = (
                        sum(
                            x["goals_conceded"]
                            for x in away_away_recent
                        )
                        / len(away_away_recent)
                    )

                else:

                    away_away_form = away_form
                    away_goals_scored_away = away_goals_scored
                    away_goals_conceded_away = away_goals_conceded

                # -------------------------
                # Differences
                # -------------------------

                attack_difference = (
                    home_goals_scored
                    - away_goals_scored
                )

                defense_difference = (
                    away_goals_conceded
                    - home_goals_conceded
                )

                form_difference = (
                    home_form
                    - away_form
                )

                home_away_attack_difference = (
                    home_goals_scored_at_home
                    - away_goals_scored_away
                )

                home_away_defense_difference = (
                    away_goals_conceded_away
                    - home_goals_conceded_at_home
                )

                # -------------------------
                # Store features
                # -------------------------

                features = {

                    "Date": match["Date"],

                    "HomeTeam": home_team,

                    "AwayTeam": away_team,

                    # Overall stats
                    "HomeGoalsScored": home_goals_scored,

                    "HomeGoalsConceded": home_goals_conceded,

                    "HomeShots": home_shots,

                    "HomeShotsOnTarget": home_shots_on_target,

                    "AwayGoalsScored": away_goals_scored,

                    "AwayGoalsConceded": away_goals_conceded,

                    "AwayShots": away_shots,

                    "AwayShotsOnTarget": away_shots_on_target,

                    # Form
                    "HomeForm": home_form,

                    "AwayForm": away_form,

                    "FormDifference": form_difference,

                    # Home/away form
                    "HomeHomeForm": home_home_form,

                    "AwayAwayForm": away_away_form,

                    "HomeAwayFormDifference": (
                        home_home_form
                        - away_away_form
                    ),

                    # Home/away goals
                    "HomeGoalsScoredAtHome": (
                        home_goals_scored_at_home
                    ),

                    "HomeGoalsConcededAtHome": (
                        home_goals_conceded_at_home
                    ),

                    "AwayGoalsScoredAway": (
                        away_goals_scored_away
                    ),

                    "AwayGoalsConcededAway": (
                        away_goals_conceded_away
                    ),

                    # Strength differences
                    "AttackDifference": attack_difference,

                    "DefenseDifference": defense_difference,

                    "HomeAwayAttackDifference": (
                        home_away_attack_difference
                    ),

                    "HomeAwayDefenseDifference": (
                        home_away_defense_difference
                    ),

                    "ShotsDifference": (
                        home_shots
                        - away_shots
                    ),

                    "ShotsOnTargetDifference": (
                        home_shots_on_target
                        - away_shots_on_target
                    ),

                    # Target
                    "Result": match["FTR"]
                }

                day_features.append(features)

        # Add today's features only AFTER ALL today's matches
        # have been created.
        feature_rows.extend(day_features)

        # --------------------------------------------------
        # NOW UPDATE HISTORY WITH TODAY'S RESULTS
        # --------------------------------------------------

        for _, match in day_matches.iterrows():

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            if match["FTR"] == "H":
                home_result = 3
                away_result = 0

            elif match["FTR"] == "D":
                home_result = 1
                away_result = 1

            else:
                home_result = 0
                away_result = 3

            # Overall history
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

            # Home history
            home_history[home_team].append({
                "goals_scored": match["FTHG"],
                "goals_conceded": match["FTAG"],
                "form": home_result
            })

            # Away history
            away_history[away_team].append({
                "goals_scored": match["FTAG"],
                "goals_conceded": match["FTHG"],
                "form": away_result
            })

    return pd.DataFrame(feature_rows)