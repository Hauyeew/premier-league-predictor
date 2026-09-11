
import pandas as pd


def create_features(
    df,
    window=5,
    venue_window=10,
    h2h_window=5,
    xg_window=5
):
    df = df.copy()

    # ============================================================
    # LOAD AND MERGE xG DATA
    # ============================================================

    xg_df = pd.read_csv("data/xg.csv")

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    xg_df["Date"] = pd.to_datetime(xg_df["Date"]).dt.date

    # ============================================================
    # NORMALIZE TEAM NAMES
    # ============================================================

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
        "AFC Bournemouth": "Bournemouth",
        "Bournemouth": "Bournemouth",

        "Brentford": "Brentford",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Southampton": "Southampton",
        "Arsenal": "Arsenal",
        "Liverpool": "Liverpool",
        "Chelsea": "Chelsea",
        "Burnley": "Burnley",
        "Watford": "Watford",
        "Leicester": "Leicester",
        "Norwich": "Norwich",
        "Sheffield United": "Sheffield United",
        "Swansea": "Swansea",
        "Stoke": "Stoke",
        "Middlesbrough": "Middlesbrough",
    }

    df["HomeTeam"] = df["HomeTeam"].replace(team_name_map)
    df["AwayTeam"] = df["AwayTeam"].replace(team_name_map)

    xg_df["HomeTeam"] = xg_df["HomeTeam"].replace(team_name_map)
    xg_df["AwayTeam"] = xg_df["AwayTeam"].replace(team_name_map)

    # ============================================================
    # MERGE xG
    # ============================================================

    df = df.merge(
        xg_df[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "HomeXG",
                "AwayXG"
            ]
        ],
        on=[
            "Date",
            "HomeTeam",
            "AwayTeam"
        ],
        how="left"
    )

    df["HomeXG"] = pd.to_numeric(
        df["HomeXG"],
        errors="coerce"
    )

    df["AwayXG"] = pd.to_numeric(
        df["AwayXG"],
        errors="coerce"
    )

    # ============================================================
    # SORT BY DATE
    # ============================================================

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # ============================================================
    # HISTORIES
    # ============================================================

    team_history = {}
    home_history = {}
    away_history = {}

    h2h_history = {}

    xg_history = {}
    home_xg_history = {}
    away_xg_history = {}

    feature_rows = []

    # ============================================================
    # PROCESS ONE DATE AT A TIME
    # ============================================================

    for date, day_matches in df.groupby(
        "Date",
        sort=True
    ):

        day_features = []

        # ========================================================
        # CREATE FEATURES USING ONLY PAST MATCHES
        # ========================================================

        for _, match in day_matches.iterrows():

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            # ----------------------------------------------------
            # INITIALIZE HISTORIES
            # ----------------------------------------------------

            for team in [
                home_team,
                away_team
            ]:

                if team not in team_history:
                    team_history[team] = []

                if team not in home_history:
                    home_history[team] = []

                if team not in away_history:
                    away_history[team] = []

                if team not in xg_history:
                    xg_history[team] = []

                if team not in home_xg_history:
                    home_xg_history[team] = []

                if team not in away_xg_history:
                    away_xg_history[team] = []

            h2h_key = tuple(
                sorted(
                    [home_team, away_team]
                )
            )

            if h2h_key not in h2h_history:
                h2h_history[h2h_key] = []

            # ----------------------------------------------------
            # RECENT MATCHES
            #
            # IMPORTANT:
            # We now allow fewer than 5 matches.
            # If Coventry has 3 matches, those 3 are used.
            # ----------------------------------------------------

            home_recent = team_history[
                home_team
            ][-window:]

            away_recent = team_history[
                away_team
            ][-window:]

            home_home_recent = home_history[
                home_team
            ][-venue_window:]

            away_away_recent = away_history[
                away_team
            ][-venue_window:]

            recent_h2h = h2h_history[
                h2h_key
            ][-h2h_window:]

            home_xg_recent = xg_history[
                home_team
            ][-xg_window:]

            away_xg_recent = xg_history[
                away_team
            ][-xg_window:]

            home_xg_at_home_recent = home_xg_history[
                home_team
            ][-venue_window:]

            away_xg_at_away_recent = away_xg_history[
                away_team
            ][-venue_window:]

            # ----------------------------------------------------
            # REQUIRE AT LEAST ONE MATCH FOR EACH TEAM
            #
            # This is the key change.
            #
            # Previously:
            #
            # len(home_recent) >= 5
            # AND
            # len(away_recent) >= 5
            #
            # Now:
            #
            # len(home_recent) >= 1
            # AND
            # len(away_recent) >= 1
            # ----------------------------------------------------

            if (
                len(home_recent) >= 1
                and len(away_recent) >= 1
            ):

                # =================================================
                # OVERALL TEAM STATS
                # =================================================

                home_goals_scored = sum(
                    x["goals_scored"]
                    for x in home_recent
                ) / len(home_recent)

                home_goals_conceded = sum(
                    x["goals_conceded"]
                    for x in home_recent
                ) / len(home_recent)

                away_goals_scored = sum(
                    x["goals_scored"]
                    for x in away_recent
                ) / len(away_recent)

                away_goals_conceded = sum(
                    x["goals_conceded"]
                    for x in away_recent
                ) / len(away_recent)

                home_shots = sum(
                    x["shots"]
                    for x in home_recent
                ) / len(home_recent)

                away_shots = sum(
                    x["shots"]
                    for x in away_recent
                ) / len(away_recent)

                home_shots_on_target = sum(
                    x["shots_on_target"]
                    for x in home_recent
                ) / len(home_recent)

                away_shots_on_target = sum(
                    x["shots_on_target"]
                    for x in away_recent
                ) / len(away_recent)

                home_form = sum(
                    x["form"]
                    for x in home_recent
                ) / len(home_recent)

                away_form = sum(
                    x["form"]
                    for x in away_recent
                ) / len(away_recent)

                # =================================================
                # HOME TEAM STRENGTH
                # =================================================

                if len(home_home_recent) > 0:

                    home_home_form = sum(
                        x["form"]
                        for x in home_home_recent
                    ) / len(home_home_recent)

                    home_goals_scored_at_home = sum(
                        x["goals_scored"]
                        for x in home_home_recent
                    ) / len(home_home_recent)

                    home_goals_conceded_at_home = sum(
                        x["goals_conceded"]
                        for x in home_home_recent
                    ) / len(home_home_recent)

                    home_win_rate = sum(
                        1
                        for x in home_home_recent
                        if x["result"] == "W"
                    ) / len(home_home_recent)

                    home_points_per_game = sum(
                        x["points"]
                        for x in home_home_recent
                    ) / len(home_home_recent)

                else:

                    home_home_form = home_form

                    home_goals_scored_at_home = (
                        home_goals_scored
                    )

                    home_goals_conceded_at_home = (
                        home_goals_conceded
                    )

                    home_win_rate = sum(
                        1
                        for x in home_recent
                        if x["result"] == "W"
                    ) / len(home_recent)

                    home_points_per_game = sum(
                        x["points"]
                        for x in home_recent
                    ) / len(home_recent)

                # =================================================
                # AWAY TEAM STRENGTH
                # =================================================

                if len(away_away_recent) > 0:

                    away_away_form = sum(
                        x["form"]
                        for x in away_away_recent
                    ) / len(away_away_recent)

                    away_goals_scored_away = sum(
                        x["goals_scored"]
                        for x in away_away_recent
                    ) / len(away_away_recent)

                    away_goals_conceded_away = sum(
                        x["goals_conceded"]
                        for x in away_away_recent
                    ) / len(away_away_recent)

                    away_win_rate = sum(
                        1
                        for x in away_away_recent
                        if x["result"] == "W"
                    ) / len(away_away_recent)

                    away_points_per_game = sum(
                        x["points"]
                        for x in away_away_recent
                    ) / len(away_away_recent)

                else:

                    away_away_form = away_form

                    away_goals_scored_away = (
                        away_goals_scored
                    )

                    away_goals_conceded_away = (
                        away_goals_conceded
                    )

                    away_win_rate = sum(
                        1
                        for x in away_recent
                        if x["result"] == "W"
                    ) / len(away_recent)

                    away_points_per_game = sum(
                        x["points"]
                        for x in away_recent
                    ) / len(away_recent)

                # =================================================
                # HISTORICAL xG
                # =================================================

                if len(home_xg_recent) > 0:

                    home_xg_for = sum(
                        x["xg_for"]
                        for x in home_xg_recent
                    ) / len(home_xg_recent)

                    home_xg_against = sum(
                        x["xg_against"]
                        for x in home_xg_recent
                    ) / len(home_xg_recent)

                else:

                    home_xg_for = 0
                    home_xg_against = 0

                if len(away_xg_recent) > 0:

                    away_xg_for = sum(
                        x["xg_for"]
                        for x in away_xg_recent
                    ) / len(away_xg_recent)

                    away_xg_against = sum(
                        x["xg_against"]
                        for x in away_xg_recent
                    ) / len(away_xg_recent)

                else:

                    away_xg_for = 0
                    away_xg_against = 0

                # =================================================
                # VENUE-SPECIFIC xG
                # =================================================

                if len(home_xg_at_home_recent) > 0:

                    home_xg_at_home = sum(
                        x["xg_for"]
                        for x in home_xg_at_home_recent
                    ) / len(home_xg_at_home_recent)

                    home_xga_at_home = sum(
                        x["xg_against"]
                        for x in home_xg_at_home_recent
                    ) / len(home_xg_at_home_recent)

                else:

                    home_xg_at_home = home_xg_for
                    home_xga_at_home = home_xg_against

                if len(away_xg_at_away_recent) > 0:

                    away_xg_at_away = sum(
                        x["xg_for"]
                        for x in away_xg_at_away_recent
                    ) / len(away_xg_at_away_recent)

                    away_xga_at_away = sum(
                        x["xg_against"]
                        for x in away_xg_at_away_recent
                    ) / len(away_xg_at_away_recent)

                else:

                    away_xg_at_away = away_xg_for
                    away_xga_at_away = away_xg_against

                xg_attack_difference = (
                    home_xg_for
                    - away_xg_for
                )

                xg_defense_difference = (
                    away_xg_against
                    - home_xg_against
                )

                venue_xg_attack_difference = (
                    home_xg_at_home
                    - away_xg_at_away
                )

                venue_xg_defense_difference = (
                    away_xga_at_away
                    - home_xga_at_home
                )

                # =================================================
                # HEAD-TO-HEAD
                # =================================================

                if len(recent_h2h) > 0:

                    home_h2h_wins = 0
                    draws = 0
                    away_h2h_wins = 0

                    home_h2h_goals = 0
                    away_h2h_goals = 0

                    for h2h in recent_h2h:

                        if h2h["home_team"] == home_team:

                            home_h2h_goals += (
                                h2h["home_goals"]
                            )

                            away_h2h_goals += (
                                h2h["away_goals"]
                            )

                            if h2h["result"] == "H":

                                home_h2h_wins += 1

                            elif h2h["result"] == "D":

                                draws += 1

                            else:

                                away_h2h_wins += 1

                        else:

                            home_h2h_goals += (
                                h2h["away_goals"]
                            )

                            away_h2h_goals += (
                                h2h["home_goals"]
                            )

                            if h2h["result"] == "A":

                                home_h2h_wins += 1

                            elif h2h["result"] == "D":

                                draws += 1

                            else:

                                away_h2h_wins += 1

                    h2h_meetings = len(
                        recent_h2h
                    )

                    h2h_home_win_rate = (
                        home_h2h_wins
                        / h2h_meetings
                    )

                    h2h_draw_rate = (
                        draws
                        / h2h_meetings
                    )

                    h2h_away_win_rate = (
                        away_h2h_wins
                        / h2h_meetings
                    )

                    h2h_goals_for = (
                        home_h2h_goals
                        / h2h_meetings
                    )

                    h2h_goals_against = (
                        away_h2h_goals
                        / h2h_meetings
                    )

                    h2h_goal_difference = (
                        h2h_goals_for
                        - h2h_goals_against
                    )

                else:

                    h2h_home_win_rate = 1 / 3
                    h2h_draw_rate = 1 / 3
                    h2h_away_win_rate = 1 / 3

                    h2h_goals_for = 0
                    h2h_goals_against = 0
                    h2h_goal_difference = 0

                    h2h_meetings = 0

                # =================================================
                # DIFFERENCES
                # =================================================

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

                shots_difference = (
                    home_shots
                    - away_shots
                )

                shots_on_target_difference = (
                    home_shots_on_target
                    - away_shots_on_target
                )

                home_win_rate_difference = (
                    home_win_rate
                    - away_win_rate
                )

                points_per_game_difference = (
                    home_points_per_game
                    - away_points_per_game
                )

                # =================================================
                # FEATURE ROW
                # =================================================

                feature_row = {

                    "Date": match["Date"],
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,

                    "HomeGoalsScored":
                        home_goals_scored,

                    "HomeGoalsConceded":
                        home_goals_conceded,

                    "HomeShots":
                        home_shots,

                    "HomeShotsOnTarget":
                        home_shots_on_target,

                    "AwayGoalsScored":
                        away_goals_scored,

                    "AwayGoalsConceded":
                        away_goals_conceded,

                    "AwayShots":
                        away_shots,

                    "AwayShotsOnTarget":
                        away_shots_on_target,

                    "HomeForm":
                        home_form,

                    "AwayForm":
                        away_form,

                    "FormDifference":
                        form_difference,

                    "HomeHomeForm":
                        home_home_form,

                    "AwayAwayForm":
                        away_away_form,

                    "HomeAwayFormDifference":
                        (
                            home_home_form
                            - away_away_form
                        ),

                    "HomeGoalsScoredAtHome":
                        home_goals_scored_at_home,

                    "HomeGoalsConcededAtHome":
                        home_goals_conceded_at_home,

                    "AwayGoalsScoredAway":
                        away_goals_scored_away,

                    "AwayGoalsConcededAway":
                        away_goals_conceded_away,

                    "HomeWinRate":
                        home_win_rate,

                    "AwayWinRate":
                        away_win_rate,

                    "HomePointsPerGame":
                        home_points_per_game,

                    "AwayPointsPerGame":
                        away_points_per_game,

                    "HomeWinRateDifference":
                        home_win_rate_difference,

                    "PointsPerGameDifference":
                        points_per_game_difference,

                    "AttackDifference":
                        attack_difference,

                    "DefenseDifference":
                        defense_difference,

                    "HomeAwayAttackDifference":
                        home_away_attack_difference,

                    "HomeAwayDefenseDifference":
                        home_away_defense_difference,

                    "ShotsDifference":
                        shots_difference,

                    "ShotsOnTargetDifference":
                        shots_on_target_difference,

                    "HomeXG":
                        home_xg_for,

                    "HomeXGA":
                        home_xg_against,

                    "AwayXG":
                        away_xg_for,

                    "AwayXGA":
                        away_xg_against,

                    "XGAttackDifference":
                        xg_attack_difference,

                    "XGDefenseDifference":
                        xg_defense_difference,

                    "HomeXGAtHome":
                        home_xg_at_home,

                    "HomeXGAAtHome":
                        home_xga_at_home,

                    "AwayXGAtAway":
                        away_xg_at_away,

                    "AwayXGAAtAway":
                        away_xga_at_away,

                    "VenueXGAttackDifference":
                        venue_xg_attack_difference,

                    "VenueXGDefenseDifference":
                        venue_xg_defense_difference,

                    "H2HHomeWinRate":
                        h2h_home_win_rate,

                    "H2HDrawRate":
                        h2h_draw_rate,

                    "H2HAwayWinRate":
                        h2h_away_win_rate,

                    "H2HGoalsFor":
                        h2h_goals_for,

                    "H2HGoalsAgainst":
                        h2h_goals_against,

                    "H2HGoalDifference":
                        h2h_goal_difference,

                    "H2HMeetings":
                        h2h_meetings,

                    "Result":
                        match["FTR"]
                }

                day_features.append(
                    feature_row
                )

        # Add today's feature rows
        feature_rows.extend(
            day_features
        )

        # ========================================================
        # UPDATE HISTORIES
        #
        # IMPORTANT:
        # UPCOMING MATCHES ARE NOT ADDED TO HISTORY.
        # ========================================================

        for _, match in day_matches.iterrows():

            # ----------------------------------------------------
            # Skip upcoming matches
            # ----------------------------------------------------

            if pd.isna(match["FTR"]):
                continue

            home_team = match["HomeTeam"]
            away_team = match["AwayTeam"]

            # ----------------------------------------------------
            # DETERMINE RESULT
            # ----------------------------------------------------

            if match["FTR"] == "H":

                home_result = "W"
                away_result = "L"

                home_points = 3
                away_points = 0

            elif match["FTR"] == "D":

                home_result = "D"
                away_result = "D"

                home_points = 1
                away_points = 1

            else:

                home_result = "L"
                away_result = "W"

                home_points = 0
                away_points = 3

            # ----------------------------------------------------
            # OVERALL HISTORY
            # ----------------------------------------------------

            team_history[home_team].append({

                "goals_scored":
                    match["FTHG"],

                "goals_conceded":
                    match["FTAG"],

                "shots":
                    match["HS"],

                "shots_on_target":
                    match["HST"],

                "form":
                    home_points,

                "result":
                    home_result,

                "points":
                    home_points
            })

            team_history[away_team].append({

                "goals_scored":
                    match["FTAG"],

                "goals_conceded":
                    match["FTHG"],

                "shots":
                    match["AS"],

                "shots_on_target":
                    match["AST"],

                "form":
                    away_points,

                "result":
                    away_result,

                "points":
                    away_points
            })

            # ----------------------------------------------------
            # HOME HISTORY
            # ----------------------------------------------------

            home_history[home_team].append({

                "goals_scored":
                    match["FTHG"],

                "goals_conceded":
                    match["FTAG"],

                "form":
                    home_points,

                "result":
                    home_result,

                "points":
                    home_points
            })

            # ----------------------------------------------------
            # AWAY HISTORY
            # ----------------------------------------------------

            away_history[away_team].append({

                "goals_scored":
                    match["FTAG"],

                "goals_conceded":
                    match["FTHG"],

                "form":
                    away_points,

                "result":
                    away_result,

                "points":
                    away_points
            })

            # ----------------------------------------------------
            # xG HISTORY
            # ----------------------------------------------------

            if (
                pd.notna(match["HomeXG"])
                and pd.notna(match["AwayXG"])
            ):

                home_xg_history[
                    home_team
                ].append({

                    "xg_for":
                        match["HomeXG"],

                    "xg_against":
                        match["AwayXG"]
                })

                away_xg_history[
                    away_team
                ].append({

                    "xg_for":
                        match["AwayXG"],

                    "xg_against":
                        match["HomeXG"]
                })

                xg_history[
                    home_team
                ].append({

                    "xg_for":
                        match["HomeXG"],

                    "xg_against":
                        match["AwayXG"]
                })

                xg_history[
                    away_team
                ].append({

                    "xg_for":
                        match["AwayXG"],

                    "xg_against":
                        match["HomeXG"]
                })

            # ----------------------------------------------------
            # H2H HISTORY
            # ----------------------------------------------------

            h2h_key = tuple(
                sorted(
                    [home_team, away_team]
                )
            )

            h2h_history[
                h2h_key
            ].append({

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "home_goals":
                    match["FTHG"],

                "away_goals":
                    match["FTAG"],

                "result":
                    match["FTR"]
            })

    return pd.DataFrame(
        feature_rows
    )

