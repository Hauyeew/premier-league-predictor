import os
import pandas as pd

from understatapi import UnderstatClient


# ==========================================
# SETTINGS
# ==========================================

SEASONS = [
    2016,  # 2016/17
    2017,  # 2017/18
    2018,  # 2018/19
    2019,  # 2019/20
    2020,  # 2020/21
    2021,  # 2021/22
    2022,  # 2022/23
    2023,  # 2023/24
    2024,  # 2024/25
    2025   # 2025/26
]

OUTPUT_FILE = "data/xg.csv"


# ==========================================
# DOWNLOAD
# ==========================================

all_matches = []

print("Starting xG download...")
print()


with UnderstatClient() as understat:

    for season in SEASONS:

        season_name = f"{season}/{str(season + 1)[-2:]}"

        print(f"Downloading {season_name}...")

        try:

            # Get every Premier League fixture
            # for this season.
            matches = understat.league(
                league="EPL"
            ).get_match_data(
                season=str(season)
            )

            print(f"  Found {len(matches)} matches")

            # ==========================================
            # EXTRACT MATCH DATA
            # ==========================================

            for match in matches:

                home = match.get("h", {})
                away = match.get("a", {})
                xg = match.get("xG", {})

                all_matches.append({
                    "Season": season_name,

                    "Date": match.get("datetime"),

                    "HomeTeam": home.get("title"),

                    "AwayTeam": away.get("title"),

                    "HomeXG": xg.get("h"),

                    "AwayXG": xg.get("a"),

                    "MatchID": match.get("id")
                })

            print(
                f"  Added {len(matches)} matches"
            )

        except Exception as e:

            print(
                f"  ERROR downloading {season_name}: {e}"
            )

        print()


# ==========================================
# CREATE DATAFRAME
# ==========================================

xg_df = pd.DataFrame(all_matches)


if xg_df.empty:

    raise RuntimeError(
        "No xG data was downloaded."
    )


# ==========================================
# CLEAN DATA
# ==========================================

xg_df["Date"] = pd.to_datetime(
    xg_df["Date"],
    errors="coerce"
)

xg_df["HomeXG"] = pd.to_numeric(
    xg_df["HomeXG"],
    errors="coerce"
)

xg_df["AwayXG"] = pd.to_numeric(
    xg_df["AwayXG"],
    errors="coerce"
)


# Remove incomplete matches
xg_df = xg_df.dropna(
    subset=[
        "Date",
        "HomeTeam",
        "AwayTeam",
        "HomeXG",
        "AwayXG"
    ]
)


# Remove duplicate MatchIDs
xg_df = xg_df.drop_duplicates(
    subset=["MatchID"]
)


# Sort chronologically
xg_df = xg_df.sort_values(
    "Date"
).reset_index(drop=True)


# ==========================================
# CREATE DATA DIRECTORY
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)


# ==========================================
# SAVE
# ==========================================

xg_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("==============================")
print("Download complete!")
print("==============================")

print(
    f"Total matches: {len(xg_df)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()

print("Columns:")
print(
    list(xg_df.columns)
)

print()

print("First 10 matches:")
print(
    xg_df.head(10).to_string(index=False)
)