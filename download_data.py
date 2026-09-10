import pandas as pd

# Latest 10 completed Premier League seasons
seasons = [
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526"
]

all_matches = []

for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

    print(f"Downloading {season}...")

    try:
        df = pd.read_csv(url)

        df["Season"] = season

        all_matches.append(df)

        print(f"  ✓ {len(df)} matches downloaded")

    except Exception as e:
        print(f"  ✗ Could not download {season}")
        print(f"    Error: {e}")


if not all_matches:
    raise RuntimeError("No seasons were downloaded.")


# Combine all seasons
matches = pd.concat(all_matches, ignore_index=True)

# Convert dates
matches["Date"] = pd.to_datetime(
    matches["Date"],
    dayfirst=True,
    errors="coerce"
)

# Sort chronologically
matches = matches.sort_values("Date")

# Remove completely empty columns
matches = matches.dropna(axis=1, how="all")

# Save
matches.to_csv("data/matches.csv", index=False)

print()
print("===================================")
print("Finished!")
print(f"Total matches: {len(matches)}")
print(f"Total columns: {len(matches.columns)}")
print("Saved as: data/matches.csv")
print("===================================")