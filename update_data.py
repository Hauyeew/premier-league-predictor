
import pandas as pd

# ==========================================
# SETTINGS
# ==========================================

SEASON = "2627"
URL = f"https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv"
FILE_PATH = "data/matches.csv"


# ==========================================
# 1. Download newest 2026/27 data
# ==========================================

print(f"Downloading {SEASON}...")

try:
    new_matches = pd.read_csv(URL)
    new_matches["Season"] = SEASON

    print(f"✓ Downloaded {len(new_matches)} matches")

except Exception as e:
    raise RuntimeError(f"Could not download {SEASON}: {e}")


# ==========================================
# 2. Load existing dataset
# ==========================================

print("Loading existing dataset...")

matches = pd.read_csv(FILE_PATH)

print(f"✓ Existing dataset: {len(matches)} matches")


# ==========================================
# 3. Convert dates
# ==========================================

matches["Date"] = pd.to_datetime(
    matches["Date"],
    dayfirst=True,
    errors="coerce"
)

new_matches["Date"] = pd.to_datetime(
    new_matches["Date"],
    dayfirst=True,
    errors="coerce"
)


# ==========================================
# 4. Remove old 2026/27 data
# ==========================================
# This is what makes the script duplicate-safe.
#
# If you already downloaded some 2026/27 games,
# we remove them first and replace them with
# the newest version from Football-Data.

matches = matches[matches["Season"].astype(str) != SEASON]


# ==========================================
# 5. Add the newest 2026/27 data
# ==========================================

matches = pd.concat(
    [matches, new_matches],
    ignore_index=True
)


# ==========================================
# 6. Sort chronologically
# ==========================================

matches = matches.sort_values(
    "Date"
).reset_index(drop=True)


# ==========================================
# 7. Remove completely empty columns
# ==========================================

matches = matches.dropna(
    axis=1,
    how="all"
)


# ==========================================
# 8. Save updated dataset
# ==========================================

matches.to_csv(
    FILE_PATH,
    index=False
)


# ==========================================
# 9. Print summary
# ==========================================

print()
print("===================================")
print("Dataset updated!")
print("===================================")
print(f"2026/27 matches: {len(new_matches)}")
print(f"Total matches:   {len(matches)}")
print(f"Total columns:   {len(matches.columns)}")
print(f"Saved to:        {FILE_PATH}")
print("===================================")
