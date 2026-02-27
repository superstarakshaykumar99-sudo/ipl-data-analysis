"""
data_cleaning.py
----------------
Load, validate and preprocess IPL raw datasets.
"""

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Expected columns for basic validation
EXPECTED_MATCHES_COLS = {"id", "season", "city", "date", "team1", "team2",
                         "toss_winner", "toss_decision", "result", "winner", "venue"}
EXPECTED_DELIVERIES_COLS = {"match_id", "inning", "batting_team", "bowling_team",
                             "over", "ball", "batter", "bowler",
                             "batsman_runs", "extra_runs", "total_runs"}

# Historical name aliases → canonical team name
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Pune Warriors": "Pune Warriors India",
}


def load_data(matches_path: str, deliveries_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches and deliveries datasets from CSV files.

    Args:
        matches_path: Path to matches.csv
        deliveries_path: Path to deliveries.csv

    Returns:
        Tuple of (matches DataFrame, deliveries DataFrame)
    """
    logger.info("Loading matches from: %s", matches_path)
    matches = pd.read_csv(matches_path)
    logger.info("Loading deliveries from: %s", deliveries_path)
    deliveries = pd.read_csv(deliveries_path)
    logger.info("Loaded %d matches and %d delivery records.", len(matches), len(deliveries))
    return matches, deliveries


def validate_schema(df: pd.DataFrame, expected_cols: set, name: str) -> None:
    """Warn if expected columns are missing from a DataFrame.

    Args:
        df: DataFrame to validate
        expected_cols: Set of column names that should be present
        name: Human-readable name for logging
    """
    missing = expected_cols - set(df.columns)
    if missing:
        logger.warning("[%s] Missing expected columns: %s", name, missing)
    else:
        logger.info("[%s] Schema validation passed.", name)


def normalize_team_names(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Replace historical franchise names with current canonical names.

    Args:
        df: DataFrame containing team name columns
        columns: List of column names to normalize

    Returns:
        DataFrame with normalized team names
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the matches DataFrame.

    Steps:
        1. Validate schema
        2. Drop duplicate rows
        3. Drop rows without a winner (super-overs / no result)
        4. Parse date → datetime
        5. Extract season (year) if missing
        6. Normalize franchise team names
        7. Strip whitespace from string columns

    Args:
        matches: Raw matches DataFrame

    Returns:
        Cleaned matches DataFrame
    """
    validate_schema(matches, EXPECTED_MATCHES_COLS, "matches")

    before = len(matches)
    matches = matches.drop_duplicates()
    matches = matches.dropna(subset=["winner"])
    logger.info("Dropped %d rows (duplicates/no-result). Remaining: %d", before - len(matches), len(matches))

    # Parse dates & extract season
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    # Handle season values like '2007/08' or '2008' — extract leading 4-digit year
    if "season" in matches.columns:
        matches["season"] = (
            matches["season"].astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .astype("Int64")
        )
    else:
        matches["season"] = matches["date"].dt.year.astype("Int64")

    # Normalize team names
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    matches = normalize_team_names(matches, team_cols)

    # Strip whitespace from object columns
    str_cols = matches.select_dtypes(include="object").columns
    for col in str_cols:
        matches[col] = matches[col].str.strip()

    return matches


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the deliveries DataFrame.

    Steps:
        1. Validate schema
        2. Drop duplicate rows
        3. Fill missing numeric values with 0
        4. Normalize team names
        5. Strip whitespace from string columns

    Args:
        deliveries: Raw deliveries DataFrame

    Returns:
        Cleaned deliveries DataFrame
    """
    validate_schema(deliveries, EXPECTED_DELIVERIES_COLS, "deliveries")

    deliveries = deliveries.drop_duplicates()

    numeric_cols = deliveries.select_dtypes(include="number").columns
    deliveries[numeric_cols] = deliveries[numeric_cols].fillna(0)

    team_cols = [c for c in ["batting_team", "bowling_team"] if c in deliveries.columns]
    deliveries = normalize_team_names(deliveries, team_cols)

    str_cols = deliveries.select_dtypes(include="object").columns
    for col in str_cols:
        deliveries[col] = deliveries[col].str.strip()

    logger.info("Deliveries cleaned. Shape: %s", deliveries.shape)
    return deliveries


if __name__ == "__main__":
    matches, deliveries = load_data("data/matches.csv", "data/deliveries.csv")
    matches = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)
    print("Matches shape:", matches.shape)
    print("Deliveries shape:", deliveries.shape)
    print(matches.dtypes)
