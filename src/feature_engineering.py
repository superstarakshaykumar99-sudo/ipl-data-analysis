"""
feature_engineering.py
-----------------------
Create derived features from IPL matches and deliveries data.
"""

import pandas as pd
import numpy as np


def add_toss_winner_match_winner(matches: pd.DataFrame) -> pd.DataFrame:
    """Add binary flag indicating whether the toss winner also won the match.

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        matches with new column ``toss_winner_won`` (1 = yes, 0 = no)
    """
    matches = matches.copy()
    matches["toss_winner_won"] = (matches["toss_winner"] == matches["winner"]).astype(int)
    return matches


def add_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Extract season (year) from the date column if not already present.

    Args:
        matches: Matches DataFrame with a ``date`` column

    Returns:
        matches with ``season`` column populated
    """
    matches = matches.copy()
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        matches["season"] = matches["date"].dt.year
    return matches


def encode_categorical(matches: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label-encode specified categorical columns (in-place codes).

    Args:
        matches: DataFrame to encode
        columns: List of column names to encode

    Returns:
        DataFrame with encoded columns
    """
    matches = matches.copy()
    for col in columns:
        if col in matches.columns:
            matches[col] = matches[col].astype("category").cat.codes
    return matches


def get_team_win_rate(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute all-time win rate for each team.

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        DataFrame with columns [``team``, ``win_rate``, ``wins``, ``total_games``]
    """
    win_counts = matches["winner"].value_counts().rename("wins")
    total_games = pd.concat([matches["team1"], matches["team2"]]).value_counts().rename("total_games")
    wr = pd.concat([win_counts, total_games], axis=1).fillna(0)
    wr["win_rate"] = (wr["wins"] / wr["total_games"]).round(4)
    wr = wr.reset_index().rename(columns={"index": "team"})
    return wr.sort_values("win_rate", ascending=False)


def batting_strike_rate(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute career strike rate per batsman (min 100 balls faced).

    Args:
        deliveries: Cleaned deliveries DataFrame

    Returns:
        DataFrame with columns [``batsman``, ``runs``, ``balls``, ``strike_rate``]
    """
    grp = deliveries.groupby("batter").agg(
        runs=("batsman_runs", "sum"),
        balls=("ball", "count"),
    ).reset_index().rename(columns={"batter": "batsman"})
    grp = grp[grp["balls"] >= 100]
    grp["strike_rate"] = ((grp["runs"] / grp["balls"]) * 100).round(2)
    return grp.sort_values("strike_rate", ascending=False)


def bowling_economy(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Compute career economy rate per bowler (min 60 legal balls bowled).

    Args:
        deliveries: Cleaned deliveries DataFrame

    Returns:
        DataFrame with columns [``bowler``, ``runs_conceded``, ``overs``, ``economy``]
    """
    # Legal deliveries (no wides/no-balls counted separately via extra_type)
    legal = deliveries[~deliveries.get("extras_type", pd.Series(dtype=str)).isin(["wides", "noballs"])] \
        if "extras_type" in deliveries.columns else deliveries

    grp = legal.groupby("bowler").agg(
        runs_conceded=("total_runs", "sum"),
        balls=("ball", "count"),
    ).reset_index()
    grp = grp[grp["balls"] >= 60]
    grp["overs"] = (grp["balls"] / 6).round(2)
    grp["economy"] = (grp["runs_conceded"] / grp["overs"]).round(2)
    return grp.sort_values("economy")


def home_ground_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    """Estimate home-ground advantage: win rate when playing in home city.

    Uses a simple heuristic: city in team name (e.g. 'Mumbai' in 'Mumbai Indians').

    Args:
        matches: Cleaned matches DataFrame with ``city`` and ``winner`` columns

    Returns:
        DataFrame with columns [``team``, ``home_matches``, ``home_wins``, ``home_win_rate``]
    """
    if "city" not in matches.columns:
        return pd.DataFrame()

    records = []
    teams = pd.concat([matches["team1"], matches["team2"]]).unique()
    for team in teams:
        team_city_keyword = team.split()[0]  # e.g. "Mumbai" from "Mumbai Indians"
        home_mask = (
            matches["city"].str.contains(team_city_keyword, case=False, na=False) &
            ((matches["team1"] == team) | (matches["team2"] == team))
        )
        home_games = matches[home_mask]
        home_wins = (home_games["winner"] == team).sum()
        records.append({
            "team": team,
            "home_matches": len(home_games),
            "home_wins": int(home_wins),
            "home_win_rate": round(home_wins / len(home_games), 4) if len(home_games) > 0 else 0.0,
        })
    return pd.DataFrame(records).sort_values("home_win_rate", ascending=False)


if __name__ == "__main__":
    import os
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")

    matches = add_toss_winner_match_winner(matches)
    matches = add_season(matches)
    print("Win Rates:\n", get_team_win_rate(matches).head())
    print("\nStrike Rates:\n", batting_strike_rate(deliveries).head())
    print("\nEconomy Rates:\n", bowling_economy(deliveries).head())
