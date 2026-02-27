"""
analysis.py
-----------
Statistical analysis functions for IPL matches and deliveries data.
"""

import pandas as pd
import numpy as np


def top_run_scorers(deliveries: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get top N run scorers across all IPL seasons.

    Args:
        deliveries: Cleaned deliveries DataFrame
        top_n: Number of top batsmen to return

    Returns:
        DataFrame with columns [``batsman``, ``total_runs``]
    """
    runs = deliveries.groupby("batter")["batsman_runs"].sum().reset_index()
    runs.columns = ["batsman", "total_runs"]
    return runs.sort_values("total_runs", ascending=False).head(top_n)


def top_wicket_takers(deliveries: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get top N wicket takers, excluding run-outs.

    Args:
        deliveries: Cleaned deliveries DataFrame
        top_n: Number of top bowlers to return

    Returns:
        DataFrame with columns [``bowler``, ``wickets``]
    """
    wickets = deliveries[
        (deliveries["dismissal_kind"].notna()) &
        (deliveries["dismissal_kind"] != "run out")
    ]
    wk = wickets.groupby("bowler")["dismissal_kind"].count().reset_index()
    wk.columns = ["bowler", "wickets"]
    return wk.sort_values("wickets", ascending=False).head(top_n)


def team_wins_per_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Get number of wins per team per season.

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        DataFrame with columns [``season``, ``winner``, ``wins``]
    """
    if "season" not in matches.columns:
        matches = matches.copy()
        matches["date"] = pd.to_datetime(matches["date"])
        matches["season"] = matches["date"].dt.year
    wins = matches.groupby(["season", "winner"]).size().reset_index(name="wins")
    return wins


def toss_impact_analysis(matches: pd.DataFrame) -> pd.DataFrame:
    """Analyse win rate based on toss decision (bat/field).

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        DataFrame with columns [``toss_decision``, ``win_rate``, ``total_matches``]
    """
    df = matches.copy()
    df["toss_winner_won"] = (df["toss_winner"] == df["winner"]).astype(int)
    analysis = df.groupby("toss_decision").agg(
        win_rate=("toss_winner_won", "mean"),
        total_matches=("toss_winner_won", "count"),
    ).reset_index()
    analysis["win_rate"] = analysis["win_rate"].round(4)
    return analysis


def venue_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Get match count per venue sorted by activity.

    Args:
        matches: Cleaned matches DataFrame

    Returns:
        DataFrame with columns [``venue``, ``total_matches``]
    """
    venue = matches.groupby("venue").size().reset_index(name="total_matches")
    return venue.sort_values("total_matches", ascending=False)


def batsman_strike_rate(deliveries: pd.DataFrame, min_balls: int = 100) -> pd.DataFrame:
    """Compute career strike rate for each batsman.

    Args:
        deliveries: Cleaned deliveries DataFrame
        min_balls: Minimum balls faced to be included

    Returns:
        DataFrame with columns [``batsman``, ``runs``, ``balls``, ``strike_rate``]
    """
    grp = deliveries.groupby("batter").agg(
        runs=("batsman_runs", "sum"),
        balls=("ball", "count"),
    ).reset_index().rename(columns={"batter": "batsman"})
    grp = grp[grp["balls"] >= min_balls]
    grp["strike_rate"] = ((grp["runs"] / grp["balls"]) * 100).round(2)
    return grp.sort_values("strike_rate", ascending=False)


def bowler_economy(deliveries: pd.DataFrame, min_balls: int = 120) -> pd.DataFrame:
    """Compute career economy rate per bowler.

    Args:
        deliveries: Cleaned deliveries DataFrame
        min_balls: Minimum balls bowled to be included

    Returns:
        DataFrame with columns [``bowler``, ``runs_conceded``, ``overs``, ``economy``]
    """
    grp = deliveries.groupby("bowler").agg(
        runs_conceded=("total_runs", "sum"),
        balls=("ball", "count"),
    ).reset_index()
    grp = grp[grp["balls"] >= min_balls]
    grp["overs"] = (grp["balls"] / 6).round(2)
    grp["economy"] = (grp["runs_conceded"] / grp["overs"]).round(2)
    return grp.sort_values("economy")


def powerplay_analysis(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Average runs scored in powerplay overs (1–6) per team.

    Args:
        deliveries: Cleaned deliveries DataFrame

    Returns:
        DataFrame with columns [``batting_team``, ``avg_powerplay_runs``]
    """
    pp = deliveries[deliveries["over"].between(1, 6)]
    result = pp.groupby(["match_id", "batting_team"])["total_runs"].sum().reset_index()
    result = result.groupby("batting_team")["total_runs"].mean().reset_index()
    result.columns = ["batting_team", "avg_powerplay_runs"]
    result["avg_powerplay_runs"] = result["avg_powerplay_runs"].round(2)
    return result.sort_values("avg_powerplay_runs", ascending=False)


def death_over_analysis(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Average runs scored in death overs (17–20) per team.

    Args:
        deliveries: Cleaned deliveries DataFrame

    Returns:
        DataFrame with columns [``batting_team``, ``avg_death_runs``]
    """
    death = deliveries[deliveries["over"].between(17, 20)]
    result = death.groupby(["match_id", "batting_team"])["total_runs"].sum().reset_index()
    result = result.groupby("batting_team")["total_runs"].mean().reset_index()
    result.columns = ["batting_team", "avg_death_runs"]
    result["avg_death_runs"] = result["avg_death_runs"].round(2)
    return result.sort_values("avg_death_runs", ascending=False)


def player_of_match_count(matches: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Count how many times each player won Player of the Match.

    Args:
        matches: Cleaned matches DataFrame
        top_n: Number of players to return

    Returns:
        DataFrame with columns [``player``, ``awards``]
    """
    if "player_of_match" not in matches.columns:
        return pd.DataFrame(columns=["player", "awards"])
    awards = matches["player_of_match"].value_counts().head(top_n).reset_index()
    awards.columns = ["player", "awards"]
    return awards


def six_four_count(deliveries: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Count sixes and fours per batsman.

    Args:
        deliveries: Cleaned deliveries DataFrame
        top_n: Number of batsmen to return

    Returns:
        DataFrame with columns [``batsman``, ``sixes``, ``fours``, ``boundaries``]
    """
    grp = deliveries.groupby("batter").agg(
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
    ).reset_index().rename(columns={"batter": "batsman"})
    grp["boundaries"] = grp["sixes"] + grp["fours"]
    return grp.sort_values("boundaries", ascending=False).head(top_n)


def run_rate_per_over(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Average run rate across all matches for each over number (1–20).

    Args:
        deliveries: Cleaned deliveries DataFrame

    Returns:
        DataFrame with columns [``over``, ``avg_runs``]
    """
    per_over = deliveries[deliveries["over"].between(1, 20)]
    result = per_over.groupby(["match_id", "over"])["total_runs"].sum().reset_index()
    result = result.groupby("over")["total_runs"].mean().reset_index()
    result.columns = ["over", "avg_runs"]
    result["avg_runs"] = result["avg_runs"].round(2)
    return result


def player_stats(deliveries: pd.DataFrame, player_name: str) -> dict:
    """Fetch combined batting and bowling stats for a given player.

    Args:
        deliveries: Cleaned deliveries DataFrame
        player_name: Player name string (case-insensitive partial match supported)

    Returns:
        Dict with ``batting`` and ``bowling`` sub-dicts
    """
    name_lower = player_name.strip().lower()

    # Batting
    bat_df = deliveries[deliveries["batter"].str.lower().str.contains(name_lower, na=False)]
    batting = {}
    if not bat_df.empty:
        batting["runs"] = int(bat_df["batsman_runs"].sum())
        batting["balls"] = len(bat_df)
        batting["strike_rate"] = round((batting["runs"] / batting["balls"]) * 100, 2) if batting["balls"] else 0
        batting["sixes"] = int((bat_df["batsman_runs"] == 6).sum())
        batting["fours"] = int((bat_df["batsman_runs"] == 4).sum())

    # Bowling
    bowl_df = deliveries[deliveries["bowler"].str.lower().str.contains(name_lower, na=False)]
    bowling = {}
    if not bowl_df.empty:
        bowling["wickets"] = int(
            bowl_df[(bowl_df["dismissal_kind"].notna()) & (bowl_df["dismissal_kind"] != "run out")].shape[0]
        )
        bowling["runs_conceded"] = int(bowl_df["total_runs"].sum())
        balls_bowled = len(bowl_df)
        overs = round(balls_bowled / 6, 2)
        bowling["economy"] = round(bowling["runs_conceded"] / overs, 2) if overs else 0

    return {"batting": batting, "bowling": bowling}


if __name__ == "__main__":
    deliveries = pd.read_csv("data/deliveries.csv")
    matches = pd.read_csv("data/matches.csv")
    print("Top Scorers:\n", top_run_scorers(deliveries))
    print("\nTop Wicket Takers:\n", top_wicket_takers(deliveries))
    print("\nToss Impact:\n", toss_impact_analysis(matches))
    print("\nPowerplay Analysis:\n", powerplay_analysis(deliveries).head())
    print("\nDeath Over Analysis:\n", death_over_analysis(deliveries).head())
    print("\nPlayer of Match:\n", player_of_match_count(matches))
