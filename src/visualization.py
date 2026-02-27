"""
visualization.py
----------------
Generate and optionally save styled IPL charts using matplotlib and seaborn.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Global style
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def _save_or_show(fig, filename: str, save: bool) -> None:
    plt.tight_layout()
    if save:
        path = os.path.join(CHARTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_top_batsmen(top_batsmen: pd.DataFrame, save: bool = True) -> None:
    """Horizontal bar chart for top run scorers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("flare", len(top_batsmen))
    bars = ax.barh(top_batsmen["batsman"], top_batsmen["total_runs"], color=colors)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
    ax.set_title("Top Run Scorers in IPL", fontweight="bold")
    ax.set_xlabel("Total Runs")
    ax.set_ylabel("Batsman")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save_or_show(fig, "top_batsmen.png", save)


def plot_top_bowlers(top_bowlers: pd.DataFrame, save: bool = True) -> None:
    """Horizontal bar chart for top wicket takers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("mako", len(top_bowlers))
    bars = ax.barh(top_bowlers["bowler"], top_bowlers["wickets"], color=colors)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
    ax.set_title("Top Wicket Takers in IPL", fontweight="bold")
    ax.set_xlabel("Wickets")
    ax.set_ylabel("Bowler")
    ax.invert_yaxis()
    _save_or_show(fig, "top_bowlers.png", save)


def plot_wins_per_season(wins_df: pd.DataFrame, save: bool = True) -> None:
    """Line chart showing team wins per season."""
    fig, ax = plt.subplots(figsize=(14, 7))
    teams = wins_df["winner"].unique()
    palette = sns.color_palette("tab20", len(teams))
    for i, team in enumerate(teams):
        td = wins_df[wins_df["winner"] == team]
        ax.plot(td["season"], td["wins"], marker="o", label=team, color=palette[i], linewidth=2)
    ax.set_title("Team Wins Per Season", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Wins")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    _save_or_show(fig, "wins_per_season.png", save)


def plot_toss_impact(toss_df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart showing win rate by toss decision."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(toss_df["toss_decision"], toss_df["win_rate"], color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=12)
    ax.set_title("Win Rate by Toss Decision", fontweight="bold")
    ax.set_xlabel("Toss Decision")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    _save_or_show(fig, "toss_impact.png", save)


def plot_run_rate_per_over(rr_df: pd.DataFrame, save: bool = True) -> None:
    """Line chart of average run rate by over number."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(rr_df["over"], rr_df["avg_runs"], alpha=0.25, color="#4C72B0")
    ax.plot(rr_df["over"], rr_df["avg_runs"], marker="o", color="#4C72B0", linewidth=2.5)
    # Shade powerplay and death overs
    ax.axvspan(1, 6, alpha=0.10, color="green", label="Powerplay (1–6)")
    ax.axvspan(17, 20, alpha=0.10, color="red", label="Death Overs (17–20)")
    ax.set_title("Average Runs Per Over (All IPL Matches)", fontweight="bold")
    ax.set_xlabel("Over Number")
    ax.set_ylabel("Avg Runs")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    _save_or_show(fig, "run_rate_per_over.png", save)


def plot_player_of_match(pom_df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart for top Player-of-the-Match award winners."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("rocket", len(pom_df))
    bars = ax.barh(pom_df["player"], pom_df["awards"], color=colors)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
    ax.set_title("Top Player of the Match Award Winners", fontweight="bold")
    ax.set_xlabel("Awards")
    ax.set_ylabel("Player")
    ax.invert_yaxis()
    _save_or_show(fig, "player_of_match.png", save)


def plot_phase_comparison(pp_df: pd.DataFrame, death_df: pd.DataFrame, save: bool = True) -> None:
    """Side-by-side comparison of powerplay vs death over run rates per team."""
    merged = pp_df.merge(death_df, on="batting_team", how="inner").head(10)
    x = range(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar([i - width / 2 for i in x], merged["avg_powerplay_runs"], width=width,
           label="Powerplay (1–6)", color="#4C72B0", edgecolor="white")
    ax.bar([i + width / 2 for i in x], merged["avg_death_runs"], width=width,
           label="Death Overs (17–20)", color="#DD8452", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["batting_team"], rotation=30, ha="right")
    ax.set_title("Powerplay vs Death Over Avg Runs (Top 10 Teams)", fontweight="bold")
    ax.set_ylabel("Avg Runs")
    ax.legend()
    _save_or_show(fig, "phase_comparison.png", save)


def plot_boundaries(boundary_df: pd.DataFrame, save: bool = True) -> None:
    """Grouped bar chart for sixes and fours per top batsman."""
    fig, ax = plt.subplots(figsize=(13, 7))
    x = range(len(boundary_df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], boundary_df["sixes"], width=width,
           label="Sixes", color="#2ecc71", edgecolor="white")
    ax.bar([i + width / 2 for i in x], boundary_df["fours"], width=width,
           label="Fours", color="#e74c3c", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(boundary_df["batsman"], rotation=40, ha="right")
    ax.set_title("Sixes & Fours by Top Batsmen", fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend()
    _save_or_show(fig, "boundaries.png", save)
