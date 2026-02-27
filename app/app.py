"""
app.py
------
Streamlit dashboard for IPL Data Analysis.
Run: streamlit run app/app.py
"""

import os
import sys
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_cleaning import clean_matches, clean_deliveries
from src.analysis import (
    top_run_scorers, top_wicket_takers, team_wins_per_season,
    toss_impact_analysis, venue_stats, batsman_strike_rate,
    bowler_economy, powerplay_analysis, death_over_analysis,
    player_of_match_count, six_four_count, run_rate_per_over,
    player_stats,
)
from src.visualization import (
    plot_top_batsmen, plot_top_bowlers, plot_wins_per_season,
    plot_toss_impact, plot_run_rate_per_over, plot_player_of_match,
    plot_phase_comparison, plot_boundaries,
)
from src.model_training import load_model, load_encoders, predict_winner, ENCODERS_PATH, MODEL_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Data Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #153354);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; margin: 0; opacity: 0.75; }
    [data-testid="stSidebar"] { background-color: #0f1b2d; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Resolve data paths (try multiple candidates for Streamlit Cloud) ──────────
def _find_data_dir() -> Path:
    """Try several candidate locations so the app works locally and on Streamlit Cloud."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data",   # normal: app/../data
        Path.cwd() / "data",                               # cwd fallback
        Path("/mount/src/ipl-data-analysis/data"),         # Streamlit Cloud mount
    ]
    for c in candidates:
        if (c / "matches.csv").exists() and (c / "deliveries.csv").exists():
            return c
    return candidates[0]   # default even if not found (shows upload UI)

DATA_DIR    = _find_data_dir()
MATCHES_PATH    = DATA_DIR / "matches.csv"
DELIVERIES_PATH = DATA_DIR / "deliveries.csv"


def is_valid_csv(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_all_data():
    matches_raw = pd.read_csv(MATCHES_PATH)
    deliveries_raw = pd.read_csv(DELIVERIES_PATH)
    matches = clean_matches(matches_raw)
    deliveries = clean_deliveries(deliveries_raw)
    return matches, deliveries


if is_valid_csv(MATCHES_PATH) and is_valid_csv(DELIVERIES_PATH):
    try:
        matches, deliveries = load_all_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
else:
    st.info(
        "📂 **No data found.** Please upload your IPL dataset files below.\n\n"
        "Download from [Kaggle IPL Dataset]"
        "(https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).",
        icon="ℹ️",
    )
    with st.expander("🔍 Debug info"):
        st.code(f"DATA_DIR     : {DATA_DIR}\n"
                f"matches.csv  : exists={MATCHES_PATH.exists()}\n"
                f"deliveries   : exists={DELIVERIES_PATH.exists()}\n"
                f"__file__     : {Path(__file__).resolve()}\n"
                f"cwd          : {Path.cwd()}")
    col1, col2 = st.columns(2)
    with col1:
        matches_file = st.file_uploader("Upload `matches.csv`", type="csv", key="matches")
    with col2:
        deliveries_file = st.file_uploader("Upload `deliveries.csv`", type="csv", key="deliveries")

    if matches_file and deliveries_file:
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        (tmp / "matches.csv").write_bytes(matches_file.read())
        (tmp / "deliveries.csv").write_bytes(deliveries_file.read())
        # Override paths to temp dir so load_all_data can find them
        MATCHES_PATH    = tmp / "matches.csv"
        DELIVERIES_PATH = tmp / "deliveries.csv"
        try:
            matches, deliveries = load_all_data()
            st.success("✅ Files loaded! Scroll up to use the dashboard.")
        except Exception as e:
            st.error(f"❌ {e}")
        st.stop()
    else:
        st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏏 IPL Dashboard")
    st.markdown("*Explore 15+ years of IPL data.*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🏏 Batting Stats", "🎳 Bowling Stats",
         "📅 Season Analysis", "🪙 Toss Analysis",
         "👤 Player Analysis", "📊 Phase Analysis", "🤖 Predict Winner"],
        label_visibility="collapsed",
    )

# ── Helper: render matplotlib figure ─────────────────────────────────────────
def render(fig_fn, *args, **kwargs):
    fig_fn(*args, save=False, **kwargs)
    st.pyplot(plt.gcf())
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 IPL Dataset Overview")

    seasons = sorted(matches["season"].dropna().unique())
    teams = pd.concat([matches["team1"], matches["team2"]]).nunique()
    players = deliveries["batter"].nunique()
    venues = matches["venue"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏟️ Total Matches", f"{len(matches):,}")
    c2.metric("📆 Seasons", f"{min(seasons)} – {max(seasons)}")
    c3.metric("🧑‍🤝‍🧑 Teams", teams)
    c4.metric("🏟️ Venues", venues)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matches (sample)")
        st.dataframe(matches.head(8), use_container_width=True)
    with col2:
        st.subheader("Deliveries (sample)")
        st.dataframe(deliveries.head(8), use_container_width=True)

    st.subheader("🏆 Most Successful Teams (All Time)")
    wins = matches["winner"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    st.dataframe(wins, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batting Stats
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏏 Batting Stats":
    st.title("🏏 Batting Statistics")

    tab1, tab2, tab3 = st.tabs(["Top Run Scorers", "Strike Rates", "Boundaries"])

    with tab1:
        top_n = st.slider("Top N Batsmen", 5, 20, 10, key="bat_n")
        top_bat = top_run_scorers(deliveries, top_n)
        st.dataframe(top_bat, use_container_width=True)
        render(plot_top_batsmen, top_bat)

    with tab2:
        min_balls = st.slider("Minimum Balls Faced", 50, 500, 200, step=50)
        sr_df = batsman_strike_rate(deliveries, min_balls=min_balls).head(15)
        st.dataframe(sr_df, use_container_width=True)

    with tab3:
        bd = six_four_count(deliveries, top_n=15)
        st.dataframe(bd, use_container_width=True)
        render(plot_boundaries, bd)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bowling Stats
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎳 Bowling Stats":
    st.title("🎳 Bowling Statistics")

    tab1, tab2 = st.tabs(["Top Wicket Takers", "Economy Rates"])

    with tab1:
        top_n = st.slider("Top N Bowlers", 5, 20, 10, key="bowl_n")
        top_bowl = top_wicket_takers(deliveries, top_n)
        st.dataframe(top_bowl, use_container_width=True)
        render(plot_top_bowlers, top_bowl)

    with tab2:
        min_balls = st.slider("Minimum Balls Bowled", 60, 600, 300, step=60)
        eco_df = bowler_economy(deliveries, min_balls=min_balls).head(15)
        st.dataframe(eco_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Season Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Season Analysis":
    st.title("📅 Season-by-Season Analysis")

    wins = team_wins_per_season(matches)
    st.dataframe(wins, use_container_width=True)
    render(plot_wins_per_season, wins)

    st.subheader("🌍 Venue Activity")
    venue = venue_stats(matches)
    st.dataframe(venue.head(15), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Toss Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🪙 Toss Analysis":
    st.title("🪙 Toss Impact on Match Outcome")

    toss = toss_impact_analysis(matches)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(toss, use_container_width=True)
    with c2:
        render(plot_toss_impact, toss)

    st.info(
        "📌 **Key Insight**: The toss gives a slight edge but is not the decisive factor. "
        "Batting/fielding conditions and team quality matter more.",
        icon="💡",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Player Analysis (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Player Analysis":
    st.title("👤 Player Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        player_name = st.text_input("🔍 Search player name", placeholder="e.g. Virat Kohli")

    if player_name.strip():
        stats = player_stats(deliveries, player_name)

        bat = stats["batting"]
        bowl = stats["bowling"]

        st.markdown("### 🏏 Batting")
        if bat:
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Runs", bat.get("runs", 0))
            b2.metric("Balls", bat.get("balls", 0))
            b3.metric("Strike Rate", bat.get("strike_rate", 0))
            b4.metric("Sixes", bat.get("sixes", 0))
            b5.metric("Fours", bat.get("fours", 0))
        else:
            st.warning("No batting records found.")

        st.markdown("### 🎳 Bowling")
        if bowl:
            b1, b2, b3 = st.columns(3)
            b1.metric("Wickets", bowl.get("wickets", 0))
            b2.metric("Runs Conceded", bowl.get("runs_conceded", 0))
            b3.metric("Economy", bowl.get("economy", 0))
        else:
            st.warning("No bowling records found.")
    else:
        st.info("Enter a player name to see their stats.")

    st.markdown("---")
    st.subheader("🏆 Player of the Match Leaders")
    pom = player_of_match_count(matches, top_n=15)
    if not pom.empty:
        st.dataframe(pom, use_container_width=True)
        render(plot_player_of_match, pom)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Phase Analysis (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Phase Analysis":
    st.title("📊 Match Phase Analysis")

    st.subheader("⚡ Average Run Rate by Over")
    rr = run_rate_per_over(deliveries)
    st.dataframe(rr, use_container_width=True)
    render(plot_run_rate_per_over, rr)

    st.markdown("---")
    st.subheader("🔥 Powerplay vs Death Overs (by Team)")
    pp = powerplay_analysis(deliveries)
    death = death_over_analysis(deliveries)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Powerplay (Overs 1–6)**")
        st.dataframe(pp.head(10), use_container_width=True)
    with col2:
        st.markdown("**Death Overs (Overs 17–20)**")
        st.dataframe(death.head(10), use_container_width=True)
    render(plot_phase_comparison, pp, death)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Winner
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Winner":
    st.title("🤖 Predict Match Winner")

    model_exists = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0
    enc_exists = os.path.exists(ENCODERS_PATH) and os.path.getsize(ENCODERS_PATH) > 0

    if not (model_exists and enc_exists):
        st.warning("⚠️ No trained model found. Run the command below first:")
        st.code("python src/model_training.py", language="bash")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
        encoders = load_encoders(ENCODERS_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    # Show model metadata if available
    if os.path.exists(os.path.join(os.path.dirname(MODEL_PATH), "model_metadata.json")):
        with open(os.path.join(os.path.dirname(MODEL_PATH), "model_metadata.json")) as f:
            meta = json.load(f)
        best = meta.get("model_name", "Unknown")
        metrics = meta.get("metrics", {}).get(best, {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", best)
        c2.metric("CV Accuracy", f"{metrics.get('cv_mean', 0):.2%}")
        c3.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
        st.markdown("---")

    # Get known teams and venues from encoders
    teams = sorted(encoders["team1"].classes_.tolist())
    venues = sorted(encoders["venue"].classes_.tolist())

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🏟️ Team 1", teams)
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, "Other"])
    with col2:
        team2 = st.selectbox("🏟️ Team 2", [t for t in teams if t != team1])
        toss_decision = st.selectbox("📋 Toss Decision", ["bat", "field"])

    venue = st.selectbox("🌍 Venue", venues)

    if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):
        tw = team1 if toss_winner == team1 else team2
        try:
            winner = predict_winner(team1, team2, tw, toss_decision, venue, model, encoders)
            st.success(f"🏆 Predicted Winner: **{winner}**")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
