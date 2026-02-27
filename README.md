# 🏏 IPL Data Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)

A comprehensive **data analysis and machine learning project** built on 15+ years of IPL (Indian Premier League) cricket data. Includes statistical analysis, interactive visualisations, and an ML-powered match-winner predictor.

---

## 📁 Project Structure

```
ipl-data-analysis/
├── data/                    # Raw CSV datasets (matches.csv, deliveries.csv)
├── notebooks/               # Jupyter notebook for EDA
│   └── ipl_analysis.ipynb
├── src/                     # Core Python package
│   ├── __init__.py
│   ├── data_cleaning.py     # Load, validate & preprocess data
│   ├── feature_engineering.py  # Derived features (strike rate, economy, etc.)
│   ├── analysis.py          # Statistical analysis functions (15+)
│   ├── visualization.py     # Chart generation (8 chart types)
│   └── model_training.py    # Train, evaluate & save ML model
├── models/                  # Saved model, encoders & metadata
├── app/
│   └── app.py               # Streamlit multi-page dashboard
├── images/charts/           # Auto-generated chart PNGs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ipl-data-analysis.git
cd ipl-data-analysis
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Data
Place `matches.csv` and `deliveries.csv` inside the `data/` directory.
> Download from [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).

---

## 🔬 Modules

| Module | Description |
|---|---|
| `data_cleaning.py` | Load & validate schema, normalize team names, handle missing values |
| `feature_engineering.py` | Strike rate, economy rate, home-ground advantage, win rate |
| `analysis.py` | 15+ analysis functions: top scorers, powerplay, death overs, player lookup |
| `visualization.py` | 8 chart types: run rate curve, phase comparison, boundaries, POM awards |
| `model_training.py` | Compare RF / GBM / LR, cross-validate, save model + encoders + metadata |

---

## 📊 Streamlit Dashboard

```bash
streamlit run app/app.py
```

### Pages

| Page | Description |
|---|---|
| 🏠 Overview | KPI metrics – matches, seasons, teams, venues |
| 🏏 Batting Stats | Top scorers, strike rates, boundaries (tabs) |
| 🎳 Bowling Stats | Top wicket takers, economy rates |
| 📅 Season Analysis | Team wins per season + venue activity |
| 🪙 Toss Analysis | Win rate by toss decision with insights |
| 👤 Player Analysis | Search any player → batting & bowling stats |
| 📊 Phase Analysis | Run rate by over, powerplay vs death overs |
| 🤖 Predict Winner | Dropdown-based ML prediction with accuracy display |

---

## 🤖 Train the Model

```bash
python src/model_training.py
```

- Trains **3 models** (RandomForest, GradientBoosting, LogisticRegression)
- Selects best by 5-fold cross-validation accuracy
- Saves model → `models/match_winner_model.pkl`
- Saves encoders → `models/label_encoders.pkl`
- Saves metadata → `models/model_metadata.json`
- Generates feature importance chart → `images/charts/feature_importance.png`

---

## 💡 Key Insights

- **Toss impact is minimal** (~52% win rate for toss winner) — team quality dominates
- **Death over specialists** are crucial: economy < 8 in overs 17–20 is elite
- **Home ground advantage** varies — Mumbai Indians and CSK benefit most
- **Powerplay runs** strongly correlate with final score

---

## 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/ipl_analysis.ipynb
```

---

## 📦 Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn
- streamlit
- jupyter / notebook / ipykernel

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
