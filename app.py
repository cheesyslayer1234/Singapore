# streamlit_4d_predictor.py

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

# -----------------------
# CONFIG
# -----------------------

CSV_FILE = "singapore_4d_history.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def normalize_probs(p):
    """Normalize an array to sum to 1 and clip negative values."""
    p = np.array(p)
    p = np.maximum(p, 0)
    s = p.sum()
    return p / s if s > 0 else np.ones(len(p)) / len(p)

# -----------------------
# LOAD DATA
# -----------------------

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    # Prepare numbers
    number_cols = [c for c in df.columns if c not in ["draw_number","draw_date"]]
    df["numbers"] = df[number_cols].apply(lambda row: [str(n).zfill(4) for n in row], axis=1)

    all_numbers = []
    for _, row in df.iterrows():
        all_numbers.extend(row["numbers"])

    dataset = pd.DataFrame({"number": all_numbers})
    dataset["d1"] = dataset["number"].str[0].astype(int)
    dataset["d2"] = dataset["number"].str[1].astype(int)
    dataset["d3"] = dataset["number"].str[2].astype(int)
    dataset["d4"] = dataset["number"].str[3].astype(int)
    return dataset

# -----------------------
# BUILD PROBABILITIES
# -----------------------

def build_digit_probs(dataset, window):
    """Calculate digit probabilities using the last `window` draws."""
    digit_probs = {}
    for pos in ["d1","d2","d3","d4"]:
        hist = dataset[pos].iloc[-window:]
        freq = hist.value_counts(normalize=True).reindex(range(10), fill_value=0)
        # recency weighting: boost digits that haven't appeared recently
        last_idx = {d: (dataset[dataset[pos]==d].index.max() if d in dataset[pos].values else -999) for d in range(10)}
        recency_weight = np.array([1 / (1 + len(dataset) - last_idx[d]) for d in range(10)])
        probs = 0.8 * freq.values + 0.2 * recency_weight
        digit_probs[pos] = normalize_probs(probs)
    return digit_probs

# -----------------------
# MONTE CARLO SAMPLING
# -----------------------

def monte_carlo_predict(digit_probs, n_samples=50000):
    predictions = []
    for _ in range(n_samples):
        d1 = np.random.choice(np.arange(10), p=digit_probs["d1"])
        d2 = np.random.choice(np.arange(10), p=digit_probs["d2"])
        d3 = np.random.choice(np.arange(10), p=digit_probs["d3"])
        d4 = np.random.choice(np.arange(10), p=digit_probs["d4"])
        predictions.append(f"{d1}{d2}{d3}{d4}")
    ranked = pd.Series(predictions).value_counts()
    return ranked.head(50).reset_index().rename(columns={"index":"number","count":"freq"})

# -----------------------
# STREAMLIT APP
# -----------------------

st.set_page_config(page_title="Singapore 4D Predictor", layout="wide")

st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🎯 Singapore 4D Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem;'>
Predict Singapore 4D numbers using historical frequency and recency patterns.<br>
No complex ML models—just simple probability & Monte Carlo simulation.
</p>
""", unsafe_allow_html=True)

dataset = load_data()

# -----------------------
# SIDEBAR SETTINGS
# -----------------------

with st.sidebar:
    st.header("Settings & Explanation")
    st.markdown("""
    <p style='font-size:0.9rem;'>
    <b>Window size:</b> How many of the most recent draws the app should look at to calculate probabilities.<br>
    - Smaller window → focuses on recent trends.<br>
    - Larger window → considers long-term patterns.<br>
    </p>
    """, unsafe_allow_html=True)

    window = st.number_input("Number of recent draws to consider", min_value=5, max_value=1000, value=100)
    n_samples = st.number_input("Monte Carlo samples", min_value=1000, max_value=200000, value=50000)

# -----------------------
# GENERATE PREDICTION
# -----------------------

if st.button("Generate Prediction"):
    with st.spinner("Calculating probabilities and running Monte Carlo..."):
        digit_probs = build_digit_probs(dataset, window=window)
        top_predictions = monte_carlo_predict(digit_probs, n_samples=n_samples)

    st.markdown("<h2 style='text-align: center;'>Top 50 Predicted Numbers</h2>", unsafe_allow_html=True)
    st.dataframe(top_predictions, height=500)

    st.markdown("<h2 style='text-align: center;'>Digit Probabilities</h2>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Digit": list(range(10)),
        "D1": digit_probs["d1"],
        "D2": digit_probs["d2"],
        "D3": digit_probs["d3"],
        "D4": digit_probs["d4"]
    })
    st.bar_chart(prob_df.set_index("Digit"))

    st.success("✅ Prediction complete!")