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
RESULTS_URL = "https://www.singaporepools.com.sg/en/product/Pages/4d_results.aspx"
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

def get_latest_website_date():
    """Scrape the latest draw date from Singapore Pools website."""
    try:
        r = requests.get(RESULTS_URL, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        date_element = soup.find(
            "span",
            {"id":"ctl00_ctl37_g_1c8ad7f0_9b2f_4f9a_ba9f_5f6e3b3c2c40_lblDrawDate"}
        )
        if date_element:
            return pd.to_datetime(date_element.text.strip(), errors="coerce")
    except Exception as e:
        print("Website check failed:", e)
    return None

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

    # Optional features
    dataset["digit_sum"] = dataset[["d1","d2","d3","d4"]].sum(axis=1)
    dataset["odd_count"] = (dataset[["d1","d2","d3","d4"]] % 2).sum(axis=1)
    dataset["high_count"] = (dataset[["d1","d2","d3","d4"]] >= 5).sum(axis=1)
    dataset["repeat_count"] = dataset.apply(
        lambda r: len([x for x in [r.d1,r.d2,r.d3,r.d4] if [r.d1,r.d2,r.d3,r.d4].count(x)>1]), axis=1
    )
    return dataset

# -----------------------
# BUILD PROBABILITIES
# -----------------------

def build_digit_probs(dataset, windows=[20,50,100,200]):
    """Calculate probabilistic predictions based on historical data."""
    digit_probs = {}
    for pos in ["d1","d2","d3","d4"]:
        probs = np.zeros(10)
        for w in windows:
            hist = dataset[pos].iloc[-w:]
            freq = hist.value_counts(normalize=True).reindex(range(10), fill_value=0)
            probs += freq.values
        digit_probs[pos] = normalize_probs(probs / len(windows))

        # recency weighting (boost digits that haven't appeared recently)
        last_idx = {d: (dataset[dataset[pos]==d].index.max() if d in dataset[pos].values else -999) for d in range(10)}
        recency_weight = np.array([1 / (1 + len(dataset) - last_idx[d]) for d in range(10)])
        digit_probs[pos] = normalize_probs(0.8 * digit_probs[pos] + 0.2 * recency_weight)

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
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🎯 Singapore 4D Probabilistic Predictor</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size: 1.2rem;'>
Predict Singapore 4D numbers using historical frequency and recency patterns.<br>
No heavy machine learning required—just probabilistic modeling and Monte Carlo simulation.
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
    Adjust the historical windows for probability calculation:
    </p>
    <ul style='font-size:0.9rem;'>
        <li><b>Window 1:</b> Recent draws (e.g., 20 most recent)</li>
        <li><b>Window 2:</b> Slightly longer history</li>
        <li><b>Window 3:</b> Medium-term history</li>
        <li><b>Window 4:</b> Long-term history</li>
    </ul>
    <p style='font-size:0.9rem;'>Increasing windows makes predictions rely more on long-term trends.</p>
    """, unsafe_allow_html=True)

    w1 = st.number_input("Window 1 (recent draws)", min_value=5, max_value=200, value=20)
    w2 = st.number_input("Window 2", min_value=10, max_value=500, value=50)
    w3 = st.number_input("Window 3", min_value=50, max_value=1000, value=100)
    w4 = st.number_input("Window 4", min_value=100, max_value=2000, value=200)
    n_samples = st.number_input("Monte Carlo samples", min_value=1000, max_value=200000, value=50000)

windows = [w1, w2, w3, w4]

# -----------------------
# GENERATE PREDICTION
# -----------------------

if st.button("Generate Prediction"):
    with st.spinner("Calculating probabilities and running Monte Carlo..."):
        digit_probs = build_digit_probs(dataset, windows=windows)
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