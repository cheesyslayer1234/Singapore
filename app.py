import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="4D AI Predictor", layout="wide")

CSV_FILE = "singapore_4d_history.csv"
MODEL_DIR = "models"

# -----------------------
# HELPER
# -----------------------

def normalize_probs(p):
    p = np.array(p)
    p = np.maximum(p, 0)
    s = p.sum()
    if s == 0:
        return np.ones(len(p)) / len(p)
    return p / s


# -----------------------
# LOAD MODELS (CACHED)
# -----------------------

@st.cache_resource
def load_models():

    xgb_models = {
        "d1": joblib.load(f"{MODEL_DIR}/xgb_d1.pkl"),
        "d2": joblib.load(f"{MODEL_DIR}/xgb_d2.pkl"),
        "d3": joblib.load(f"{MODEL_DIR}/xgb_d3.pkl"),
        "d4": joblib.load(f"{MODEL_DIR}/xgb_d4.pkl")
    }

    lstm_model = load_model(f"{MODEL_DIR}/lstm_model.keras")

    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")

    return xgb_models, lstm_model, feature_columns


# -----------------------
# LOAD DATA
# -----------------------

@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)


# -----------------------
# BUILD DATASET
# -----------------------

def build_dataset(df):

    number_cols = [c for c in df.columns if c not in ["draw_number","draw_date"]]

    numbers=[]

    for _,row in df.iterrows():

        draw_nums=[]

        for c in number_cols:
            draw_nums.append(str(row[c]).zfill(4))

        numbers.append(draw_nums)

    df["numbers"]=numbers

    all_numbers=[]
    draw_idx=[]

    for i,row in df.iterrows():

        for n in row["numbers"]:
            all_numbers.append(n)
            draw_idx.append(i)

    dataset=pd.DataFrame({
        "draw":draw_idx,
        "number":all_numbers
    })

    dataset["d1"]=dataset["number"].str[0].astype(int)
    dataset["d2"]=dataset["number"].str[1].astype(int)
    dataset["d3"]=dataset["number"].str[2].astype(int)
    dataset["d4"]=dataset["number"].str[3].astype(int)

    dataset["digit_sum"]=dataset[["d1","d2","d3","d4"]].sum(axis=1)

    dataset["odd_count"]=(dataset[["d1","d2","d3","d4"]] % 2).sum(axis=1)

    dataset["high_count"]=(dataset[["d1","d2","d3","d4"]] >=5).sum(axis=1)

    dataset["repeat_count"]=dataset.apply(
        lambda r: len([x for x in [r.d1,r.d2,r.d3,r.d4] if [r.d1,r.d2,r.d3,r.d4].count(x)>1]),
        axis=1
    )

    return dataset


# -----------------------
# BUILD FEATURES
# -----------------------

def build_prediction_features(dataset):

    windows=[20,50,100,200]

    i=len(dataset)

    predict={}

    for w in windows:

        hist=dataset.iloc[i-w:i]

        for digit in range(10):

            for pos in ["d1","d2","d3","d4"]:

                predict[f"freq_{digit}_{pos}_w{w}"]=(hist[pos]==digit).mean()

        predict[f"sum_mean_w{w}"]=hist["digit_sum"].mean()
        predict[f"sum_std_w{w}"]=hist["digit_sum"].std()

        predict[f"odd_ratio_w{w}"]=hist["odd_count"].mean()
        predict[f"high_ratio_w{w}"]=hist["high_count"].mean()
        predict[f"repeat_ratio_w{w}"]=hist["repeat_count"].mean()

    for digit in range(10):

        for pos in ["d1","d2","d3","d4"]:

            idx=dataset[dataset[pos]==digit].index
            gap = 999 if len(idx)==0 else i-idx[-1]

            predict[f"gap_{digit}_{pos}"]=gap

    prev=dataset.iloc[-1]

    predict["prev_d1"]=prev["d1"]
    predict["prev_d2"]=prev["d2"]
    predict["prev_d3"]=prev["d3"]
    predict["prev_d4"]=prev["d4"]

    return pd.DataFrame([predict]).fillna(0)


# -----------------------
# STREAMLIT UI
# -----------------------

st.title("🎯 Singapore 4D AI Predictor")

st.write("Machine learning ensemble using XGBoost + LSTM")

if st.button("Generate Prediction"):

    with st.spinner("Loading models..."):

        xgb_models, lstm_model, feature_columns = load_models()

    df = load_data()

    dataset = build_dataset(df)

    X_pred = build_prediction_features(dataset)

    X_pred = X_pred.reindex(columns=feature_columns, fill_value=0)

    # -----------------------
    # XGB PROBABILITIES
    # -----------------------

    digit_probs_xgb={}

    for pos in ["d1","d2","d3","d4"]:

        probs=xgb_models[pos].predict_proba(X_pred)[0]

        digit_probs_xgb[pos]=normalize_probs(probs)

    # -----------------------
    # LSTM PROBABILITIES
    # -----------------------

    seq_len=30

    seq_data=dataset[["d1","d2","d3","d4"]].values

    seq_input=seq_data[-seq_len:].reshape(1,seq_len,4)

    lstm_out=lstm_model.predict(seq_input)[0]

    lstm_probs={
    "d1":normalize_probs(lstm_out[0:10]),
    "d2":normalize_probs(lstm_out[10:20]),
    "d3":normalize_probs(lstm_out[20:30]),
    "d4":normalize_probs(lstm_out[30:40])
    }

    # -----------------------
    # ENSEMBLE
    # -----------------------

    digit_probs={}

    for pos in ["d1","d2","d3","d4"]:

        combined=(digit_probs_xgb[pos]*0.7 + lstm_probs[pos]*0.3)

        digit_probs[pos]=normalize_probs(combined)

    # -----------------------
    # MONTE CARLO
    # -----------------------

    predictions=[]

    for _ in range(50000):

        d1=np.random.choice(np.arange(10),p=digit_probs["d1"])
        d2=np.random.choice(np.arange(10),p=digit_probs["d2"])
        d3=np.random.choice(np.arange(10),p=digit_probs["d3"])
        d4=np.random.choice(np.arange(10),p=digit_probs["d4"])

        predictions.append(f"{d1}{d2}{d3}{d4}")

    ranked=pd.Series(predictions).value_counts()

    result=ranked.head(50).reset_index()

    result.columns=["number","freq"]

    st.subheader("Top 50 Predicted Numbers")

    st.dataframe(result)

    st.subheader("Digit Probabilities")

    prob_df=pd.DataFrame({
        "Digit":list(range(10)),
        "D1":digit_probs["d1"],
        "D2":digit_probs["d2"],
        "D3":digit_probs["d3"],
        "D4":digit_probs["d4"]
    })

    st.bar_chart(prob_df.set_index("Digit"))