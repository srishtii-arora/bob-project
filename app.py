"""
Penguin Species Classifier
A beginner-friendly ML project using the Palmer Penguins dataset.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

# ── 1. Load & clean data ──────────────────────────────────────────────────────

@st.cache_data
def load_and_train():
    # Palmer Penguins dataset hosted on GitHub (no extra download needed)
    url = (
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    )
    df = pd.read_csv(url)

    # Keep only the four measurement columns and the target
    df = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "species"]]
    df = df.dropna()  # remove rows with missing values

    X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
    y = df["species"]

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 3. Train a Decision Tree ──────────────────────────────────────────────
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

    return model, accuracy, cm_df


# ── 5. Streamlit UI ───────────────────────────────────────────────────────────

st.title("🐧 Penguin Species Classifier")
st.write(
    "Enter a penguin's measurements below and the model will predict its species. "
    "This is a beginner learning project and is **not** suitable for wildlife "
    "research or real-world decisions."
)

model, accuracy, cm_df = load_and_train()

# ── Model performance ─────────────────────────────────────────────────────────
with st.expander("Model performance (on test set)"):
    st.metric("Test accuracy", f"{accuracy:.1%}")
    st.write("**Confusion matrix**")
    st.dataframe(cm_df)
    st.caption(
        "Rows = actual species · Columns = predicted species · "
        "Dataset: Palmer Penguins (Gorman et al., 2014)"
    )

# ── Prediction form ───────────────────────────────────────────────────────────
st.subheader("Predict a species")

col1, col2 = st.columns(2)

with col1:
    bill_length = st.number_input(
        "Bill length (mm)", min_value=30.0, max_value=65.0, value=44.0, step=0.1
    )
    bill_depth = st.number_input(
        "Bill depth (mm)", min_value=13.0, max_value=22.0, value=17.0, step=0.1
    )

with col2:
    flipper_length = st.number_input(
        "Flipper length (mm)", min_value=170.0, max_value=235.0, value=200.0, step=1.0
    )
    body_mass = st.number_input(
        "Body mass (g)", min_value=2700.0, max_value=6300.0, value=4200.0, step=50.0
    )

if st.button("Predict species", type="primary"):
    sample = pd.DataFrame(
        [[bill_length, bill_depth, flipper_length, body_mass]],
        columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
    )
    prediction = model.predict(sample)[0]
    st.success(f"Predicted species: **{prediction}**")
