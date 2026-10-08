# 🐧 Penguin Species Classifier

A beginner-friendly machine-learning project that predicts the species of a
Palmer Penguin (Adélie, Chinstrap, or Gentoo) from four body measurements using
a **Decision Tree classifier** and a **Streamlit** web app.

> ⚠️ **Disclaimer:** This is a learning exercise. The model is **not** suitable
> for wildlife research, conservation work, or any real-world decisions.

---

## What it does

| Step | Description |
|------|-------------|
| Load data | Downloads the Palmer Penguins CSV from GitHub automatically |
| Clean | Drops the ~11 rows that contain missing values |
| Split | 80 % training / 20 % test |
| Train | `DecisionTreeClassifier` from scikit-learn |
| Evaluate | Prints test accuracy and a confusion matrix inside the app |
| Predict | Streamlit form — enter four measurements, get a species label |

---

## Project files

```
app.py            ← all code: data loading, training, and Streamlit UI
requirements.txt  ← Python dependencies
README.md         ← this file
```

---

## Setup

### 1 · Prerequisites

- Python 3.9 or newer
- `pip` (comes with Python)

### 2 · Install dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### 3 · Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in
your browser.

The dataset is downloaded automatically the first time the app starts (internet
connection required). After that it is cached by Streamlit so subsequent runs
are instant.

---

## Using the app

1. Expand **"Model performance (on test set)"** to see the accuracy and
   confusion matrix.
2. Enter bill length, bill depth, flipper length, and body mass in the form.
3. Click **Predict species** — the predicted label appears below the button.

---

## Dataset

**Palmer Penguins** — Gorman, Williams & Fraser (2014).  
Hosted at <https://github.com/mwaskom/seaborn-data>.  
Original paper: *Ecological Sexual Dimorphism and Environmental Variability
within a Community of Antarctic Penguins*, PLOS ONE.

---

## Learning pointers

- [`load_and_train()`](app.py) — decorated with `@st.cache_data` so the model
  is trained once and reused across UI interactions.
- [`DecisionTreeClassifier`](app.py) — one of the simplest, most interpretable
  ML models; no feature scaling is required.
- [`train_test_split`](app.py) — keeps a held-out test set so accuracy reflects
  unseen data, not training performance.
- [`confusion_matrix`](app.py) — shows which species the model confuses; a
  perfect model has non-zero values only on the diagonal.
