# 🏏 Sri Lanka Cricket — Data-Driven Player Selection System

![App Preview](https://img.shields.io/badge/Status-Live-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-F7931E)

## 🌐 Live Application

Access the deployed Streamlit web application here:  
👉 **[Sri Lanka Cricket Selection Dashboard](https://sri-lanka-cricket-data-driven-player-selection-system-fappvxlh.streamlit.app/)**

---

## 📖 Project Overview

The **Sri Lanka Cricket Data-Driven Player Selection System** is an end-to-end Machine Learning pipeline and interactive dashboard designed to assist cricket selectors. It replaces traditional subjective selection methods by objectively evaluating players based on their recent T20 form window (last 10 matches) rather than career reputation.

The system leverages **Random Forest Classifiers** to assign a dynamic form label (`Poor`, `Average`, `Good`, `Excellent`) to every active Sri Lankan player and recommends the optimal Playing XI mapped to specific team roles.

### ✨ Key Features

1. **Interactive Dashboard:** Modern web application built with Streamlit (`streamlit-option-menu`, dynamic Plotly charts).
2. **Machine Learning Labels:** Automated rolling EWMA feature calculation to classify player form using trained Random Forest Models.
3. **AI-Driven XI Recommendations:** Selectors can generate the mathematically strongest XI against specific opponents at specific venues.
4. **Transparent XAI (Explainable AI):** Integration with `SHAP` (SHapley Additive exPlanations) provides waterfall charts to explicitly explain _why_ a player was rated highly or poorly, completely removing algorithmic black boxes.

---

## 📊 Dataset & Sources

The backbone of this system relies on high-quality, ball-by-ball international cricket data:

- **Source:** [Cricsheet.org](https://cricsheet.org/downloads/)
- **Data Used:** T20 Internationals (Men) & Lanka Premier League (LPL)
- **Format:** "Ashwin" structured CSV format
- **Timeframe:** Evaluates continuous rolling form windows (last 10 match averages).

_Note: The raw datasets are parsed into comprehensive per-match player statistics and then aggregated into recent form features (batting strike rates, dot ball percentages, bowling economy, etc.) found in the `data/processed/` directory._

---

## 🏗️ Project Architecture & Structure

```text
├── data/
│   ├── raw/             # Cricsheet raw CSV records
│   └── processed/       # Aggregated stats & form features
├── models/              # Saved Random Forest classifiers & scalers (.pkl)
├── notebooks/           # Jupyter notebooks for EDA, Training, & Evaluation
├── src/                 # Core Python backend (feature engineering, inference)
├── app/                 # Streamlit frontend application (`streamlit_app.py`)
├── requirements.txt     # Dependency list for deployment
└── README.md            # Project documentation
```

---

## 🚀 Setup & Installation (Local Development)

If you wish to run the project locally, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/Kavinduprabasara/Sri-Lanka-Cricket-Data-Driven-Player-Selection-System.git
cd Sri-Lanka-Cricket-Data-Driven-Player-Selection-System
```

**2. Create a virtual environment & install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the Streamlit Application:**

```bash
streamlit run app/streamlit_app.py
```

The application will launch on `http://localhost:8501`.

---

## 🧠 Machine Learning Insights

The performance classification uses a Random Forest approach because:

- **Non-Linear Relationships:** Form isn't linear. A player might have a low average but a very high boundary-hitting rate.
- **Explainability:** Tree-based models map perfectly to SHAP explainers, allowing us to generate visual proof of why a player is recommended.
- **Handling Class Imbalance:** We utilized SMOTE/class-weight balancing to ensure the model recognizes exceptional (`Excellent`) players accurately.

---

_Developed as part of a Machine Learning Assignment evaluating Data-Driven Decision Support Systems._
