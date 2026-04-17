# 💹 RiskPulse Finance AI

> **Production-grade NLP + Deep Learning system for Financial Stress Risk Assessment**  
> Built with BiLSTM · TensorFlow/Keras · Streamlit

---

## 📌 Project Overview

RiskPulse Finance AI is an end-to-end machine learning system designed for fintech companies to assess **financial stress risk** from client or patient statements. It combines state-of-the-art NLP with mental health classification to generate actionable **Financial Stress Risk Scores** (0–100).

The system classifies text into 4 mental health categories and maps each to a financial risk band:

| Class | Financial Risk Score | Interpretation |
|---|---|---|
| Normal | 0–25 | Stable financial health |
| Anxiety | 25–60 | Moderate stress indicators |
| Depression | 60–100 | High-risk — immediate attention |
| Other | 40–70 | Uncertain — monitor closely |

---

## 📁 Project Structure

```
RiskPulse Finance AI/
│
├── data/
│   └── dataset.csv                  # Mental health text dataset
│
├── notebooks/
│   └── EDA.ipynb                    # Full Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py        # Cleaning, label mapping, train/test split
│   ├── feature_engineering.py       # Tokenizer + Padding, TF-IDF
│   ├── model.py                     # BiLSTM & baseline architectures
│   ├── train.py                     # End-to-end training pipeline
│   ├── evaluate.py                  # Metrics, confusion matrix, plots
│   └── risk_score.py                # Financial Stress Risk Score logic
│
├── app/
│   └── streamlit_app.py             # Production Streamlit UI
│
├── models/                          # Auto-generated after training
│   ├── trained_model.keras
│   ├── trained_model.h5
│   ├── best_bilstm.keras
│   ├── tokenizer.pkl
│   ├── tfidf.pkl
│   ├── label_encoder.pkl
│   ├── baseline_lr.pkl
│   ├── class_weights.json
│   ├── classes.json
│   ├── metrics.json
│   ├── training_log.csv
│   ├── X_test.npy
│   ├── y_test.npy
│   └── evaluation_results.csv
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### Primary: BiLSTM Deep Learning Model

```
Input (token IDs, max_len=150)
    ↓
Embedding Layer (vocab_size × 128)
    ↓
SpatialDropout1D (0.3)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
GlobalMaxPooling1D
    ↓
Dense (64, ReLU)  →  Dropout (0.3)
    ↓
Dense (32, ReLU)  →  Dropout (0.15)
    ↓
Dense (4, Softmax)   ← output: [Normal, Anxiety, Depression, Other]
```

**Training config:**
- Loss: `categorical_crossentropy`
- Optimizer: `Adam (lr=1e-3)`
- Class imbalance: `class_weight` (inverse frequency)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Baseline: Logistic Regression on TF-IDF
- TF-IDF (15,000 features, bigrams, sublinear_tf)
- Logistic Regression (multinomial, balanced class weights)

---

## 📊 Dataset Description

- **Source:** Combined mental health text statements
- **Raw classes (7):** Normal, Anxiety, Depression, Stress, Bipolar, Suicidal, Personality disorder
- **Mapped to 4 fintech risk classes:**
  - Normal → Normal
  - Anxiety → Anxiety
  - Depression, Stress, Bipolar, Suicidal → Depression
  - Personality disorder → Other
- **~53,000 samples** after cleaning
- **Train/Test:** 80/20 stratified split

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/riskpulse-finance-ai.git
cd riskpulse-finance-ai

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Train the Model
```bash
python src/train.py
```
This will:
- Preprocess and clean the dataset
- Fit tokenizer + TF-IDF on training data only
- Train the BiLSTM model with class weights
- Save all model artifacts to `models/`

### Step 2: Evaluate (Optional)
```bash
python src/evaluate.py
```
Generates:
- Classification report (accuracy, precision, recall, F1)
- Confusion matrix plot
- Training history plot
- Risk score distribution

### Step 3: Run EDA Notebook (Optional)
```bash
cd notebooks
jupyter notebook EDA.ipynb
```

### Step 4: Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 💡 Example Predictions

| Input Statement | Class | Risk Score | Level |
|---|---|---|---|
| "I feel great, work is going well" | Normal | 12.5 | 🟢 Low Risk |
| "I can't stop worrying about my debt" | Anxiety | 44.2 | 🟡 Medium Risk |
| "Nothing feels worth it, I've given up" | Depression | 78.3 | 🔴 High Risk |
| "My thoughts race and I feel disconnected" | Other | 55.1 | 🟡 Medium Risk |

---

## 📈 Expected Model Performance

| Metric | BiLSTM | Baseline LR |
|---|---|---|
| Accuracy | ~88–92% | ~82–85% |
| F1 (weighted) | ~88–91% | ~81–84% |

*Actual results may vary based on hardware and random seed.*

---

## ⚠️ Disclaimer

RiskPulse Finance AI is a **decision-support tool** only. Risk scores are derived from NLP analysis and should **not** replace professional medical, psychological, or financial advice. Always consult qualified practitioners for clinical or financial decisions.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x / Keras |
| NLP | Keras Tokenizer, TF-IDF |
| ML Baseline | scikit-learn |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, wordcloud |
| Web App | Streamlit |
| Imbalance | class_weight (Keras) |

---

*Built with ❤️ by the RiskPulse Finance AI team.*
