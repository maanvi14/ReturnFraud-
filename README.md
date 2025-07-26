##  RETURN FRAUD DETECTION VIA GRAPH EMBEDDINGS(Node2Vec) AND FAISS(COSINE SIMILARITY) 

• Built a return fraud detection system (92.7% accuracy) using XGBoost, Node2Vec embeddings, FAISS, and
cosine similarity with hybrid (60/40) ML – graph risk scoring.

• Deployed LLaMA3 via Ollama to generate explainable fraud risk assessments with trust-tier classification,
combining model insights and behavioral patterns into actionable user-level insights.




---
<img width="2880" height="1800" alt="image" src= "https://github.com/user-attachments/assets/d05221c8-bdc3-4fa0-89ce-b9d419f078b7" />
<img width="2880" height="1800" alt="image" src= "https://github.com/user-attachments/assets/908910c7-3752-4bcb-8620-3a16117df958" />
<img width="2880" height="1800" alt="image" src= "https://github.com/user-attachments/assets/0012ebdf-5de4-404b-b2ef-0263a05093c6" />
<img width="2880" height="1800" alt="image" src= "https://github.com/user-attachments/assets/ea416589-e902-4088-89e9-d8ee5b22c745" />


##  Tech Stack

*Data & Processing*
Python, Pandas, NumPy, scikit-learn

*Graph Embeddings*
NetworkX, Node2Vec

*Machine Learning Model*
XGBoost, scikit-learn

*Similarity Search*
FAISS (Facebook AI Similarity Search)

*Frontend*
Streamlit

*LLM Integration*
Ollama (LLaMA, Mistral, Phi3) - for LLM's based fraud Risk Assesment 

*Deployment*
Streamlit Cloud ([App Link](https://returnfraud.streamlit.app))

---
#  Return Fraud Detection System 

## Project Structure

```
Return-Fraud-Detection/
├── data/                       # CSV datasets (train & test)
├── preprocessing/              # Feature engineering, encoding, ratios
├── graph/                     # Graph construction, Node2Vec embedding
├── model/                     # XGBoost training and evaluation
├── similarity/                # FAISS indexing and scoring
├── dashboard/                 # Streamlit UI app
├── llm_assistant/             # LLM-based risk explanation (Ollama)
├── trust_score.py             # Hybrid scoring layer
└── main_app.py                # Streamlit app entry point
```

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ReturnFraud-Detection-Via-Graph-Embeddings
```

### Step 2: Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Data

Ensure the following files exist in the `data/` folder:

* `return_fraud_train_india.csv`
* `return_fraud_unseen_india.csv`

---

## Running the System Locally

### Step 1: Generate Graph Embeddings

```bash
python graph/generate_embeddings.py
```

### Step 2: Train Model

```bash
python model/train_xgboost.py
```

### Step 3: Compute Graph Similarities

```bash
python similarity/faiss_similarity.py
```

### Step 4: Generate Hybrid Trust Scores

```bash
python trust_score.py
```

### Step 5: Launch Streamlit Dashboard

```bash
streamlit run main_app.py
```

---

## Enable LLM-Based Risk Explanations

Install Ollama: [https://ollama.com/](https://ollama.com/)

Start an Ollama model:

```bash
ollama run llama3 # or phi3, mistral
```

Configure Streamlit app to use `llm_assistant/generate_explanation.py`

---

## Deployed App

[🔗 View Streamlit App](https://returnfraud.streamlit.app)

Explore fraud scores, inspect users, visualize fraud rings, and test LLM explanations.

---

## Features

* Behavioral + location-based feature engineering
* Graph embeddings using Node2Vec
* XGBoost-based fraud classifier
* FAISS-based similarity scoring
* Hybrid scoring engine (60% model + 40% graph)
* Trust tiering: Highly Trusted, Trusted, Watchlist, High Risk, Banned
* Fraud ring detection
* LLM-powered reasoning for explainable fraud
* Interactive dashboard via Streamlit

---

## Usage Flow

1. Upload data → Preprocessing
2. Graph embedding → Similarity scoring
3. Train model → Score generation
4. Dashboard display → LLM Fraud Assesment
