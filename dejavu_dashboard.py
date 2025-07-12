import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import numpy as np
import ollama
import os
import datetime

# --- FAISS + LLM Utilities ---
def get_top_k_similar_users(query_vector, user_ids, faiss_index, k=5):
    D, I = faiss_index.search(np.array([query_vector]), k)
    similar_users = [user_ids[i] for i in I[0] if i < len(user_ids)]
    return similar_users

def build_prompt(current_user_id, similar_user_ids, df):
    current = df[df["user_id"] == current_user_id].to_dict(orient="records")[0]
    similar = df[df["user_id"].isin(similar_user_ids)].to_dict(orient="records")

    prompt = f"""You are an anti-fraud analyst. The following user is suspected of return fraud:

User: {current_user_id}
Behavior Summary: {current}

Top Similar Users (by device/IP/return behavior):
"""
    for i, sim_user in enumerate(similar, 1):
        prompt += f"\n{i}. User ID: {sim_user['user_id']} - Behavior: {sim_user}"

    prompt += "\n\nAssess the likelihood of fraud. Explain why based on device/IP overlaps, return timing, and past similar users."
    return prompt

def get_llm_fraud_risk(prompt, model="llama3"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"❌ Ollama Error: {e}"

def log_admin_action(user_id, action):
    log_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "action": action
    }])
    log_entry.to_csv("admin_actions_log.csv", mode='a', header=not os.path.exists("admin_actions_log.csv"), index=False)

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="DejaVuAI: Fraud Trust Score Dashboard")
st.title("DejaVuAI - Return Fraud Detection Admin Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("trust_scores_with_tiers.csv", skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df['user_id'] = df['user_id'].astype(str)
        df.dropna(subset=["user_id", "final_trust_score"], inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return pd.DataFrame()

df = load_data()

@st.cache_resource
def build_faiss_and_embeddings(df, dim=128):
    user_embeddings = {
        row["user_id"]: np.random.rand(dim).astype('float32')
        for _, row in df.iterrows()
    }
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(list(user_embeddings.values())))
    return user_embeddings, index

st.session_state["user_embeddings"], st.session_state["faiss_index"] = build_faiss_and_embeddings(df)

# --- Sampling for Display ---
def sample_trust_tiers(df, seed=42, sample_size=2):
    np.random.seed(seed)
    sampled_df = (
        df.groupby("trust_tier", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), sample_size)))
        .reset_index(drop=True)
    )
    return sampled_df

if "sample_seed" not in st.session_state:
    st.session_state["sample_seed"] = 42

if st.button("Refresh Sample"):
    st.session_state["sample_seed"] = np.random.randint(0, 1_000_000)

sample_df = sample_trust_tiers(df, seed=st.session_state["sample_seed"])

# --- Sidebar ---
with st.sidebar:
    st.header("Filter Options")
    selected_tier = st.selectbox("Select Trust Tier", ["All"] + sorted(df["trust_tier"].dropna().unique()))
    user_search = st.text_input("Search by User ID")
    enable_llm = st.checkbox("Enable LLM Fraud Assessment")
    ollama_model = st.selectbox("Ollama Model", ["llama3", "mistral", "phi3"])

# --- Metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Trusted", df[df["trust_tier"] == "Trusted"].shape[0])
with col2:
    st.metric("Watchlist", df[df["trust_tier"] == "Watchlist"].shape[0])
with col3:
    st.metric("Banned", df[df["trust_tier"] == "Banned"].shape[0])

# --- Sample Preview ---
st.markdown("### Sampled User Trust Scores")
st.dataframe(sample_df.style.background_gradient(cmap="RdYlGn_r", subset=["final_trust_score"]))

# --- Optional Filtering ---
display_df = df.copy()
if selected_tier != "All":
    display_df = display_df[display_df["trust_tier"] == selected_tier]
if user_search:
    display_df = display_df[display_df["user_id"].str.contains(user_search, case=False)]

# --- User Inspection ---
st.markdown("---")
st.markdown("### Inspect Specific User")
if not display_df.empty:
    selected_user = st.selectbox("Select User ID to Inspect", display_df["user_id"].unique())
    user_row = df[df["user_id"] == selected_user].iloc[0]

    st.markdown(f"**User ID**: `{user_row['user_id']}`")
    st.markdown(f"**Final Trust Score**: `{user_row['final_trust_score']:.4f}`")
    st.markdown(f"**Classifier Score**: `{user_row['fraud_model_score']:.4f}`")
    st.markdown(f"**Graph Similarity Score**: `{user_row['graph_similarity_score']:.4f}`")
    st.markdown(f"**Trust Tier**: `{user_row['trust_tier']}`")

    st.markdown("#### Admin Actions (Simulated)")
    if st.button("Flag for Manual Review"):
        st.success("User flagged for review!")
        log_admin_action(user_row['user_id'], "Flagged for Review")

    if st.button("Mark as Trusted"):
        st.info("User moved to Trusted (not persisted).")
        log_admin_action(user_row['user_id'], "Marked as Trusted")

    if enable_llm:
        st.markdown("#### LLM-based Fraud Assessment")
        embedding = st.session_state["user_embeddings"].get(selected_user)
        if embedding is not None:
            similar_ids = get_top_k_similar_users(
                embedding,
                list(st.session_state["user_embeddings"].keys()),
                st.session_state["faiss_index"]
            )
            prompt = build_prompt(selected_user, similar_ids, df)
            if st.button("Run LLM Risk Assessment"):
                result = get_llm_fraud_risk(prompt, model=ollama_model)
                st.markdown("**LLM Risk Assessment:**")
                st.write(result)
        else:
            st.warning("No embedding found for this user.")

# --- Trust Score Distribution ---
st.markdown("---")
st.markdown("### Trust Score Distribution by Tier")
fig, ax = plt.subplots()
sns.histplot(df, x="final_trust_score", hue="trust_tier", multiple="stack", palette="Set2", ax=ax)
st.pyplot(fig)