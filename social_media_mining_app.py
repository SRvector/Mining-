
"""
Social Media Interaction Pattern Mining – Single‑file Streamlit App
-----------------------------------------------------------------
Upload a CSV file with at least the following columns:

- Text           : Content of the post
- Hashtags       : Space‑ or comma‑separated hashtags (e.g., "#AI #Data")
- Platform       : Social media platform (Twitter, Instagram, etc.)
- Retweets       : Number of retweets/shares (numeric)
- Likes          : Number of likes (numeric)

The app will:
1. Clean text and split hashtags.
2. Discover association rules between hashtags (Apriori).
3. Cluster posts/users based on engagement metrics.
4. Visualise rules and clusters interactively.

Run with:
    streamlit run social_media_mining_app.py
"""

import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------------------------------------------------------------
# -------------------------- Preprocessing --------------------------
# ------------------------------------------------------------------

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of","at","by",
    "for","with","about","against","between","into","through","during","before",
    "after","above","below","to","from","up","down","in","out","on","off","over",
    "under","again","further","then","once","here","there","when","where","why","how",
    "all","any","both","each","few","more","most","other","some","such","no","nor",
    "not","only","own","same","so","than","too","very","s","t","can","will","just",
    "don","should","now","d","ll","m","o","re","ve","y","ain","aren","couldn",
    "didn","doesn","hadn","hasn","haven","isn","ma","mightn","mustn","needn","shan",
    "shouldn","wasn","weren","won","wouldn"
}

def _clean_text(text: str) -> str:
    """Remove URLs, special chars, and stop‑words."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(word for word in text.split() if word.lower() not in STOP_WORDS)

def _split_hashtags(tags: str):
    if pd.isna(tags):
        return []
    # Replace commas with spaces, split, keep tokens starting with #
    tokens = re.split(r"[\s,]+", tags.strip())
    return [t for t in tokens if t]

def preprocess_dataframe(file) -> pd.DataFrame:
    """Load & preprocess the uploaded CSV into a DataFrame."""
    df = pd.read_csv(file)
    if "Hashtags" not in df.columns:
        st.error("❌ Column 'Hashtags' not found – please check your dataset.")
        st.stop()
    df["Cleaned_Text"] = df["Text"].apply(_clean_text) if "Text" in df.columns else ""
    df["Hashtags_List"] = df["Hashtags"].apply(_split_hashtags)
    return df

# ------------------------------------------------------------------
# --------------------- Association Rule Mining --------------------
# ------------------------------------------------------------------

def discover_hashtag_rules(data: pd.DataFrame, min_sup: float = 0.01):
    """Run Apriori on hashtag transactions and return association rules."""
    transactions = data["Hashtags_List"].tolist()
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_bin = pd.DataFrame(te_ary, columns=te.columns_)
    freq_sets = apriori(df_bin, min_support=min_sup, use_colnames=True)
    if freq_sets.empty:
        return pd.DataFrame()
    rules = association_rules(freq_sets, metric="lift", min_threshold=1.0)
    # Sort by lift descending
    rules.sort_values("lift", ascending=False, inplace=True)
    # Convert frozensets to strings for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules

# ------------------------------------------------------------------
# -------------------------- Clustering ----------------------------
# ------------------------------------------------------------------

def cluster_posts(data: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """KMeans clustering on engagement metrics."""
    if "Retweets" not in data.columns or "Likes" not in data.columns:
        st.warning("⚠️ Columns 'Retweets' and 'Likes' not found – clustering skipped.")
        data["Cluster"] = -1
        return data
    df = data.copy()
    # Encode Platform if present
    if "Platform" in df.columns:
        le = LabelEncoder()
        df["Platform_Encoded"] = le.fit_transform(df["Platform"].astype(str))
    else:
        df["Platform_Encoded"] = 0
    X = df[["Retweets", "Likes", "Platform_Encoded"]].fillna(0)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df["Cluster"] = kmeans.fit_predict(X)
    return df

# ------------------------------------------------------------------
# ------------------------ Visualisations --------------------------
# ------------------------------------------------------------------

def plot_clusters_plotly(df: pd.DataFrame):
    if df["Cluster"].nunique() <= 1:
        st.info("Not enough cluster variation to plot.")
        return
    fig = px.scatter(
        df,
        x="Retweets",
        y="Likes",
        color=df["Cluster"].astype(str),
        hover_data=["Platform"] if "Platform" in df.columns else None,
        title="Clusters of Posts by Engagement"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_rules_bar(rules: pd.DataFrame, top_n: int = 20):
    if rules.empty:
        st.info("No association rules to display.")
        return
    top_rules = rules.head(top_n).copy()
    fig = px.bar(
        top_rules,
        x="antecedents",
        y="lift",
        color="confidence",
        title=f"Top {len(top_rules)} Association Rules (by Lift)",
        labels={"antecedents": "Antecedent Hashtags", "lift": "Lift"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# -----------------------------  UI --------------------------------
# ------------------------------------------------------------------

st.set_page_config(page_title="Social Media Pattern Mining", layout="wide")
st.title("🔍 Social Media Interaction Pattern Mining")

uploaded_file = st.file_uploader("📤 Upload your social‑media CSV file", type=["csv"])

with st.sidebar:
    st.header("⚙️ Settings")
    min_support = st.slider("Minimum Support (Apriori)", 0.005, 0.2, 0.01, 0.005)
    n_clusters = st.slider("Number of Clusters (KMeans)", 2, 10, 5, 1)
    top_n_rules = st.slider("Show Top N Rules", 5, 30, 20, 1)

if uploaded_file:
    df_clean = preprocess_dataframe(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df_clean.head())

    # ---------------- Association Rules ----------------
    st.subheader("🔗 Association Rules between Hashtags")
    rules_df = discover_hashtag_rules(df_clean, min_sup=min_support)
    st.write(f"Found {len(rules_df)} rules with support ≥ {min_support}")
    st.dataframe(rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].head(top_n_rules))
    plot_rules_bar(rules_df, top_n=top_n_rules)

    # ------------------- Clustering --------------------
    st.subheader("🌀 Post Clustering by Engagement")
    clustered = cluster_posts(df_clean, k=n_clusters)
    st.dataframe(clustered[["Retweets", "Likes", "Cluster"]].head())
    plot_clusters_plotly(clustered)

    # ------------------- Download Results --------------
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(clustered)
    st.download_button(
        label="💾 Download Clustered Data as CSV",
        data=csv,
        file_name="clustered_posts.csv",
        mime="text/csv"
    )

    st.success("✅ Analysis complete! Explore the sidebar to tweak parameters.")

else:
    st.info("👆 Upload a CSV to begin.")
