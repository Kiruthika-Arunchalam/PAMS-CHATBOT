import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="PAMS Chatbot", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .app-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: .25rem;
        }
        .app-subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 1rem;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">💬 PAMS Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask about schedules, workflows, and system guidance.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    try:
        loaded_df = pd.read_csv("Principal Schedule Queries.csv", encoding="latin1")
        loaded_df.columns = loaded_df.columns.str.strip()
        loaded_df["User Query"] = loaded_df["User Query"].astype(str).str.lower().str.strip()
        loaded_df["Structured Answer"] = loaded_df["Structured Answer"].astype(str)
        return loaded_df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def is_greeting(message: str) -> bool:
    normalized = normalize_text(message)
    greeting_patterns = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "yo",
        "sup",
        "whats up",
        "watsup",
        "wassup",
        "whatsapp",
        "good morning",
        "good afternoon",
        "good evening",
    }
    return normalized in greeting_patterns


GREETING_RESPONSE = (
    "👋 Hi! I can help with PAMS schedule and process questions. "
    "Try asking something like: **How do I check principal schedule?**"
)

FALLBACK_RESPONSE = (
    "I couldn’t find an exact answer yet. Please rephrase your question with "
    "details like *principal name, date, class,* or *schedule type*."
)


df = load_data()
if df is None:
    st.stop()

# -----------------------------
# VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["User Query"])


# -----------------------------
# CHATBOT LOGIC
# -----------------------------
def get_answer(user_input: str):
    normalized_input = normalize_text(user_input)

    if is_greeting(normalized_input):
        return GREETING_RESPONSE

    # Exact match
    if normalized_input in df["User Query"].values:
        return df[df["User Query"] == normalized_input]["Structured Answer"].values[0]

    # Similarity match
    user_vec = vectorizer.transform([normalized_input])
    similarity = cosine_similarity(user_vec, X)

    idx = similarity.argmax()
    score = similarity[0][idx]

    if score > 0.30:
        return df.iloc[idx]["Structured Answer"]

    # Give top suggestion preview for better guidance
    top_indices = similarity[0].argsort()[::-1][:2]
    suggestions = [df.iloc[i]["User Query"] for i in top_indices if similarity[0][i] > 0.12]

    if suggestions:
        suggestion_text = "\n".join([f"- {query}" for query in suggestions])
        return f"{FALLBACK_RESPONSE}\n\nYou can also try:\n{suggestion_text}"

    return FALLBACK_RESPONSE


# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        (
            "assistant",
            "Hello! I’m your PAMS assistant. Ask me anything related to principal schedules.",
        )
    ]

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# -----------------------------
# INPUT BOX
# -----------------------------
user_input = st.chat_input("Message PAMS Assistant...")

if user_input:
    response = get_answer(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Options")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = [
            (
                "assistant",
                "Hello! I’m your PAMS assistant. Ask me anything related to principal schedules.",
            )
        ]
        st.rerun()

    st.write("📊 Total Q&A Loaded:", len(df))
    st.caption("Tip: Start with a greeting like *Hi* or ask a full schedule question.")
