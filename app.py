
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
st.write("✅ App Started Successfully")
# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="PAMS Chatbot", page_icon="🤖")
st.title("🤖 PAMS Help Chatbot")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    import pandas as pd
    
    try:
        df = pd.read_csv("Principal Schedule Queries.csv", encoding='latin1')
        df.columns = df.columns.str.strip()
        df['User Query'] = df['User Query'].str.lower()
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# -----------------------------
# VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['User Query'])
# -----------------------------
# CHATBOT LOGIC
# -----------------------------
def get_answer(user_input):
    user_input = user_input.lower()

    # ✅ Exact match
    if user_input in df['User Query'].values:
        return df[df['User Query'] == user_input]['Structured Answer'].values[0]

    # ✅ Similarity match
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    idx = similarity.argmax()
    score = similarity[0][idx]

    if score > 0.35:
        return df.iloc[idx]['Structured Answer']
    else:
        return "❌ Sorry, I couldn’t find a matching answer."

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# INPUT BOX
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    response = get_answer(user_input)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Options")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

    st.write("📊 Total Q&A Loaded:", len(df))
