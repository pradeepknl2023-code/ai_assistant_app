import streamlit as st
import openai

# =========================
# Page Config & Theme
# =========================
st.set_page_config(
    page_title="AI‑PO‑Assistantce",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e0f7fa);
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .user-msg {
        background-color: #ffd54f;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        font-weight: bold;
    }
    .ai-msg {
        background-color: #4fc3f7;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# OpenAI API Key
# =========================
# Streamlit Cloud: store your API key in Secrets
# Example: OPENAI_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXX"
openai.api_key = st.secrets["OPENAI_API_KEY"]

# =========================
# Sidebar
# =========================
st.sidebar.title("🛠 AI‑PO‑Assistantce")
st.sidebar.markdown(
    """
**Instructions:**  
- Ask anything about your project, user stories, or tasks.  
- The AI provides clean, structured answers.  
- Scroll down to see full conversation history.
"""
)

# =========================
# App Title
# =========================
st.title("🤖 AI‑PO‑Assistantce")
st.subheader("Your Cloud-Based AI Product Owner Assistant")

# =========================
# Conversation History
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# User Input
# =========================
user_input = st.text_input("Enter your question or requirement:")

if st.button("Send") and user_input.strip():
    # Save user input
    st.session_state.history.append({"role": "user", "content": user_input})

    try:
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.history,
            max_tokens=300,
            temperature=0.7,
        )

        assistant_msg = response.choices[0].message.content

        # Save assistant message
        st.session_state.history.append({"role": "assistant", "content": assistant_msg})

    except Exception as e:
        st.error(f"⚠️ OpenAI API error: {e}")

# =========================
# Display Conversation
# =========================
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-msg'>You: {chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-msg'>AI: {chat['content']}</div>", unsafe_allow_html=True)
