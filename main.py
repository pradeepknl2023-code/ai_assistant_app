import streamlit as st
import openai

# =========================
# Page Config & Theme
# =========================
st.set_page_config(
    page_title="AI PO Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors & CSS
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
# API Key
# =========================
openai.api_key = st.secrets["OPENAI_API_KEY"]

# =========================
# Sidebar
# =========================
st.sidebar.title("🛠 AI PO Assistant")
st.sidebar.markdown(
    """
**Instructions:**  
- Ask anything related to your project or business.  
- The AI will provide clean, structured responses.  
- Scroll down for conversation history.
"""
)

# =========================
# Title
# =========================
st.title("🤖 AI PO Assistant")
st.subheader("Your Clean, Cloud-Based AI Project Assistant")

# =========================
# Conversation State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# User Input
# =========================
user_input = st.text_input("Enter your question or requirement:")

if st.button("Send"):
    if user_input.strip():
        # Save user input
        st.session_state.history.append({"role": "user", "content": user_input})

        # Call OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=st.session_state.history,
            max_tokens=300,
            temperature=0.7,
        )

        # Get assistant message
        assistant_msg = response["choices"][0]["message"]["content"]
        st.session_state.history.append({"role": "assistant", "content": assistant_msg})

# =========================
# Display Conversation
# =========================
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-msg'>You: {chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-msg'>AI: {chat['content']}</div>", unsafe_allow_html=True)
