import streamlit as st
import google.generativeai as genai

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Gemini AI Engine", layout="wide")
st.title("🚀 Gemini AI Engine (Google Only)")

# -------------------------
# LOAD API KEY FROM SECRETS
# -------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Gemini API Key not found in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# MODEL - Fixed model name
# -------------------------
try:
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# -------------------------
# USER INPUT
# -------------------------
user_prompt = st.text_area("Enter your prompt:")

if st.button("Run AI"):
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(user_prompt)
                st.success("✅ Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

                # Auto-suggest fallback models on failure
                st.info("💡 Trying to list available models...")
                try:
                    available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
                    st.write("Available models:", available)
                except:
                    st.warning("Could not list models. Check your API key.")
