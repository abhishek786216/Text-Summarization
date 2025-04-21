import streamlit as st
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(page_title="Text Summarizer", layout="centered")

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# App title
st.title("📝 Text Summarizer using BART")

# Text input
user_input = st.text_area("Enter the text you want to summarize:", height=300)

# Parameters
max_length = st.slider("Max summary length", 50, 300, 130)
min_length = st.slider("Min summary length", 10, 100, 30)

# Summarize button
if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=max_length, min_length=min_length, do_sample=False)
            st.subheader("🧾 Summary:")
            st.success(summary[0]['summary_text'])
