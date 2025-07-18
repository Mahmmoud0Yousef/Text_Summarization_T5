import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import string
import time
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# تحميل الموديل والـ tokenizer
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("models/t5_summarizer_model")
    tokenizer = T5Tokenizer.from_pretrained("models/t5_summarizer_model")
    return model, tokenizer

model, tokenizer = load_model()

# إعداد الـ stopwords
stop_word = set(stopwords.words("english"))
negation_words = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "can't", "couldn't", "isn't", "aren't", "wasn't", "weren't",
    "without", "nor"
}
filtered_stopwords = stop_word - negation_words

# تنظيف النصوص
def Preprocessing_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered = [word for word in words if word not in filtered_stopwords]
    return " ".join(filtered)

# دالة التلخيص
def summarize_dialogue(dialogue, max_len, min_len):
    dialogue = Preprocessing_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# إعداد الصفحة
st.set_page_config(page_title="T5 Summarizer", layout="wide")
st.title("📝 Dialogue to Summary - T5 Summarizer")

# إعدادات في الشريط الجانبي
with st.sidebar:
    st.header("⚙️ Summarization Settings")
    max_len = st.slider("Maximum summary length", 50, 300, 150, step=10)
    min_len = st.slider("Minimum summary length", 10, 100, 30, step=5)

# رفع ملف
st.markdown("### 📁 Upload a text file (one dialogue per line):")
uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])

if uploaded_file:
    raw_lines = uploaded_file.read().decode("utf-8").splitlines()
    cleaned_lines = []
    summaries = []

    with st.spinner("⏳ Summarizing dialogues..."):
        for line in raw_lines:
            if line.strip():
                cleaned_lines.append(line)
                summary = summarize_dialogue(line, max_len=max_len, min_len=min_len)
                summaries.append(summary)

    if cleaned_lines:
        df = pd.DataFrame({"Dialogue": cleaned_lines, "Summary": summaries})
        st.success(f"✅ {len(df)} dialogues summarized.")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "bulk_summaries.csv", "text/csv")
    else:
        st.warning("⚠️ No valid dialogues found in the file.")

st.divider()

# إدخال يدوي
st.markdown("### 🧾 Paste a single dialogue to summarize:")
dialogue = st.text_area("Enter full conversation", height=250)

if st.button("🔍 Summarize"):
    if dialogue.strip():
        start_time = time.time()
        with st.spinner("Summarizing..."):
            summary = summarize_dialogue(dialogue, max_len=max_len, min_len=min_len)
        elapsed_time = time.time() - start_time

        st.success("✅ Summarization complete!")
        st.markdown("### 📄 Summary:")
        st.text_area("Result", summary, height=150)
        st.info(f"⏱️ Time taken: {elapsed_time:.2f} seconds")

        df = pd.DataFrame([{"Dialogue": dialogue, "Summary": summary}])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "summary.csv", "text/csv")
    else:
        st.warning("⚠️ Please enter a conversation.")
