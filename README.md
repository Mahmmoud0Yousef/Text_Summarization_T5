
# Text_Summariztion_T5

This project implements a **text summarization** system that converts long dialogues into concise summaries using the **T5 transformer model**. The system includes a user-friendly **Streamlit** interface for easy interaction.

## 🔍 Project Description

The application takes in lengthy dialogue-style text (e.g., conversations, meeting transcripts) and returns a summary that captures the essential information. It leverages the power of the **T5 model (Text-to-Text Transfer Transformer)** to perform abstractive summarization.

## 🚀 Features

- Summarize dialogue into concise summaries
- Built using HuggingFace Transformers and Streamlit
- Lightweight and easy to use
- Supports both notebook-based exploration and a Streamlit web app

## 🧠 Model

- **Model used:** T5 (You can choose any variant like `t5-base`, `t5-small`, etc.)
- Fine-tuned on dialogue summarization tasks

## 🛠️ Installation

```bash
git clone https://github.com/your-username/Text_Summariztion_T5.git
cd Text_Summariztion_T5
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install transformers streamlit torch
```

## ▶️ How to Run

To launch the Streamlit app:

```bash
streamlit run app.py
```

To test or explore the model in Jupyter:

```bash
jupyter notebook Project.ipynb
```

## 📁 Project Structure

```
Text_Summariztion_T5/
│
├── app.py                  # Streamlit application
├── Project.ipynb           # Jupyter Notebook with model/testing
└── README.md               # Project documentation
```

## 📌 Notes

- Make sure you have Python 3.7+ installed.
- Internet connection is required to load pre-trained models from HuggingFace if not cached.

## 📃 License

This project is licensed under the MIT License.
