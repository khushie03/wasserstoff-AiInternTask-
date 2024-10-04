import streamlit as st
from main import *
import os
import fitz 
import pandas as pd

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

st.title("PDF NER and Summarization Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    filename = uploaded_file.name
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf_text = extract_text_from_pdf(filepath)

    if pdf_text and pdf_text.strip():
        try:
            ner_tags_df = tag_text(pdf_text, tags, model, xlmr_tokenizer)
            summarized_text = summarize_dialogue(pdf_text)
            ner_tags_df.columns = ['Token', 'Tag']

            st.success('File successfully uploaded and processed!')
            st.subheader("Summary")
            st.write(summarized_text)
            st.subheader("NER Tags")
            st.dataframe(ner_tags_df)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("The uploaded PDF file is empty or couldn't be processed.")
else:
    st.warning("Please upload a PDF file.")
