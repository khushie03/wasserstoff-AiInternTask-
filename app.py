import streamlit as st
import os
import json
import fitz  
from main import tag_text, summarize_dialogue  

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

st.title("PDF NER and Summarization Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    filename = uploaded_file.name
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    pdf_text = extract_text_from_pdf(filepath)
    ner_tags_df = tag_text(pdf_text, tags, model, xlmr_tokenizer)  
    summarized_text = summarize_dialogue(pdf_text) 
    ner_json = ner_tags_df.to_dict(orient='records')

    st.success('File successfully uploaded and processed!')
    st.subheader("Summary")
    st.write(summarized_text)
    
    st.subheader("NER Tags")
    st.json(ner_json)
else:
    st.warning("Please upload a PDF file.")
