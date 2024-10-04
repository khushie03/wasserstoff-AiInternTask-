import streamlit as st
from main import *
import os
import fitz
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://khushi1103p:Prusshita1234@cluster0.mjnio.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.mydb
collection = db.mycollection

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

st.set_page_config(
    page_title="NER Summarizer",
    page_icon=":robot_face:",
    layout="wide"
)

st.title("YOUR PERSONAL ASSISTANT")

image_url = "https://media.istockphoto.com/id/1948375456/photo/employee-downloading-computer-files-or-installing-software-on-laptop-computer-but-the.webp?a=1&b=1&s=612x612&w=0&k=20&c=Ef0t30STrpb_l6Lc5aQ6URoIqKy6nbihe99y1eEcn7U="  
st.image(image_url, width=400)  

selected_page = st.sidebar.selectbox(
    "Select a page:",
    ["PDF Summarization", "View Database"]
)

if selected_page == "PDF Summarization":
    st.header("PDF Summarization and NER Tagging Tool")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_text = extract_text_from_pdf(filepath)

        if pdf_text.strip():
            ner_tags_df = tag_text(pdf_text, tags, model, xlmr_tokenizer)
            summarized_text = summarize_dialogue(pdf_text)
            ner_tags_df.columns = ['Token', 'Tag']

            document = {
                "pdf_file_name": filename,
                "ner_tags": ner_tags_df.to_dict(orient="records"),
                "summarized_text": summarized_text
            }
            
            collection.insert_one(document)

            st.success('File successfully uploaded and processed!')
            st.subheader("Summary")
            st.write(summarized_text)
            st.subheader("NER Tags")
            st.dataframe(ner_tags_df)
        else:
            st.warning("The uploaded PDF file is empty or couldn't be processed.")
    else:
        st.warning("Please upload a PDF file.")

elif selected_page == "View Database":
    st.header("View Database Entries")
    entries = collection.find()
    pdf_file_names = [entry['pdf_file_name'] for entry in entries]

    selected_files = st.multiselect("Select PDF files to view their entries", options=["All"] + pdf_file_names)

    if selected_files:
        if "All" in selected_files:
            selected_entries = collection.find()
        else:
            selected_entries = collection.find({"pdf_file_name": {"$in": selected_files}})
        
        data = []
        for entry in selected_entries:
            data.append({
                "PDF File Name": entry['pdf_file_name'],
                "Summarized Text": entry['summarized_text'],
                "NER Tags": entry['ner_tags']
            })
        
        if data:
            st.table(pd.DataFrame(data))
        else:
            st.warning("No entries found for the selected files.")
