from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
import json
from main import *
import fitz  
from main import tag_text, summarize_dialogue  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['pdf_file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        pdf_text = extract_text_from_pdf(filepath)
        ner_tags_df = tag_text(pdf_text, tags, model, xlmr_tokenizer)  
        summarized_text = summarize_dialogue(pdf_text) 
        print(summarized_text)
        ner_json = ner_tags_df.to_dict(orient='records')
        flash('File successfully uploaded and processed!', 'success')
        print(ner_json)
        return redirect(url_for('view_results', filename=filename, ner_tags=json.dumps(ner_json), summary=summarized_text))
    else:
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_results(filename):
    ner_tags = request.args.get('ner_tags', [])
    summary = request.args.get('summary', '')
    ner_tags = json.loads(ner_tags) if isinstance(ner_tags, str) else ner_tags

    return render_template('view_results.html', filename=filename, ner_tags=ner_tags, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)