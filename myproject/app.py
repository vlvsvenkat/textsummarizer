import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline
import PyPDF2

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf'}

# Specify the model explicitly
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_text(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def summarize_text(text):
    max_chunk_length = 1024  # Adjusted to a safer length for BART
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    return ' '.join(summaries)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            file.save(file_path)
        
        text = pdf_to_text(file_path)
        summary = summarize_text(text)
        
        os.remove(file_path)  # Delete the file after processing

        return render_template('result.html', summary=summary)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
