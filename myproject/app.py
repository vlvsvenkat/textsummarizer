from flask import Flask, request, render_template, redirect
import PyPDF2
from transformers import pipeline

app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def read_pdf(file):
    # Initialize PdfReader
    reader = PyPDF2.PdfReader(file)
    number_of_pages = len(reader.pages)
    full_text = ""
    # Extract text from each page
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        full_text += page.extract_text()
    return full_text

def summarize_text(text):
    # Use the summarizer to summarize the text
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    # Read and summarize the PDF file
    text = read_pdf(file)
    summary = summarize_text(text)
    # Render the result.html template with the summary
    return render_template('result.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
