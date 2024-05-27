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

def split_text(text, max_chunk_size):
    words = text.split()
    for i in range(0, len(words), max_chunk_size):
        yield ' '.join(words[i:i + max_chunk_size])

def summarize_text(text):
    max_chunk_size = 512  # Setting chunk size to 512 tokens (adjust as needed)
    chunks = list(split_text(text, max_chunk_size))
    summaries = [summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)

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
