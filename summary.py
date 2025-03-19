import PyPDF2
from transformers import pipeline

import os
print("Current working directory:", os.getcwd())

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF file.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

def summarize_text(text, max_length=150, min_length=40):
    """
    Summarizes text using a pre-trained summarization model.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # The model may require text chunks if the text is long.
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    pdf_file = "sample1.pdf"  # Ensure you have a sample.pdf in the same folder.
    
    # Extract text from the PDF.
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_file)
    if not text:
        print("No text could be extracted. Please check your PDF file.")
        exit()
    
    print("Text extraction complete. Summarizing text...")
    
    # Generate the summary.
    summary = summarize_text(text)
    
    # Output the summary.
    print("\nSummary:\n", summary)
    
    # Save the summary to a file.
    with open("summary.txt", "w") as out_file:
        out_file.write(summary)
    
    print("\nThe summary has been saved to summary.txt")
