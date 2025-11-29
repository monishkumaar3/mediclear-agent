import os
import json
import uuid
from pathlib import Path
import nltk

# Ensure the 'punkt' tokenizer is downloaded for sentence splitting.
# The download is safe to run multiple times and will be skipped if already present.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' tokenizer for NLTK...")
    nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# --- Configuration ---
# Define the directories for raw data and processed output
RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
OUT_FILE = PROCESSED_DIR / 'chunks.jsonl'

def extract_text_from_file(path: Path) -> str:
    """
    Reads and returns the content of a text file.
    Args:
        path: A Path object pointing to the input file.
    Returns:
        The text content of the file as a string.
    """
    print(f"  - Extracting text from: {path.name}")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"    - Error reading file {path.name}: {e}")
        return ""

def chunk_text(text: str, max_chunk_words=150):
    """
    Splits a long text into smaller chunks based on sentence boundaries.
    Args:
        text: The input text to be chunked.
        max_chunk_words: The approximate maximum number of words per chunk.
    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # Split the text into individual sentences
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split()) # Approximate word count
        
        # If adding the new sentence exceeds the max length, save the current chunk
        if current_length + sentence_length > max_chunk_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last remaining chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def ingest_folder(input_folder: Path = RAW_DIR, out_file: Path = OUT_FILE):
    """
    Processes all text files in a folder, chunks them, and saves them to a JSONL file.
    Args:
        input_folder: The directory containing raw text files.
        out_file: The path to the output JSONL file.
    """
    print(f"Starting ingestion from '{input_folder}'...")
    
    # Ensure the output directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    docs_processed = 0
    with open(out_file, 'w', encoding='utf-8') as f_out:
        # Iterate over all files in the input folder
        for file_path in input_folder.glob('*.*'):
            if file_path.is_file():
                docs_processed += 1
                text = extract_text_from_file(file_path)
                chunks = chunk_text(text)
                
                print(f"    - Found {len(chunks)} chunks.")
                
                # CORRECTED: Changed loop variable from 'chunk_text' to 'chunk'
                for i, chunk in enumerate(chunks):
                    # Create a dictionary for each chunk
                    doc = {
                        "id": str(uuid.uuid4()),
                        "source": file_path.name,
                        "chunk_index": i,
                        "text": chunk # CORRECTED: Using the new variable name
                    }
                    # Write the chunk as a JSON line to the output file
                    f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nIngestion complete. Processed {docs_processed} files.")
    print(f"Output written to: {out_file}")

if __name__ == "__main__":
    ingest_folder()

