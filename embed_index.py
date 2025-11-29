import os
import json
import numpy as np
from pathlib import Path
import faiss
import google.generativeai as genai # CORRECTED: Import the main module

# Load .env file to get the API key
from dotenv import load_dotenv
load_dotenv()

# CORRECTED: Configure the genai module directly
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Paths
CHUNKS_FILE = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'chunks.jsonl'
INDEX_FILE = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'faiss.index'
METADATA_FILE = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'metadata.jsonl'

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using the generative AI SDK.
    """
    # IMPROVED: Embed the entire batch in a single API call for efficiency.
    response = genai.embed_content(
        model="models/text-embedding-004",  # CORRECTED: Must match the agent's model
        content=texts,
        task_type="retrieval_document"      # CORRECTED: Use "retrieval_document" for docs
    )
    # The response['embedding'] contains a list of embeddings for the batch.
    return response['embedding']

def build_index(chunks_file=CHUNKS_FILE):
    """
    Load chunks, generate embeddings, build FAISS index, and store metadata.
    """
    texts = []
    metas = []
    
    # Load chunks
    print("Loading text chunks from file...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            texts.append(j['text'])
            metas.append({
                "id": j['id'],
                "source": j['source'],
                "chunk_index": j['chunk_index'],
                "text": j['text']
            })
    
    if not texts:
        raise RuntimeError('No chunks found. Run the data ingestion script first.')

    print(f"Found {len(texts)} text chunks to embed.")

    # Embed in batches
    batch_size = 100 # Gemini API can handle up to 100 texts per call
    embeddings = []
    print("Generating embeddings in batches...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = embed_texts(batch)
        embeddings.extend(embs)
        print(f"  Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    # Convert to FAISS format
    dim = len(embeddings[0])
    xb = np.array(embeddings).astype('float32')

    # Normalize vectors for cosine similarity (using Inner Product index)
    faiss.normalize_L2(xb)

    # Build FAISS index
    print(f"Building FAISS index with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)  # Inner Product is equivalent to cosine similarity on normalized vectors
    index.add(xb)
    faiss.write_index(index, str(INDEX_FILE))

    # Write metadata
    print("Writing metadata...")
    with open(METADATA_FILE, 'w', encoding='utf-8') as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f'\nIndex built successfully!')
    print(f'  - Index file: {INDEX_FILE} ({len(embeddings)} vectors)')
    print(f'  - Metadata file: {METADATA_FILE}')

if __name__ == "__main__":
    build_index()
