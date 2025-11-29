import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. SETUP & CONFIG ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Paths
INDEX_FILE = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'faiss.index'
METADATA_FILE = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'metadata.jsonl'

# Load FAISS & Metadata
try:
    index = faiss.read_index(str(INDEX_FILE))
    metadata = []
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line))
except Exception as e:
    print(f"Error loading data: {e}. Did you run ingest.py and embed_index.py?")
    index = None 
    metadata = []

# --- SAFETY SETTINGS ---
# This ensures the model answers medical questions without blocking
safety_config = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 2. THE TOOL (Retrieval) ---
def retrieve_tool(query_text, k=3):
    """Tool: Converts query to embedding and finds relevant medical docs."""
    if index is None: return "Error: Database not loaded."
    
    try:
        # We stick to text-embedding-004 as it worked in your previous step
        q_resp = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="retrieval_query"
        )
    except Exception as e:
        return f"Embedding Error: {e}"

    query_vector = np.array([q_resp['embedding']]).astype('float32')
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, k)
    
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx]['text'])
    
    return "\n\n".join(results)

# --- 3. AGENT 1: THE RESEARCHER ---
def researcher_agent(user_query):
    """Role: Fetches raw, accurate medical data."""
    facts = retrieve_tool(user_query)
    
    if not facts or "Error" in facts:
        return "NO_DATA_FOUND"

    prompt = f"""
    You are a Medical Researcher. 
    Analyze these medical notes and extract the key clinical facts regarding: "{user_query}"
    
    RAW NOTES:
    {facts}
    
    Output a concise summary of the clinical facts.
    """
    
    # UPDATED: Using a model from your available list
    model = genai.GenerativeModel("models/gemini-2.0-flash") 
    response = model.generate_content(prompt, safety_settings=safety_config)
    return response.text

# --- 4. AGENT 2: THE TRANSLATOR ---
def translator_agent(clinical_summary, history):
    """Role: Takes the Researcher's output and translates it for the patient."""
    
    prompt = f"""
    You are MediClear, an empathetic patient advocate.
    
    CONTEXT (Previous Chat):
    {history}
    
    INPUT (Clinical Facts):
    {clinical_summary}
    
    TASK:
    Explain the clinical facts above to the patient in simple, 5th-grade language.
    - Be comforting but honest.
    - Use analogies if helpful.
    - Avoid complex jargon.
    """
    
    # UPDATED: Using a model from your available list
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt, safety_settings=safety_config)
    return response.text

# --- 5. ORCHESTRATOR ---
def run_mediclear(user_query, chat_history):
    print("  [System] Agent 1 (Researcher) is analyzing records...")
    try:
        clinical_facts = researcher_agent(user_query)
    except Exception as e:
        return f"Error connecting to AI Model: {e}"

    if clinical_facts == "NO_DATA_FOUND":
        return "I couldn't find specific details in your medical records."

    print("  [System] Agent 2 (Translator) is simplifying the explanation...")
    try:
        final_answer = translator_agent(clinical_facts, chat_history)
    except Exception as e:
        return f"Error during translation: {e}"
    
    return final_answer

# --- 6. RUNNABLE DEMO ---
if __name__ == "__main__":
    print("--- MediClear Agent Started (Using Gemini 2.0 Flash) ---")
    history = []
    
    while True:
        q = input("\nPatient (You): ")
        if q.lower() in ['exit', 'quit']: break
        
        response = run_mediclear(q, history)
        history.append(f"User: {q}\nAgent: {response}")
        print(f"\nMediClear: {response}")