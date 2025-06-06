import faiss
import numpy as np
import json
import os
import logging
import pickle
import requests
import time
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key not found")

# Global variables
index = None
challenges = []
EMBEDDING_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"

# Get embeddings using OpenRouter API
def get_embeddings(texts: list) -> list:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Copilot Agent"
    }
    
    payload = {
        "model": "text-embedding-ada-002",  # Verified working model
        "input": texts
    }
    
    try:
        response = requests.post(EMBEDDING_ENDPOINT, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return [item["embedding"] for item in response.json()["data"]]
        logger.warning(f"Embedding API error: {response.status_code}")
        return [generate_simple_embedding(text) for text in texts]
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return [generate_simple_embedding(text) for text in texts]

def generate_simple_embedding(text: str) -> list:
    """Fallback embedding generation without API"""
    vec = [0.0] * 1536  # ADA uses 1536-dimensional embeddings
    for i, char in enumerate(text[:1536]):
        vec[i] = ord(char) / 256.0
    return vec

# Load challenges and create index
def load_rag_index():
    global index, challenges
    
    # Load challenges
    challenges = []
    for i in range(1, 6):
        try:
            with open(f'data/challenge{i}.json', 'r') as f:
                challenges.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading challenge{i}.json: {str(e)}")
    
    # Create index if not exists
    if not os.path.exists("rag_index.faiss"):
        logger.info("Creating new FAISS index...")
        create_index()
    else:
        logger.info("Loading existing FAISS index...")
        index = faiss.read_index("rag_index.faiss")
        if os.path.exists("challenges_cache.pkl"):
            with open("challenges_cache.pkl", "rb") as f:
                challenges = pickle.load(f)

def create_index():
    global index, challenges
    
    # Get embeddings
    descriptions = [c.get("description", "") or f"{c.get('title', '')} {c.get('overview', '')}" 
                   for c in challenges]
    
    logger.info(f"Embedding {len(descriptions)} descriptions...")
    
    # Process all texts at once
    embeddings = get_embeddings(descriptions)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create and save index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "rag_index.faiss")
    
    # Save challenges cache
    with open("challenges_cache.pkl", "wb") as f:
        pickle.dump(challenges, f)

def retrieve_similar_challenges(query: str, k: int = 3) -> list:
    if index is None:
        load_rag_index()
    
    try:
        # Embed query
        query_embedding = get_embeddings([query])[0]
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search index
        distances, indices = index.search(query_embedding, k)
        
        # Return similar challenges
        return [challenges[i] for i in indices[0] if i < len(challenges)]
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return []

# Preload index when module is imported
try:
    load_rag_index()
except Exception as e:
    logger.error(f"Failed to load RAG index: {str(e)}")