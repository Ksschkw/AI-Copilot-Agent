import os
import json
import requests
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_API = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_API_TOKEN", "")  
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score

class RAGSystem:
    def __init__(self):
        self.challenges = self._load_challenges()
    
    def _load_challenges(self) -> List[Dict]:
        """Load all challenge JSON files from data directory"""
        challenges = []
        for i in range(1, 6):
            try:
                with open(f'data/challenge{i}.json', 'r') as f:
                    data = json.load(f)
                    challenges.append({
                        'id': i,
                        'text': f"{data.get('title','')} {data.get('description','')}",
                        'data': data
                    })
            except Exception:
                continue
        return challenges
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from free API with fallback"""
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        
        try:
            response = requests.post(
                EMBEDDING_API,
                headers=headers,
                json={"inputs": texts},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        
        # Fallback: Simple keyword-based vectors
        all_words = list(set(word for text in texts for word in text.lower().split()))
        return [
            [text.lower().count(word) for word in all_words]
            for text in texts
        ]
    
    def find_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Find similar challenges without vector DB"""
        if not self.challenges:
            return []
        
        # Get embeddings
        texts = [c['text'] for c in self.challenges]
        text_embeddings = self._get_embeddings(texts)
        query_embedding = self._get_embeddings([query])[0]
        
        # Calculate similarities
        results = []
        for challenge, emb in zip(self.challenges, text_embeddings):
            try:
                similarity = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                if similarity > SIMILARITY_THRESHOLD:
                    results.append((similarity, challenge['data']))
            except Exception:
                continue
        
        # Return top k matches
        return [item[1] for item in sorted(results, reverse=True)[:k]]

# Initialize RAG system
rag = RAGSystem()

# Public interface
def retrieve_similar_challenges(query: str, k: int = 3) -> List[Dict]:
    return rag.find_similar(query, k)