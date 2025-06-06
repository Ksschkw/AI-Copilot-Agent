import faiss
import numpy as np
from huggingface_hub import InferenceClient
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
client = InferenceClient(api_key=API_KEY)

# Load dummy challenges
challenges = []
for i in range(1, 4):
    with open(f'data/challenge{i}.json', 'r') as f:
        challenge = json.load(f)
        challenges.append(challenge)

# Embed descriptions
descriptions = [c['description'] for c in challenges]
embeddings = []
for desc in descriptions:
    response = client.feature_extraction(desc, model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings.append(response)
embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_similar_challenges(query, k=2):
    query_embedding = client.feature_extraction(query, model="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    similar_challenges = [challenges[i] for i in indices[0]]
    return similar_challenges