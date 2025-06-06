from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import requests
from rag import retrieve_similar_challenges
import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not API_KEY:
    raise HTTPException(status_code=500, detail="Hugging Face API key not found")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API endpoint
MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

# Prompt template
def format_prompt(user_input, context):
    return f"""
System: You are an AI Copilot Agent helping users define challenge specs for platforms like Topcoder or Kaggle. Respond directly to the user's input with a single question or statement to understand their goal or refine the scope. Do not assume further user responses or generate multi-turn conversations.
Context: {context}

User: {user_input}

Assistant:
"""

# In-memory state
state = {"fields": {}, "reasoning_trace": [], "platform": None, "challenge_type": None}
feedback = {}

# Schema loading
def load_schema(platform: str, challenge_type: str):
    schema_file = f"schemas/{platform.lower()}_{challenge_type.lower()}.json"
    try:
        with open(schema_file, "r") as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Schema not found")

# Feedback storage
def store_feedback(field, suggestion, accepted, reason=None):
    if field not in feedback:
        feedback[field] = []
    feedback[field].append({
        "suggestion": suggestion,
        "accepted": accepted,
        "reason": reason
    })

# Generate spec
def generate_spec():
    return {
        "fields": state["fields"],
        "reasoning_trace": state["reasoning_trace"]
    }

# API Endpoints
class ScopeInput(BaseModel):
    user_input: str
    platform: str | None = None
    challenge_type: str | None = None

@app.post("/scope")
async def scope_dialogue(input: ScopeInput):
    logger.debug(f"Received input: {input.user_input}")
    try:
        # Placeholder context until retrieve_similar_challenges is implemented
        context = [{"challenge": "placeholder"}]
        context_str = json.dumps(context, indent=2)
        logger.debug(f"Context: {context_str}")

        # Format the prompt
        prompt = format_prompt(input.user_input, context_str)

        # API call to Hugging Face
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "return_full_text": False,  # Only return generated text
                "stop": ["User:", "System:"]  # Stop before multi-turn generation
            }
        }
        response = requests.post(MODEL_ENDPOINT, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        response_text = response.json()[0]["generated_text"].strip()
        logger.debug(f"LLM response: {response_text}")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in scope_dialogue: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error invoking LLM: {str(e)}")

@app.post("/load_schema")
async def load_challenge_schema(platform: str, challenge_type: str):
    schema = load_schema(platform, challenge_type)
    state["platform"] = platform
    state["challenge_type"] = challenge_type
    return schema

@app.post("/retrieve_similar")
async def retrieve_similar(query: str):
    similar_challenges = retrieve_similar_challenges(query)
    return similar_challenges

class FeedbackInput(BaseModel):
    field: str
    suggestion: str
    accepted: bool
    reason: str | None = None

@app.post("/store_feedback")
async def store_user_feedback(input: FeedbackInput):
    store_feedback(input.field, input.suggestion, input.accepted, input.reason)
    return {"message": "Feedback stored"}

class FieldInput(BaseModel):
    field: str
    value: str | dict
    reasoning: str

@app.post("/add_field")
async def add_field(input: FieldInput):
    state["fields"][input.field] = input.value
    state["reasoning_trace"].append({
        "field": input.field,
        "source": input.reasoning,
        "confidence": 0.9
    })
    return {"message": f"Added {input.field}"}

@app.post("/generate_spec")
async def generate_challenge_spec():
    spec = generate_spec()
    return spec

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    state["reasoning_trace"].append({
        "field": "image_context",
        "source": "User uploaded image for context",
        "confidence": 0.8
    })
    return {"message": "Image uploaded and processed"}