from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import logging
import uuid
import base64
from dotenv import load_dotenv
from rag import retrieve_similar_challenges
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise HTTPException(status_code=500, detail="OpenRouter API key not found")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state management
sessions: Dict[str, Dict] = {}

class ScopeInput(BaseModel):
    user_input: str
    platform: Optional[str] = None
    challenge_type: Optional[str] = None
    session_id: Optional[str] = None

class FieldInput(BaseModel):
    field: str
    value: str
    reasoning: str
    session_id: str

class SessionData(BaseModel):
    session_id: str
    state: Dict[str, Any]

# OpenRouter endpoint
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Schema loading
def load_schema(platform: str, challenge_type: str) -> Dict:
    key = f"{platform.lower().replace(' ', '_')}_{challenge_type.lower().replace(' ', '_')}"
    
    schema_map = {
        "topcoder_design": "topcoder_design.json",
        "kaggle_data_science": "kaggle_data_science.json",
        "herox_innovation": "herox_innovation.json",
        "zindi_ai_challenge": "zindi_ai.json"
    }
    
    filename = schema_map.get(key)
    if filename:
        try:
            with open(f"schemas/{filename}", "r") as f:
                return json.load(f)
        except:
            pass
    
    # Default schema
    return {
        "platform": platform,
        "challenge_type": challenge_type,
        "fields": {
            "title": {"required": True},
            "description": {"required": True},
            "timeline": {"required": True},
            "prizes": {"required": True}
        }
    }

# Initialize session
def init_session(session_id: str) -> Dict:
    sessions[session_id] = {
        "fields": {},
        "reasoning_trace": [],
        "platform": None,
        "challenge_type": None,
        "chat_history": [],
        "rag_context": []
    }
    return sessions[session_id]

# Get or create session
def get_session(session_id: Optional[str]) -> Dict:
    if not session_id or session_id not in sessions:
        return init_session(str(uuid.uuid4()))
    return sessions[session_id]

# Query OpenRouter model
def query_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Copilot Agent"
    }
    
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",  # Verified working model
        "messages": [{
            "role": "system",
            "content": "You are an AI Copilot Agent helping users define challenge specs."
        }, {
            "role": "user",
            "content": prompt
        }],
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        
        error = response.json().get("error", {})
        return f"AI service error: {response.status_code} - {error.get('message', '')}"
    
    except Exception as e:
        logger.error(f"Network error: {str(e)}")
        return "Network issue detected. Please check your connection."

# Generate response with context
def generate_response(user_input: str, session: Dict) -> str:
    # Build context
    context = {
        "platform": session["platform"],
        "challenge_type": session["challenge_type"],
        "current_fields": session["fields"],
        "rag_context": session["rag_context"][-3:] if session["rag_context"] else "No context"
    }
    
    # Format prompt
    prompt = f"""
    Current Context:
    {json.dumps(context, indent=2)}
    
    Chat History:
    {session['chat_history'][-5:] if session['chat_history'] else 'No history'}
    
    User: {user_input}
    Assistant:"""
    
    # Query model
    response = query_openrouter(prompt)
    
    # Update chat history
    session["chat_history"].append(f"User: {user_input}")
    session["chat_history"].append(f"Assistant: {response}")
    if len(session["chat_history"]) > 10:
        session["chat_history"] = session["chat_history"][-10:]
    
    return response

# API Endpoints
@app.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4())
    init_session(session_id)
    return {"session_id": session_id}

@app.post("/scope")
async def scope_dialogue(input: ScopeInput):
    try:
        session = get_session(input.session_id)
        if input.platform: session["platform"] = input.platform
        if input.challenge_type: session["challenge_type"] = input.challenge_type
        
        response = generate_response(input.user_input, session)
        sessions[input.session_id] = session
        return {"response": response, "session_id": input.session_id}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"response": "System error. Please try again.", "session_id": input.session_id}

@app.post("/add_field")
async def add_field(input: FieldInput):
    try:
        session = get_session(input.session_id)
        session["fields"][input.field] = input.value
        session["reasoning_trace"].append({
            "field": input.field,
            "source": input.reasoning,
            "confidence": 0.9
        })
        return {"message": f"Added {input.field}", "session_id": input.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_spec")
async def generate_spec(session_id: str):
    session = get_session(session_id)
    return {
        "fields": session["fields"],
        "reasoning_trace": session["reasoning_trace"],
        "platform": session["platform"],
        "challenge_type": session["challenge_type"]
    }

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), session_id: str = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")
    
    session = get_session(session_id)
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    session["reasoning_trace"].append({
        "field": "image_context",
        "source": "User uploaded image",
        "confidence": 0.8,
        "image": base64_image
    })
    
    session["rag_context"].append("User uploaded a mockup image")
    return {"message": "Image processed", "session_id": session_id}

@app.post("/retrieve_similar")
async def retrieve_similar(query: str, session_id: str):
    session = get_session(session_id)
    similar_challenges = retrieve_similar_challenges(query)
    session["rag_context"].extend([c["description"] for c in similar_challenges])
    return similar_challenges

@app.get("/load_schema")
async def load_challenge_schema(platform: str, challenge_type: str):
    return load_schema(platform, challenge_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)