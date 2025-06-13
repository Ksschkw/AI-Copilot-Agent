from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from typing import Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

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

# Session state for LangGraph
class AgentState(TypedDict):
    session_id: str
    fields: Dict[str, Any]
    reasoning_trace: list
    platform: Optional[str]
    challenge_type: Optional[str]
    chat_history: list
    rag_context: list
    messages: MessagesState

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

# Query OpenRouter model
def query_openrouter(messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Copilot Agent"
    }
    
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": messages,
        "max_tokens": 512,
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

# LangGraph Agent
def agent_node(state: AgentState) -> AgentState:
    # Build context for prompt
    context = {
        "platform": state["platform"],
        "challenge_type": state["challenge_type"],
        "current_fields": state["fields"],
        "rag_context": state["rag_context"][-3:] if state["rag_context"] else "No context"
    }
    
    # Prepare messages
    messages = [
        SystemMessage(content="You are an AI Copilot Agent helping users define challenge specs. Provide complete, detailed responses **using markdown formatting** (e.g., headers, lists, tables). Ask for clarification if the user's input is vague. Do not assume constraints. Limit replies to 512 tokens."),
        *state["messages"][-5:],  # Include last 5 messages
        HumanMessage(content=state["messages"][-1].content)  # Latest user input
    ]
    
    # Add context to the last message
    messages[-1].content = f"""
    Current Context:
    {json.dumps(context, indent=2)}
    
    User: {messages[-1].content}
    Assistant:
    """
    
    # Map LangChain messages to OpenRouter format
    def map_message_role(message):
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "user"
        return "assistant"  # Fallback for any other message type
    
    openrouter_messages = [
        {"role": map_message_role(m), "content": m.content} for m in messages
    ]
    
    # Query model
    response = query_openrouter(openrouter_messages)
    
    # Update state
    state["chat_history"].append(f"User: {messages[-1].content}")
    state["chat_history"].append(f"Assistant: {response}")
    if len(state["chat_history"]) > 10:
        state["chat_history"] = state["chat_history"][-10:]
    
    state["messages"].append(HumanMessage(content=response))
    return state

# Define LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
agent = workflow.compile()

# Session management
sessions: Dict[str, AgentState] = {}

def init_session(session_id: str) -> AgentState:
    sessions[session_id] = {
        "session_id": session_id,
        "fields": {},
        "reasoning_trace": [],
        "platform": None,
        "challenge_type": None,
        "chat_history": [],
        "rag_context": [],
        "messages": []
    }
    return sessions[session_id]

def get_session(session_id: Optional[str]) -> AgentState:
    if not session_id or session_id not in sessions:
        return init_session(str(uuid.uuid4()))
    return sessions[session_id]

# API Endpoints
@app.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4())
    init_session(session_id)
    return {"session_id": session_id}

class ScopeInput(BaseModel):
    user_input: str
    platform: Optional[str] = None
    challenge_type: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/scope")
async def scope_dialogue(input: ScopeInput):
    try:
        session = get_session(input.session_id)
        if input.platform:
            session["platform"] = input.platform
        if input.challenge_type:
            session["challenge_type"] = input.challenge_type
        
        session["messages"].append(HumanMessage(content=input.user_input))
        result = agent.invoke(session)
        
        sessions[input.session_id] = result
        return {"response": result["messages"][-1].content, "session_id": input.session_id}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"response": "System error. Please try again.", "session_id": input.session_id}

class FieldInput(BaseModel):
    field: str
    value: str
    reasoning: str
    session_id: str

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
async def upload_image(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        contents = await file.read()
        
        base64_image = base64.b64encode(contents).decode('utf-8')
        session["reasoning_trace"].append({
            "field": "image_context",
            "source": "User uploaded image for context",
            "confidence": 0.8,
            "image": base64_image
        })
        
        image_description = "User uploaded a reference image for design context"
        session["rag_context"].append(image_description)
        
        return {"message": "Image uploaded successfully", "session_id": session_id}
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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