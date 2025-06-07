# AI Copilot Agent README

## Overview
This project is aimed at building an AI Copilot Agent that assists users in defining and launching structured innovation or development challenges on platforms like Topcoder, HeroX, Kaggle, Zindi, or internal systems. The agent engages users in a dynamic scoping dialogue, refines project details, and produces a structured challenge specification using reasoning, memory, and platform-specific schemas. The primary implementation (`mainv2.py`) uses OpenRouter for LLM access (avoiding OpenAI’s card requirement) and FastAPI with LangGraph for agent orchestration. An alternative version (`main.py`) without an agent framework is included but is not the primary focus. Both versions function well, with `mainv2.py` being the recommended implementation due to its compliance with the competition’s agent framework requirement.

## Objectives
The Copilot Agent:
- Accepts high-level user input (e.g., “I want to build an app for cleaning services”).
- Initiates a contextual scoping dialogue to clarify goals, feasibility, and challenge type.
- Loads platform-specific schemas dynamically to guide spec creation.
- Integrates Retrieval-Augmented Generation (RAG) to retrieve similar challenges.
- Outputs a structured JSON spec with reasoning traces.

## Features
- **Dynamic Scoping Dialogue**: Adapts questions based on user responses and platform needs.
- **Schema Configurability**: Supports customizable challenge field definitions.
- **RAG Integration**: Enhances context with similar challenge retrieval.
- **Memory Handling**: Tracks session state and conversation history.
- **UI**: Provides a basic web interface for interaction.

## Screenshots

### Web Frontend

![Web Frontend Screenshot](./images/image.png)

This is the main interface where users interact with the AI Copilot Agent, select platforms (e.g., Topcoder, Kaggle), choose challenge types (e.g., Design, Data Science), and define challenge specifications through a chat-based dialogue.

## Agent Orchestration Setup
The agent is primarily orchestrated in `mainv2.py` using FastAPI and LangGraph, a lightweight framework for stateful workflows that meets the competition’s requirement for an agent framework. An alternative implementation, `main.py`, uses FastAPI without an agent framework but is not the primary focus.

- **Primary Implementation (`mainv2.py`)**:
  - **Backend**: FastAPI handles API endpoints and session state.
  - **Agent Framework**: LangGraph manages the scoping dialogue via a `StateGraph` with a single `agent_node`, invoking the OpenRouter LLM for responses.
  - **Session Management**: A dictionary (`sessions`) stores session data with unique IDs generated via `uuid`.
  - **API Endpoints**:
    - `/start_session`: Creates a new session.
    - `/scope`: Handles scoping dialogue with user input.
    - `/add_field`: Updates challenge spec fields.
    - `/generate_spec`: Returns the completed spec.
    - `/upload_image`: Processes image uploads for context.
    - `/retrieve_similar`: Fetches similar challenges via RAG.
    - `/load_schema`: Loads platform-specific schemas.
  - **LLM Integration**: Uses OpenRouter API (`OPENROUTER_API_KEY` from `.env`) to query the `deepseek/deepseek-chat-v3-0324:free` model, chosen for its free tier and 512-token response limit.

- **Alternative Implementation (`main.py`)**:
  - A simpler version without an agent framework, using pure FastAPI for orchestration.
  - Fully functional and compatible with the frontend but does not meet the competition’s agent framework requirement.
  - Included for reference or fallback use.

**Recommendation**: Use `mainv2.py` for its compliance with competition requirements and robust state management via LangGraph.

## Memory Handling
Memory is managed through LangGraph’s `AgentState` (in `mainv2.py`) or a `sessions` dictionary (in `main.py`):
- **Fields**: Current challenge spec data (e.g., `title`, `description`).
- **Reasoning Trace**: Logs of agent decisions (e.g., why a field was set).
- **Platform & Challenge Type**: Selected platform (e.g., "Topcoder") and type (e.g., "Design").
- **Chat History**: Last 10 user-agent exchanges for context continuity.
- **RAG Context**: Retrieved similar challenge descriptions (limited to last 3 for brevity).
- **Messages** (in `mainv2.py`): LangChain’s `MessagesState` for tracking conversation flow.

The session persists across API calls, updated dynamically as users interact, ensuring the agent recalls prior inputs and adapts accordingly.

## Prompt Strategies
Prompts are crafted dynamically in `mainv2.py` (and similarly in `main.py`) to guide the OpenRouter model:
- **Context Inclusion**: 
  - Platform and challenge type.
  - Current fields and recent RAG context.
  - Chat history (last 5 exchanges).
- **Prompt Format**:
  ```
  Current Context:
  {JSON dump of session context}
  
  User: {user_input}
  Assistant:
  ```
- **System Instruction**: The model is instructed to act as a Copilot, provide detailed responses, and ask clarifying questions without assuming constraints, capped at 512 tokens.
- **Adaptation**: Prompts evolve with user feedback, avoiding repetition (e.g., if a tech stack is rejected).

## Schema Configurability
Schemas define platform-specific fields and are loaded dynamically:
- **Storage**: JSON files in `schemas/` (e.g., `topcoder_design.json`, `kaggle_data_science.json`).
- **Loading**: The `load_schema` function maps platform and challenge type to a schema file, falling back to a default if none exists.
- **Usage**: 
  - Guides the agent’s questions (e.g., "What’s the timeline?" for a required field).
  - Tracks progress and validates the final spec.
- **Extensibility**: Add new schemas by placing JSON files in `schemas/`, making it configurable without code changes.

Example schema (`topcoder_design.json`):
```json
{
  "platform": "Topcoder",
  "challenge_type": "Design",
  "fields": {
    "title": {"required": true},
    "overview": {"required": true},
    "objectives": {"required": true},
    "timeline": {"required": true},
    "prize_structure": {"required": true}
  }
}
```

## UI
The UI is a simple HTML page (`index.html`) with JavaScript:
- **Chat Interface**: Displays conversation history and accepts user input.
- **Schema Fields**: Shows spec progress and current fields.
- **Controls**: Dropdowns for platform/type selection, buttons for new sessions, and image uploads.
- **Interaction**: JavaScript fetches from FastAPI endpoints (e.g., `/scope`, `/generate_spec`).
- **Styling**: Basic CSS for layout; no React (yet).

Both `main.py` and `mainv2.py` are fully compatible with this frontend.

## RAG Integration
The RAG system (`rag.py`) retrieves similar challenges:
- **Data**: Dummy challenges in `data/` (e.g., `challenge3.json`).
- **Embedding**: Uses Hugging Face’s `sentence-transformers/all-MiniLM-L6-v2` API for embeddings, with a keyword-based fallback if the API fails.
- **Similarity Search**: Cosine similarity with a 0.6 threshold, returning top 3 matches.
- **No Vector DB**: Simple in-memory search due to small dataset size.

## File Structure
```
AI-Copilot-Agent/
├── main.py              # Original backend without agent framework
├── mainv2.py            # Primary backend with LangGraph agent framework
├── rag.py               # RAG system for similar challenge retrieval
├── schemas/             # Platform-specific schema files
│   ├── topcoder_design.json
│   ├── kaggle_data_science.json
│   └── ...
├── data/                # Dummy challenge data
│   ├── challenge1.json
│   ├── challenge2.json
│   ├── challenge3.json
│   └── ...
├── images/              # Frontend screenshot
│   └── image.png
├── .env                 # Environment variables (OPENROUTER_API_KEY)
├── requirements.txt     # Python dependencies
└── index.html           # Basic frontend UI
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ksschkw/AI-Copilot-Agent.git
   cd AI-Copilot-Agent
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   fastapi
   uvicorn
   langgraph
   langchain-core
   requests
   python-dotenv
   etc
   ```
3. **Set Environment Variables**:
   - Create a `.env` file:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key
     
     ```
4. **Run the Backend**:
   - For the primary implementation (recommended):
     ```bash
     uvicorn mainv2:app --reload
     ```
   - For the alternative (no agent framework):
     ```bash
     uvicorn main:app --reload
     ```
5. **Access the UI**:
   - Open `index.html` in a browser or serve it locally.

## Usage
1. Open `index.html` and click "New Session".
2. Select a platform and challenge type.
3. Respond to the agent’s questions to scope your challenge.
4. Upload images (if needed) for additional context.
5. Generate the final spec once all fields are complete.

## Extending the System
- **New Schemas**: Add JSON files to `schemas/`.
- **More Data**: Expand `data/` with additional challenge examples.
- **UI Upgrade**: Integrate React for a dynamic frontend.
- **Advanced LangGraph**: Add nodes for RAG or field validation in `mainv2.py`.

## Bonus Features (Planned)
- **Vision Model**: Analyze uploaded images for context.
- **React Frontend**: Enhance UI with chat, checklist, and spec views.

This README covers the agent’s setup, functionality, and usage, meeting the competition’s deliverable requirements. The primary implementation (`mainv2.py`) uses LangGraph for agent orchestration, while `main.py` provides a functional alternative without an agent framework.