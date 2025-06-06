# AI Copilot Agent

This is a backend implementation of an AI Copilot Agent for defining and launching structured challenges, built for a challenge with a 1-day deadline (completed in ~21 hours).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Replace `"your-api-key"` in `main.py` with your OpenAI API key.
3. Run: `python main.py`

## Agent Orchestration
- **Framework**: OpenAI Assistants API
- **Features**: Memory across turns, function calling, tool integration
- **LLM**: GPT-4o (fallback to GPT-4 if needed)
- **RAG**: FAISS with OpenAI embeddings (`text-embedding-3-small`)
- **Execution**: Console-based interaction

## Memory Handling
- The Assistants API maintains conversation state (e.g., filled fields, scope).
- Feedback is stored in an in-memory dictionary in `functions.py`.

## Prompt Strategies
- Instructions guide the assistant to:
  - Conduct a scoping dialogue iteratively.
  - Use RAG for suggestions.
  - Adapt based on feedback.
  - Generate a spec with reasoning.

## Schema Configurability
- Schemas are JSON files in `schemas/` (e.g., `topcoder_design.json`).
- Loaded dynamically via `load_challenge_schema` based on platform and challenge type.

## Deliverables
- Run `main.py` and interact to generate 3 session logs (copy console output).
- Logs show user input, assistant responses, and final spec.

## Notes
- Focused on backend due to time and user's React inexperience.
- Dummy data in `data/` supports RAG; expand as needed.