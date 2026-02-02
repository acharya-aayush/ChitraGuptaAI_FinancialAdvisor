# ChitraGupta

## Overview
ChitraGupta is a local Flask application that runs a GGUF Llama model with llama-cpp-python for Nepal business and tax Q&A.
The backend builds prompts from static knowledge, industry heuristics, and short in-session conversation history.

## Core Functionality
- Exposes chat APIs through Flask routes.
- Loads a local GGUF model and performs direct local inference.
- Classifies query intent using keyword rules.
- Detects industry from keywords and injects matching industry heuristics into prompts.
- Injects Nepal tax and compliance context from in-code and JSON knowledge sources.
- Keeps short in-memory conversation history per process.

## System Overview (Simple Flow)
User Input -> Backend -> Model -> Response

Detailed runtime flow:
User Query -> /chat route -> StandaloneAdvisor -> llama-cpp inference -> JSON response

## Tech Stack (Backend-Focused)
- Flask (HTTP server and API routes)
- llama-cpp-python (local LLM inference)
- GGUF model file (llama3.gguf)
- Python JSON-based context data (industry heuristics)
- Optional CUDA acceleration via llama-cpp-python CUDA wheel

## Setup Instructions (VERY IMPORTANT)

### Prerequisites
- Python 3.10 or newer
- Windows shell support for batch scripts
- Optional NVIDIA GPU with CUDA (CPU execution is possible but slower)
- Model file: models/llama3.gguf

### Installation
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   python -m pip install -r requirements.txt

3. Install llama-cpp-python:

   GPU build (CUDA 12.4 wheel):
   python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

   CPU fallback:
   python -m pip install llama-cpp-python

4. Place the model file at models/llama3.gguf.
5. Optional shortcut: run install.bat to execute package installation steps.

### Run Application
Use either command:

run.bat

or

python app.py

### Access
http://localhost:5000

## Project Structure
- app.py: Flask entry point, route handlers, startup initialization.
- standalone_advisor.py: Model loading, intent detection, context building, prompt formatting, inference.
- config.py: Static token values (not used by app.py runtime path).
- data/industry_heuristics.json: Industry mapping used for context injection.
- models/llama3.gguf: Local GGUF model file.
- install.bat: Dependency installation script.
- run.bat: Application run script.

## Limitations
- The model can produce incorrect or incomplete responses.
- This project is not a production-grade advisory system.
- Performance and output characteristics depend on local hardware and model size.
- Conversation memory is process-local and is lost on restart.
- No authentication, persistent storage, or rate limiting is implemented in the API.
- Embedding and FAISS artifacts exist under data/processed but are not used by the current runtime pipeline.

## Team
- Aayush Acharya
- Nidhi Pradhan
- Suravi Paudel
- Mentor: Er. Sujan Sharma
