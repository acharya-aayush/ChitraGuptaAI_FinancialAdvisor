# Technical Documentation

## 1. System Overview
This project is a local LLM inference system with rule-based context injection.

- Inference is executed locally using llama-cpp-python and a GGUF model file.
- The backend is a Flask API server.
- Runtime context is built from static knowledge plus industry heuristics.
- A full retrieval pipeline (vector search during query time) is not active in the current Python runtime path.

## 2. Backend Architecture

### Components
- Flask server (entry point): app.py
  - Serves the chat page at /.
  - Exposes API endpoints:
    - POST /chat
    - POST /memory/clear
    - GET /memory/stats
    - GET /health

- AI engine module: standalone_advisor.py
  - Loads GGUF model with llama_cpp.Llama.
  - Performs intent classification and industry detection.
  - Builds context blocks from static dictionaries and JSON heuristic data.
  - Constructs model prompt in Llama 3 chat-token format.
  - Executes inference and maintains short in-memory conversation state.

- Config handling: config.py
  - Contains static tokens/keys.
  - Not imported by app.py or standalone_advisor.py in the current execution path.

- Data and knowledge sources
  - data/industry_heuristics.json: industry-specific context used at runtime.
  - In-code dictionary (NEPAL_KNOWLEDGE): VAT, tax, registration, licensing facts.
  - data/processed/* files exist but are not called from the current request pipeline.

### Architecture Diagram
Client
  |
  v
Flask (app.py)
  |
  v
StandaloneAdvisor (singleton)
  |-------------------------> Industry Heuristics JSON
  |-------------------------> In-code Nepal Knowledge
  |
  v
llama_cpp.Llama (GGUF local model)
  |
  v
JSON API response

## 3. End-to-End Data Flow

MANDATORY FLOW DIAGRAM

User Query
   ↓
Preprocessing
   ↓
(Chunking / Context Building if applicable)
   ↓
Embedding / Retrieval (if used)
   ↓
Prompt Construction
   ↓
LLM Inference (llama.cpp)
   ↓
Postprocessing
   ↓
Response to User

### Step-by-step behavior in this implementation
1. User Query
   - Client sends message to POST /chat.

2. Preprocessing
   - Input is validated (non-empty string).
   - Intent is classified via keyword matching.
   - Industry is detected via keyword groups and stored in current session state.

3. Chunking / Context Building
   - Runtime does not chunk source documents during query execution.
   - Context is assembled from:
     - In-code Nepal knowledge dictionary
     - Matching industry entry from data/industry_heuristics.json

4. Embedding / Retrieval
   - No embedding computation or vector retrieval is executed in chat().
   - Existing data/processed embeddings and FAISS files are currently unused by request-time inference.

5. Prompt Construction
   - System instruction block is created.
   - Context block is appended when available.
   - Last three conversation turns are appended as prior history.
   - Current user query is appended in Llama 3 tokenized chat format.

6. LLM Inference (llama.cpp)
   - llama_cpp.Llama instance generates text.
   - Generation parameters include max_tokens, temperature, top_p, and stop tokens.

7. Postprocessing
   - Assistant text is extracted and stripped.
   - Current interaction is appended to in-memory history (capped at 10 exchanges).
   - Metadata (intent, industry, response_time, token usage) is prepared.

8. Response to User
   - Flask returns JSON with success flag, response text, and metadata.

## 4. AI Model Integration

### Model Type and Runtime
- Model format: GGUF.
- Inference backend: llama-cpp-python (Python bindings to llama.cpp).
- Default runtime model target: models/llama3.gguf.

### Model Loading Sequence
1. Flask process starts.
2. app.py calls get_advisor() during startup.
3. StandaloneAdvisor() initializes once (singleton).
4. _find_model() checks known model paths and raises FileNotFoundError if missing.
5. Llama is created with configured runtime parameters.

### Current Llama Initialization Parameters
- n_gpu_layers = -1 (attempt full GPU offload)
- n_ctx = 4096
- n_batch = 512
- verbose = False

### GPU vs CPU Execution
- If llama-cpp-python CUDA build and compatible GPU are available, layers are offloaded to GPU.
- If not, CPU execution is possible with an appropriate llama-cpp-python build, but latency is typically higher.

### Inference Call Pattern
The runtime calls self.llm(prompt, ...) with:
- max_tokens = 512
- stop = ["<|eot_id|>", "<|end_of_text|>"]
- temperature = 0.7
- top_p = 0.9

## 5. Retrieval / Context Handling

### Current State
The active system uses direct prompting plus rule-based context injection, not full RAG retrieval.

### How context is currently prepared
- Documents and structured files exist under data/.
- At runtime, only these sources are injected into prompts:
  - NEPAL_KNOWLEDGE dictionary in code
  - Matching industry record from data/industry_heuristics.json

### Chunking and selection behavior
- No runtime chunking of long documents is performed.
- No top-k semantic retrieval is performed.
- Selection logic is keyword-driven:
  - VAT keywords -> VAT block
  - Tax keywords -> income tax block
  - Registration keywords -> company registration block
  - License keywords -> licensing block
  - Detected industry -> industry mapping block

### Context injection point
Selected context is appended into the system prompt under a RELEVANT KNOWLEDGE section before inference.

### Note on existing retrieval artifacts
Files such as data/processed/document_chunks.json, embeddings.npy, and faiss_index.bin are present, but not integrated into the chat request path in standalone_advisor.py.

## 6. Prompt Construction

### System Prompt Structure
The prompt contains:
1. Base system instruction text.
2. Optional RELEVANT KNOWLEDGE block.
3. Optional current industry note.
4. Recent conversation context (last 3 turns).
5. Current user message.

### Multi-turn memory behavior
- Conversation state is stored in process memory.
- Up to 10 exchanges are retained.
- Only the latest 3 exchanges are embedded in new prompts.
- Memory is reset by POST /memory/clear or process restart.

### Prompt Assembly Format
Llama 3 control tokens are used directly:
- <|begin_of_text|>
- <|start_header_id|>system<|end_header_id|>
- <|start_header_id|>user<|end_header_id|>
- <|start_header_id|>assistant<|end_header_id|>
- <|eot_id|>

## 7. Limitations
- Hallucination risk is present; generated content can be incorrect.
- There is no guaranteed correctness.
- Domain understanding is limited to the model behavior plus injected context.
- Performance depends on model size, llama-cpp build, and local hardware.
- No production controls: authentication, authorization, rate limiting, and audit logging are absent.
- In-memory conversation state is not durable and does not support multi-instance consistency.
- Static secrets in config.py are a security risk and should be removed from source control.

## 8. Improvement Areas
- Add a proper RAG pipeline for request-time retrieval.
- Implement document chunking with overlap and metadata attribution.
- Integrate FAISS retrieval in the active query path (or migrate to managed vector storage if needed).
- Add prompt optimization and response-grounding templates.
- Add caching for repeated queries and frequently used context blocks.
- Add evaluation metrics: retrieval hit rate, answer grounding checks, latency, and token usage statistics.
- Add persistence for conversation memory with per-session isolation.
- Add API hardening: auth, input validation policy, request limits, and structured error handling.

## 9. Execution Details

### How backend runs
1. Start command: run.bat or python app.py.
2. Flask app starts on 0.0.0.0:5000.
3. Startup attempts to initialize StandaloneAdvisor and load model before serving requests.

### How model is initialized
- Model path resolution occurs in _find_model().
- Llama object is created once and reused through a module-level singleton.
- Each /chat request uses the same in-process model instance.

### Where data is stored
- Runtime context file: data/industry_heuristics.json.
- Static rule context: defined in standalone_advisor.py.
- Model file: models/llama3.gguf.
- Optional preprocessed retrieval artifacts: data/processed/* (currently not consumed at request time).
- Memory in active flow: Python list in RAM (not persisted by default).