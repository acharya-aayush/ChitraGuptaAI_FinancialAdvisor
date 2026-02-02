# ChitraGupta

AI-powered Nepal Business & Tax Advisor using LLaMA 3.2

## Quick Start

### Local Development

1. **Prerequisites**: Install [Ollama](https://ollama.com) and pull the model:
   ```
   ollama pull llama3.2:3b
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run**:
   ```
   python app.py
   ```

4. Open http://localhost:5000

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Note**: Ollama must be running on your host machine. The container connects to it via `host.docker.internal:11434`.

## Project Structure

```
chitragupta/
 app.py                 # Flask application
 lightweight_advisor.py # AI advisor engine
 config.py             # Configuration
 templates/
    premium_chat.html # Chat interface
 static/
    chitragupta.png   # Logo
 data/                 # Knowledge base
 Dockerfile
 docker-compose.yml
 requirements.txt
```

## Tech Stack

- **Backend**: Flask + Gunicorn
- **LLM**: Ollama (LLaMA 3.2 3B)
- **Container**: Docker

## Features

- Nepal VAT, Income Tax, Company Registration guidance
- Multi-turn conversation with memory
- Industry-specific advice
- No heavy ML dependencies (uses Ollama API)

## License

MIT
