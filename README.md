# ChitraGupta

AI-powered Nepal Business & Tax Advisor

## Overview

ChitraGupta is a standalone AI financial advisor specializing in Nepal's business landscape. It provides expert guidance on company registration, VAT compliance, income tax, licensing requirements, and industry-specific regulations.

## Features

- Nepal-specific tax and business law guidance
- Company registration procedures and requirements
- VAT and income tax consultation
- Industry-specific advice (retail, technology, hospitality, manufacturing, etc.)
- Multi-turn conversational memory
- Completely offline operation
- GPU-accelerated inference (RTX 4050)

## Tech Stack

- **Backend**: Flask
- **AI Model**: LLaMA 3.2 3B (GGUF format)
- **Inference**: llama-cpp-python with CUDA support
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (optional but recommended)

### Setup

1. **Install dependencies**:
   ```bash
   install.bat
   ```

   This installs Flask and llama-cpp-python with GPU acceleration.

2. **Run the application**:
   ```bash
   run.bat
   ```

3. **Access the interface**:
   Open http://localhost:5000 in your browser

## Project Structure

```
ChitraGupta/
├── app.py                    # Flask application
├── standalone_advisor.py     # AI advisor engine
├── config.py                 # Configuration settings
├── models/
│   └── llama3.gguf          # LLaMA 3.2 3B model
├── templates/
│   └── premium_chat.html    # Chat interface
├── static/
│   └── chitragupta.png      # Application logo
├── data/                     # Knowledge base
├── install.bat               # One-time setup script
└── run.bat                   # Application launcher
```

## Model

The project uses LLaMA 3.2 3B Instruct in GGUF format, loaded directly via llama-cpp-python. The model is fully offloaded to GPU for optimal performance.

## Usage

1. Start the application using `run.bat`
2. Enter your business or tax-related questions
3. Receive context-aware, Nepal-specific guidance
4. Use the clear button to start a new conversation

## Built For

Gen AI Workshop

## Team

**Aayush Acharya**  
[LinkedIn](https://www.linkedin.com/in/acharyaaayush) | [GitHub](https://github.com/acharya-aayush) | [Instagram](https://www.instagram.com/acharya.404)  
Email: acharyaaayush2k4@gmail.com

**Nidhi Pradhan**  
[LinkedIn](https://www.linkedin.com/in/nidhi-pradhan-79bb6a257/)

**Suravi Paudel**  
[LinkedIn](https://www.linkedin.com/in/suravi-poudel-115713311/)

**Mentor: Er. Sujan Sharma**  
[LinkedIn](https://www.linkedin.com/in/sujan-sharma45/)

## License

MIT
