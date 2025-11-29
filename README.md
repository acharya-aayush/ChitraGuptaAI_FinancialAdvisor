# ChitraGupta 🇳🇵

An AI financial advisor for Nepal that actually knows the laws.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2_3B-orange?style=flat-square)

## What is this?

ChitraGupta is a chatbot that can answer business questions specific to Nepal. Instead of generic advice, it knows actual Nepal tax laws, VAT rates, registration requirements, etc. because we trained it on 400+ pages of official government documents..

## Quick Demo

**You:** "What's the VAT threshold in Nepal?"  
**ChitraGupta:** "VAT registration is required if your annual turnover exceeds Rs. 50 lakh. According to Nepal Income Tax Rules 2059, Article 15..."

**You:** "How much capital for a clothing store?"  
**ChitraGupta:** "Small clothing store: 5-15 lakh (25% margin), Medium: 15-50 lakh (30% margin). Peak seasons are Dashain and Tihar..."

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` and start asking questions.

*Note: First run downloads the AI model (~3GB). After that, everything runs offline.*

## Tech Stack

- **LLaMA 3.2 3B** (4-bit quantized to fit on normal computers)
- **RAG system** (finds relevant legal docs before answering)
- **FAISS** (super fast document search)
- **Flask** (simple web interface)

## Why This?

Instead of just asking AI to guess, we:
1. Find the most relevant legal documents for your question
2. Give those documents + your question to the AI
3. AI answers based on actual facts, not hallucinations

## Team

**Aayush Acharya**
LinkedIn: https://www.linkedin.com/in/acharyaaayush
GitHub: https://github.com/acharya-aayush   
Instagram: https://www.instagram.com/acharya.404 
Email: acharyaaayush2k4@gmail.com

**Nidhi Pradhan** 
LinkedIn: https://www.linkedin.com/in/nidhi-pradhan-79bb6a257/

**Suravi Paudel**
LinkedIn: https://www.linkedin.com/in/suravi-poudel-115713311/

**Mentor: Er. Sujan Sharma**  
LinkedIn: https://www.linkedin.com/in/sujan-sharma45/

Built for the Gen AI workshop.

## Project Structure

```
app.py                     # Flask web app
src/ultra_premium_advisor.py   # AI engine
data/processed/            # Legal documents + embeddings
templates/index.html       # Web interface
```

## What Makes This Special

Most AI chatbots give generic business advice. ChitraGupta knows Nepal's specific laws, current tax rates (2025), and industry benchmarks. It's like having a Nepalese business consultant that never sleeps.

---

**Contact:**  
Aayush Acharya: acharyaaayush2k4@gmail.com  
LinkedIn: https://www.linkedin.com/in/acharyaaayush/
