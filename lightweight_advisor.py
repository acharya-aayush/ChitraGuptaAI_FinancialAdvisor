"""
Lightweight ChitraGupta Advisor - No heavy dependencies
Uses only Ollama for LLM inference without sentence transformers
"""

import os
import json
import logging
import re
import requests
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intent constants
INTENT_NEPAL_LAW = "NEPAL_LAW"
INTENT_GENERAL_FINANCE = "GENERAL_FINANCE"
INTENT_CHIT_CHAT = "CHIT_CHAT"

# Nepal business knowledge base (built-in, no RAG needed)
NEPAL_KNOWLEDGE = {
    "vat": {
        "rate": "13%",
        "threshold": "NPR 5 million annual turnover",
        "filing": "Monthly by 25th",
        "registration": "Inland Revenue Department (IRD)"
    },
    "income_tax": {
        "corporate": "25%",
        "manufacturing": "20%",
        "it_export": "1% on export earnings",
        "individual_max": "36%"
    },
    "company_registration": {
        "authority": "Office of Company Registrar (OCR)",
        "pvt_ltd_min_capital": "NPR 100,000",
        "pub_ltd_min_capital": "NPR 10,000,000",
        "timeline": "1-2 weeks",
        "documents": ["MOA", "AOA", "Citizenship copies", "Address proof"]
    },
    "business_structures": {
        "sole_proprietorship": "DAO registration, unlimited liability",
        "partnership": "OCR registration, shared liability",
        "pvt_ltd": "OCR registration, limited liability, 1-101 shareholders",
        "pub_ltd": "OCR + SEBON, can offer public shares"
    },
    "licenses": {
        "pan": "Mandatory for all businesses - free at IRD",
        "vat": "Required if turnover > NPR 5 million",
        "industry": "Department of Industry for manufacturing",
        "food": "DFTQC license for food businesses",
        "import_export": "Customs + DOI license"
    }
}


class LightweightAdvisor:
    """Simple advisor using only Ollama - no heavy ML dependencies"""
    
    def __init__(self):
        logger.info("Initializing Lightweight ChitraGupta Advisor...")
        
        # Support Docker environment variable for Ollama host
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_url = f"{ollama_host}/api/generate"
        self.model_name = "llama3.2:3b"
        
        # Load industry heuristics
        self.industry_heuristics = self._load_industry_heuristics()
        
        # Session memory (simple list)
        self.conversation_history = []
        self.current_industry = None
        
        # Test Ollama connection
        self._test_ollama()
        
        logger.info("✅ Lightweight Advisor ready!")
    
    def _load_industry_heuristics(self) -> Dict:
        """Load industry-specific knowledge"""
        path = Path(__file__).parent / "data" / "industry_heuristics.json"
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load industry heuristics: {e}")
        return {"industry_mappings": {}}
    
    def _test_ollama(self):
        """Test Ollama connection"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": "test", "stream": False},
                timeout=30
            )
            if response.status_code == 200:
                logger.info(f"✅ Connected to Ollama ({self.model_name})")
            else:
                raise Exception(f"Ollama returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            raise
    
    def classify_intent(self, query: str) -> str:
        """Quick intent classification"""
        q = query.lower()
        
        # Chit-chat patterns
        if re.match(r'^(hi|hello|hey|namaste|thanks|bye)[\s!?.]*$', q):
            return INTENT_CHIT_CHAT
        
        # Nepal law keywords
        law_kw = ['vat', 'tax', 'pan', 'register', 'license', 'company', 'pvt', 'ltd', 
                  'ocr', 'ird', 'customs', 'duty', 'compliance', 'nepal']
        if sum(1 for kw in law_kw if kw in q) >= 2:
            return INTENT_NEPAL_LAW
        
        return INTENT_GENERAL_FINANCE
    
    def detect_industry(self, query: str) -> Optional[str]:
        """Detect industry from query"""
        q = query.lower()
        
        industry_map = {
            'software': ['software', 'app', 'tech', 'it', 'coding', 'saas'],
            'restaurant': ['restaurant', 'cafe', 'food', 'eatery', 'kitchen'],
            'retail': ['shop', 'store', 'retail', 'sell', 'electronics', 'smartphone'],
            'import_export': ['import', 'export', 'trading', 'customs'],
            'healthcare': ['clinic', 'hospital', 'medical', 'pharmacy', 'health'],
            'education': ['school', 'training', 'coaching', 'institute'],
            'construction': ['construction', 'building', 'contractor', 'real estate'],
            'consulting': ['consulting', 'consultancy', 'advisory', 'freelance']
        }
        
        for industry, keywords in industry_map.items():
            if any(kw in q for kw in keywords):
                self.current_industry = industry
                return industry
        
        return self.current_industry  # Return sticky context if no new match
    
    def _build_knowledge_context(self, industry: Optional[str]) -> str:
        """Build relevant knowledge context"""
        ctx = """NEPAL BUSINESS FACTS:
• VAT: 13% (register if turnover > NPR 5 million)
• Corporate Income Tax: 25%
• PAN: Mandatory for ALL businesses (free at IRD)
• Private Limited: Min capital NPR 100,000, register at OCR
• Registration timeline: 1-2 weeks
"""
        
        if industry:
            ind_data = self.industry_heuristics.get('industry_mappings', {}).get(industry, {})
            if ind_data:
                ctx += f"""
SPECIFIC TO {industry.upper()}:
• Structure: {ind_data.get('recommended_structure', 'Private Limited')}
• Special licenses: {', '.join(ind_data.get('required_licenses', ['Standard business license'])[:3])}
• Key rules: {'; '.join(ind_data.get('specific_rules', [])[:2])}
"""
        return ctx
    
    def chat(self, user_message: str) -> tuple:
        """Main chat function"""
        start_time = time.time()
        
        # Classify intent
        intent = self.classify_intent(user_message)
        
        # Detect industry
        industry = self.detect_industry(user_message)
        
        # Handle chit-chat directly
        if intent == INTENT_CHIT_CHAT:
            response = self._handle_chitchat(user_message)
            return response, {'intent': intent, 'response_time': time.time() - start_time}
        
        # Build prompt
        knowledge = self._build_knowledge_context(industry)
        
        prompt = f"""You are ChitraGupta, Nepal's expert business advisor. Give DIRECT, HELPFUL answers.

RULES:
1. START with specific advice - NO questions first
2. Include exact numbers (VAT 13%, corporate tax 25%, etc.)
3. Use bullet points for steps
4. Maximum ONE clarifying question at the END

{knowledge}

USER QUESTION: {user_message}

Give a direct, actionable answer:"""

        # Generate response
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 600,
                        "num_ctx": 4096
                    }
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Clean up any system prompt leakage
                result = self._clean_response(result)
            else:
                result = "I'm having trouble connecting to my AI backend. Please try again."
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            result = f"Error generating response: {str(e)}"
        
        # Store in history
        self.conversation_history.append({
            'user': user_message,
            'assistant': result[:500],
            'industry': industry
        })
        
        metadata = {
            'intent': intent,
            'industry_detected': industry,
            'response_time': time.time() - start_time,
            'total_interactions': len(self.conversation_history)
        }
        
        return result, metadata
    
    def _handle_chitchat(self, msg: str) -> str:
        """Handle greetings and simple messages"""
        m = msg.lower()
        if any(g in m for g in ['hi', 'hello', 'hey', 'namaste']):
            return """Namaste! 🙏 I'm ChitraGupta, your Nepal business expert.

I can help you with:
• **Starting a business** - Registration, licenses, structure
• **Tax guidance** - VAT, income tax, PAN, TDS
• **Compliance** - What you need to stay legal

**What business are you thinking about?** Just tell me (like "I want to start a restaurant") and I'll give you the complete roadmap!"""
        
        if any(t in m for t in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask more about your business in Nepal. 🙏"
        
        return "I'm here to help with Nepal business questions. What would you like to know?"
    
    def _clean_response(self, text: str) -> str:
        """Clean up response text"""
        # Remove common artifacts
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'USER:.*?ASSISTANT:', '', text, flags=re.DOTALL)
        
        # Clean multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_industry = None
        logger.info("Memory cleared")


# Global instance
_advisor = None

def get_advisor() -> LightweightAdvisor:
    """Get or create advisor instance"""
    global _advisor
    if _advisor is None:
        _advisor = LightweightAdvisor()
    return _advisor

def get_response(message: str) -> tuple:
    """Main API function"""
    return get_advisor().chat(message)

def clear_memory():
    """Clear advisor memory"""
    if _advisor:
        _advisor.clear_memory()
