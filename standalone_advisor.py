"""
ChitraGupta Standalone Advisor
No Ollama dependency - loads GGUF model directly via llama-cpp-python
Optimized for RTX 4050 GPU
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nepal business knowledge base
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


class StandaloneAdvisor:
    """ChitraGupta using local GGUF model - no external dependencies"""
    
    def __init__(self):
        logger.info("Initializing ChitraGupta Standalone Advisor...")
        
        # Find model file
        model_path = self._find_model()
        
        logger.info(f"Loading model from: {model_path}")
        logger.info("Initializing GPU layers (RTX 4050)...")
        
        # Load model with GPU acceleration
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,      # Offload everything to GPU
            n_ctx=4096,           # Context window
            n_batch=512,          # Batch size for prompt processing
            verbose=False
        )
        
        # Load industry data
        self.industry_heuristics = self._load_industry_heuristics()
        
        # Session state
        self.conversation_history = []
        self.current_industry = None
        
        logger.info("ChitraGupta Standalone ready!")
    
    def _find_model(self) -> Path:
        """Locate the GGUF model file"""
        base = Path(__file__).parent
        
        # Check multiple locations
        paths = [
            base / "models" / "llama3.gguf",
            base / "llama3.gguf",
            Path("models/llama3.gguf"),
        ]
        
        for p in paths:
            if p.exists():
                return p
        
        raise FileNotFoundError(
            "Model not found! Place llama3.gguf in the 'models' folder.\n"
            "Get it from: C:\\Users\\<YOU>\\.ollama\\models\\blobs\\ (largest file ~2GB)"
        )
    
    def _load_industry_heuristics(self) -> Dict:
        """Load industry-specific data"""
        path = Path(__file__).parent / "data" / "industry_heuristics.json"
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load industry data: {e}")
        return {"industry_mappings": {}}
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent"""
        query_lower = query.lower()
        
        nepal_keywords = [
            'nepal', 'kathmandu', 'pvt ltd', 'vat', 'pan', 'ird', 'ocr',
            'rupee', 'nrs', 'npr', 'lakhs', 'crore', 'register company',
            'business nepal', 'tax nepal', 'license nepal'
        ]
        
        finance_keywords = [
            'tax', 'business', 'company', 'register', 'license', 'invest',
            'startup', 'profit', 'revenue', 'capital', 'loan', 'accounting'
        ]
        
        if any(kw in query_lower for kw in nepal_keywords):
            return "NEPAL_LAW"
        elif any(kw in query_lower for kw in finance_keywords):
            return "GENERAL_FINANCE"
        else:
            return "CHIT_CHAT"
    
    def _detect_industry(self, query: str) -> Optional[str]:
        """Detect industry from query"""
        query_lower = query.lower()
        
        industry_keywords = {
            "restaurant": ["restaurant", "food", "cafe", "hotel", "momo", "khaja"],
            "retail": ["shop", "store", "retail", "grocery", "supermarket"],
            "technology": ["tech", "software", "it", "app", "website", "digital"],
            "manufacturing": ["factory", "manufacturing", "production", "textile"],
            "education": ["school", "college", "tuition", "training", "coaching"],
            "healthcare": ["clinic", "hospital", "pharmacy", "medical", "health"],
            "construction": ["construction", "real estate", "building", "contractor"],
            "agriculture": ["farm", "agriculture", "poultry", "dairy", "organic"],
            "tourism": ["travel", "tour", "trek", "hotel", "tourism", "guide"],
            "sports": ["futsal", "gym", "fitness", "sports", "swimming"]
        }
        
        for industry, keywords in industry_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return industry
        
        return None
    
    def _build_context(self, query: str, intent: str) -> str:
        """Build knowledge context for the prompt"""
        context_parts = []
        query_lower = query.lower()
        
        # Add relevant Nepal knowledge
        if 'vat' in query_lower:
            context_parts.append(f"VAT Info: {json.dumps(NEPAL_KNOWLEDGE['vat'])}")
        if 'tax' in query_lower or 'income' in query_lower:
            context_parts.append(f"Income Tax: {json.dumps(NEPAL_KNOWLEDGE['income_tax'])}")
        if 'register' in query_lower or 'company' in query_lower:
            context_parts.append(f"Registration: {json.dumps(NEPAL_KNOWLEDGE['company_registration'])}")
        if 'license' in query_lower or 'pan' in query_lower:
            context_parts.append(f"Licenses: {json.dumps(NEPAL_KNOWLEDGE['licenses'])}")
        
        # Add industry-specific context
        if self.current_industry:
            industry_data = self.industry_heuristics.get("industry_mappings", {}).get(self.current_industry, {})
            if industry_data:
                context_parts.append(f"Industry ({self.current_industry}): {json.dumps(industry_data)}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _build_prompt(self, query: str, context: str, intent: str) -> str:
        """Build the Llama 3 format prompt"""
        
        # Build conversation history string
        history_str = ""
        if self.conversation_history:
            recent = self.conversation_history[-3:]  # Last 3 exchanges
            for h in recent:
                history_str += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
        
        system_prompt = """You are ChitraGupta, an expert AI business and tax consultant specializing in Nepal.

RULES:
- Give DIRECT, actionable answers - no interrogation
- Use Nepal-specific laws, rates, and procedures
- Mention specific amounts in NPR when relevant
- Be helpful even for casual conversation
- Keep responses concise but complete"""

        if context:
            system_prompt += f"\n\nRELEVANT KNOWLEDGE:\n{context}"
        
        if self.current_industry:
            system_prompt += f"\n\nUser is interested in: {self.current_industry} business"
        
        # Llama 3 chat format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>"""
        
        if history_str:
            prompt += f"""<|start_header_id|>user<|end_header_id|>

Previous conversation:
{history_str}<|eot_id|>"""
        
        prompt += f"""<|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def chat(self, query: str) -> Tuple[str, Dict]:
        """Process a chat message and return response"""
        start_time = time.time()
        
        # Classify and detect
        intent = self._classify_intent(query)
        detected_industry = self._detect_industry(query)
        
        if detected_industry:
            self.current_industry = detected_industry
        
        # Build context and prompt
        context = self._build_context(query, intent)
        prompt = self._build_prompt(query, context, intent)
        
        # Generate response
        output = self.llm(
            prompt,
            max_tokens=512,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            temperature=0.7,
            top_p=0.9,
        )
        
        response = output['choices'][0]['text'].strip()
        
        # Store in history
        self.conversation_history.append({
            "user": query,
            "assistant": response
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        elapsed = time.time() - start_time
        
        metadata = {
            "intent": intent,
            "industry": self.current_industry,
            "response_time": elapsed,
            "tokens": output.get('usage', {})
        }
        
        return response, metadata
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_industry = None
        logger.info("Memory cleared")
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return {
            "interactions": len(self.conversation_history),
            "current_industry": self.current_industry,
            "memory_active": len(self.conversation_history) > 0
        }


# Singleton instance
_advisor = None

def get_advisor() -> StandaloneAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = StandaloneAdvisor()
    return _advisor

def get_response(message: str) -> Tuple[str, Dict]:
    return get_advisor().chat(message)

def clear_memory():
    get_advisor().clear_memory()
