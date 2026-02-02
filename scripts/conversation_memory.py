"""
Enhanced Conversation Memory System for ChitraGupta
Provides persistent conversation history and user profile building
"""

import json
import time
import re
import math
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle
import os

class ConversationMemory:
    """Enhanced conversation memory with context persistence"""
    
    def __init__(self, user_id: str = "default_user", memory_file: str = "data/memory/conversation_memory.pkl"):
        self.user_id = user_id
        self.memory_file = memory_file
        self.conversation_history = []
        self.user_profile = {}
        self.extracted_facts = {}
        self.session_context = {}
        self.interaction_count = 0
        
        # Ensure memory directory exists
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        
        # Load existing memory if available
        self.load_memory()
        
    def add_interaction(self, user_message: str, ai_response: str, extracted_context: Dict):
        """Store each conversation turn with timestamp"""
        self.interaction_count += 1
        
        interaction = {
            'id': f"{self.user_id}_{int(time.time())}_{self.interaction_count}",
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'user_message': user_message,
            'ai_response': ai_response,
            'extracted_context': extracted_context,
            'turn_number': self.interaction_count,
            'session_id': self.get_current_session_id()
        }
        
        self.conversation_history.append(interaction)
        self._update_user_profile(extracted_context)
        self._extract_persistent_facts(user_message, ai_response)
        
        # Keep only last 50 interactions to prevent memory bloat
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
            
        # Auto-save after each interaction
        self.save_memory()
        
    def _update_user_profile(self, context: Dict):
        """Build persistent user profile from conversations"""
        business = context.get('business', {})
        personal = context.get('personal', {})
        
        # Update business profile
        for key, value in business.items():
            if value and value != "NA" and value != "Not specified":
                # Only update if new information is more specific
                current_value = self.user_profile.get(f'business_{key}', "")
                if len(str(value)) > len(str(current_value)):
                    self.user_profile[f'business_{key}'] = value
                    self.user_profile[f'business_{key}_last_updated'] = time.time()
                
        # Update personal profile  
        for key, value in personal.items():
            if value and value != "NA" and value != "Not specified":
                current_value = self.user_profile.get(f'personal_{key}', "")
                if len(str(value)) > len(str(current_value)):
                    self.user_profile[f'personal_{key}'] = value
                    self.user_profile[f'personal_{key}_last_updated'] = time.time()
                
    def _extract_persistent_facts(self, user_msg: str, ai_response: str):
        """Extract and store important facts mentioned"""
        user_lower = user_msg.lower()
        
        # Extract numerical facts with context
        fact_patterns = {
            'monthly_revenue': r'(?:monthly )?(?:revenue|income|earning).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|thousand|crore|rs|rupees?)?',
            'annual_revenue': r'(?:annual|yearly) (?:revenue|income|earning).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|thousand|crore|rs|rupees?)?',
            'investment_amount': r'(?:invest|investment|capital).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|thousand|crore|rs|rupees?)?',
            'employee_count': r'(?:employee|staff|worker).*?(\d+)\s*(?:people|persons?)?',
            'business_years': r'(?:business|company).*?(\d+)\s*(?:years?|months?)',
            'target_revenue': r'(?:target|goal|aim).*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|thousand|crore|rs|rupees?)?'
        }
        
        for fact_type, pattern in fact_patterns.items():
            match = re.search(pattern, user_lower)
            if match:
                self.extracted_facts[fact_type] = {
                    'value': match.group(1),
                    'timestamp': time.time(),
                    'context': user_msg[:100]  # Store context for reference
                }
        
        # Extract business type mentions
        business_types = {
            'clothing': ['clothing', 'garment', 'fashion', 'textile', 'apparel'],
            'electronics': ['electronics', 'gadgets', 'mobile', 'computer', 'tech'],
            'consultancy': ['consulting', 'consultancy', 'advisory', 'service'],
            'fintech': ['fintech', 'financial technology', 'payment', 'banking'],
            'restaurant': ['restaurant', 'hotel', 'food', 'catering', 'hospitality'],
            'retail': ['retail', 'shop', 'store', 'mart', 'supermarket'],
            'manufacturing': ['manufacturing', 'factory', 'production', 'industry'],
            'agriculture': ['agriculture', 'farming', 'crops', 'livestock'],
            'education': ['education', 'school', 'training', 'institute'],
            'healthcare': ['healthcare', 'medical', 'hospital', 'clinic']
        }
        
        for business_type, keywords in business_types.items():
            if any(keyword in user_lower for keyword in keywords):
                self.extracted_facts['business_type'] = {
                    'value': business_type,
                    'timestamp': time.time(),
                    'keywords_found': [k for k in keywords if k in user_lower]
                }
                break
                
        # Extract location mentions
        nepal_locations = ['kathmandu', 'pokhara', 'chitwan', 'butwal', 'dharan', 'birgunj', 'janakpur']
        for location in nepal_locations:
            if location in user_lower:
                self.extracted_facts['business_location'] = {
                    'value': location.title(),
                    'timestamp': time.time()
                }
                break
                
    def get_session_conversation_history(self) -> List[Dict]:
        """Get conversation history for current session only"""
        current_session = self.get_current_session_id()
        return [interaction for interaction in self.conversation_history 
                if interaction.get('session_id') == current_session]
    
    def get_conversation_context(self, limit: int = 10) -> str:
        """Generate conversation context string for current session only"""
        session_history = self.get_session_conversation_history()
        recent_interactions = session_history[-limit:] if session_history else []
        
        if not recent_interactions:
            return "No previous conversation in this session."
            
        context_parts = []
        for interaction in recent_interactions:
            timestamp = interaction.get('timestamp', time.time())
            formatted_time = self._format_timestamp(timestamp)
            
            context_parts.append(f"[{formatted_time}] User: {interaction['user_message'][:100]}...")
            context_parts.append(f"[{formatted_time}] Assistant: {interaction['ai_response'][:100]}...")
        
        return "\n".join(context_parts)
    
    def get_current_session_facts(self) -> Dict:
        """Get extracted facts for current session only"""
        session_history = self.get_session_conversation_history()
        session_facts = {}
        
        # Extract facts from current session only
        for interaction in session_history:
            if 'extracted_context' in interaction:
                context = interaction['extracted_context']
                if 'business_type' in context:
                    session_facts['business_type'] = context['business_type']
                if 'location' in context:
                    session_facts['location'] = context['location']
                if 'revenue_target' in context:
                    session_facts['revenue_target'] = context['revenue_target']
                    
        return session_facts
        """Get recent conversation context for LLaMA prompt"""
        if not self.conversation_history:
            return "No previous conversation history."
            
        recent_history = self.conversation_history[-last_n_turns:]
        context_parts = ["=== RECENT CONVERSATION HISTORY ==="]
        
        for i, interaction in enumerate(recent_history, 1):
            context_parts.append(f"\n--- Turn {interaction['turn_number']} ({self._format_time_ago(interaction['timestamp'])}) ---")
            context_parts.append(f"User: {interaction['user_message'][:150]}{'...' if len(interaction['user_message']) > 150 else ''}")
            context_parts.append(f"ChitraGupta: {interaction['ai_response'][:200]}{'...' if len(interaction['ai_response']) > 200 else ''}")
            
        return "\n".join(context_parts)
        
    def get_user_profile_summary(self) -> str:
        """Get comprehensive user profile for context"""
        if not self.user_profile and not self.extracted_facts:
            return "=== NEW USER ===\nNo previous context available. This is our first interaction."
            
        profile_parts = ["=== USER PROFILE SUMMARY ==="]
        
        # Business context
        business_info = {k: v for k, v in self.user_profile.items() 
                        if k.startswith('business_') and not k.endswith('_last_updated')}
        if business_info:
            profile_parts.append("\n🏢 BUSINESS PROFILE:")
            for key, value in business_info.items():
                clean_key = key.replace('business_', '').replace('_', ' ').title()
                last_updated = self.user_profile.get(f"{key}_last_updated", 0)
                time_info = self._format_time_ago(last_updated) if last_updated else "unknown"
                profile_parts.append(f"  • {clean_key}: {value} (updated {time_info})")
                
        # Personal context
        personal_info = {k: v for k, v in self.user_profile.items() 
                        if k.startswith('personal_') and not k.endswith('_last_updated')}
        if personal_info:
            profile_parts.append("\n👤 PERSONAL PROFILE:")
            for key, value in personal_info.items():
                clean_key = key.replace('personal_', '').replace('_', ' ').title()
                last_updated = self.user_profile.get(f"{key}_last_updated", 0)
                time_info = self._format_time_ago(last_updated) if last_updated else "unknown"
                profile_parts.append(f"  • {clean_key}: {value} (updated {time_info})")
                
        # Important facts
        if self.extracted_facts:
            profile_parts.append("\n📊 KEY EXTRACTED FACTS:")
            for key, value_info in self.extracted_facts.items():
                if isinstance(value_info, dict):
                    value = value_info.get('value', value_info)
                    timestamp = value_info.get('timestamp', 0)
                    time_info = self._format_time_ago(timestamp) if timestamp else "unknown"
                else:
                    value = value_info
                    time_info = "unknown"
                    
                clean_key = key.replace('_', ' ').title()
                profile_parts.append(f"  • {clean_key}: {value} (mentioned {time_info})")
                
        # Conversation statistics
        profile_parts.append(f"\n📈 INTERACTION STATS:")
        profile_parts.append(f"  • Total conversations: {len(self.conversation_history)}")
        profile_parts.append(f"  • Current session: {self.interaction_count} interactions")
        if self.conversation_history:
            first_interaction = self.conversation_history[0]['timestamp']
            profile_parts.append(f"  • Relationship duration: {self._format_time_ago(first_interaction)}")
                
        return "\n".join(profile_parts) if len(profile_parts) > 1 else "No previous context available."
        
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp to human-readable time ago"""
        if not timestamp:
            return "unknown time"
            
        diff = time.time() - timestamp
        
        if diff < 60:
            return "just now"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(diff / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp to readable date/time string"""
        if not timestamp:
            return "unknown time"
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%H:%M")
        except Exception:
            return "unknown time"
            
    def get_current_session_id(self) -> str:
        """Generate unique session ID for each browser session"""
        import uuid
        # Generate a unique session ID for each app restart/browser session
        if not hasattr(self, '_current_session_id'):
            self._current_session_id = f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        return self._current_session_id
        
    def generate_contextual_followup(self, current_response: str, conversation_history: List[Dict]) -> List[str]:
        """Generate smart, actionable follow-up questions based on current session"""
        
        followups = []
        
        # Use session-specific facts instead of global
        session_facts = self.get_current_session_facts()
        business_type = session_facts.get('business_type', {}).get('value', '')
        
        # Check what topics were discussed in this session only
        response_lower = current_response.lower()
        session_history = self.get_session_conversation_history()
        
        # Make follow-ups specific and actionable based on business context
        if business_type == 'clothing' or 'clothing' in response_lower:
            if 'vat' in response_lower:
                followups.append("Calculate exact VAT amounts for clothing items priced at NPR 500, 1000, and 2000")
                followups.append("Show me the monthly VAT filing process step-by-step")
            elif 'registration' in response_lower:
                followups.append("Create a detailed timeline: clothing business registration in Kathmandu")
                followups.append("List exact documents needed for textile business license application")
            else:
                followups.append("Compare profit margins: locally made vs imported clothing in Nepal")
                followups.append("Design seasonal inventory plan for Dashain/Tihar clothing sales")
                
        elif business_type == 'online' or 'online' in response_lower:
            if 'vat' in response_lower:
                followups.append("Calculate digital service tax for online sales of NPR 50,000/month")
                followups.append("Setup e-commerce VAT compliance checklist for Nepal")
            elif 'registration' in response_lower:
                followups.append("Guide me through e-commerce business registration process")
                followups.append("Compare costs: sole proprietorship vs private limited for online business")
            else:
                followups.append("Analyze digital payment gateway options and their fees in Nepal")
                followups.append("Create marketing budget breakdown for online business launch")
        
        # Financial planning follow-ups - make them specific
        if 'tax' in response_lower and 'lakh' in response_lower:
            followups.append("Break down exact tax calculation for NPR 50 lakh annual income")
            followups.append("Compare tax savings: individual vs company registration")
            
        if 'loan' in response_lower:
            followups.append("Compare EMI calculations: NPR 10 lakh vs 20 lakh business loan")
            followups.append("Find best bank loan rates for your specific business type")
            
        # Progressive conversation follow-ups
        session_conversation_count = len(session_history)
        if session_conversation_count == 1:
            followups.append("Set realistic monthly revenue target for your first 6 months")
            followups.append("Calculate startup costs breakdown for your business idea")
        elif session_conversation_count >= 3:
            followups.append("Create personalized business launch checklist based on our discussion")
            followups.append("Generate monthly financial planning calendar tailored to your business")
            
        # Remove duplicates and return top 3 most relevant
        unique_followups = list(dict.fromkeys(followups))
        return unique_followups[:3]
        if conversation_count == 1:
            followups.append("What's your target monthly revenue goal for the first year?")
            followups.append("Would you like me to create a business launch checklist?")
        elif conversation_count == 3:
            followups.append("Should we develop a 6-month financial roadmap for your business?")
            followups.append("Would you like to set up regular business health check-ins?")
        elif conversation_count >= 5:
            followups.append("Based on our discussions, would you like a comprehensive business plan summary?")
            
        # Remove duplicates and return top 3
        unique_followups = list(dict.fromkeys(followups))  # Preserves order
        return unique_followups[:3]
        
    def save_memory(self):
        """Save conversation memory to disk"""
        try:
            memory_data = {
                'user_id': self.user_id,
                'conversation_history': self.conversation_history,
                'user_profile': self.user_profile,
                'extracted_facts': self.extracted_facts,
                'session_context': self.session_context,
                'interaction_count': self.interaction_count,
                'last_saved': time.time()
            }
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
                
        except Exception as e:
            print(f"Warning: Could not save conversation memory: {e}")
            
    def load_memory(self):
        """Load conversation memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    
                self.conversation_history = memory_data.get('conversation_history', [])
                self.user_profile = memory_data.get('user_profile', {})
                self.extracted_facts = memory_data.get('extracted_facts', {})
                self.session_context = memory_data.get('session_context', {})
                self.interaction_count = memory_data.get('interaction_count', 0)
                
                print(f"Loaded conversation memory: {len(self.conversation_history)} previous interactions")
                
        except Exception as e:
            print(f"Warning: Could not load conversation memory: {e}")
            # Initialize with empty values if loading fails
            self.conversation_history = []
            self.user_profile = {}
            self.extracted_facts = {}
            self.session_context = {}
            self.interaction_count = 0
            
    def clear_memory(self):
        """Clear all conversation memory (use with caution)"""
        self.conversation_history = []
        self.user_profile = {}
        self.extracted_facts = {}
        self.session_context = {}
        self.interaction_count = 0
        
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
            
        print("Conversation memory cleared.")
        
    def export_conversation_history(self, format: str = 'json') -> str:
        """Export conversation history for analysis"""
        if format == 'json':
            return json.dumps(self.conversation_history, indent=2, default=str)
        elif format == 'txt':
            lines = []
            for interaction in self.conversation_history:
                lines.append(f"=== Turn {interaction['turn_number']} - {interaction['datetime']} ===")
                lines.append(f"User: {interaction['user_message']}")
                lines.append(f"ChitraGupta: {interaction['ai_response']}")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError("Format must be 'json' or 'txt'")
            
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Generate insights about the conversation patterns"""
        if not self.conversation_history:
            return {"message": "No conversation history available"}
            
        insights = {
            'total_interactions': len(self.conversation_history),
            'session_count': len(set(i['session_id'] for i in self.conversation_history)),
            'avg_user_message_length': sum(len(i['user_message']) for i in self.conversation_history) / len(self.conversation_history),
            'avg_response_length': sum(len(i['ai_response']) for i in self.conversation_history) / len(self.conversation_history),
            'most_common_topics': self._get_common_topics(),
            'business_evolution': self._track_business_evolution(),
            'engagement_pattern': self._analyze_engagement_pattern()
        }
        
        return insights
        
    def _get_common_topics(self) -> List[str]:
        """Identify most commonly discussed topics"""
        topic_keywords = {
            'tax': ['tax', 'vat', 'income tax'],
            'business_registration': ['registration', 'license', 'permit'],
            'investment': ['investment', 'funding', 'capital'],
            'compliance': ['compliance', 'filing', 'audit'],
            'planning': ['plan', 'strategy', 'roadmap']
        }
        
        topic_counts = {topic: 0 for topic in topic_keywords}
        
        for interaction in self.conversation_history:
            text = (interaction['user_message'] + ' ' + interaction['ai_response']).lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topic_counts[topic] += 1
                    
        # Sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics if count > 0]
        
    def _track_business_evolution(self) -> Dict[str, Any]:
        """Track how business understanding evolved over time"""
        evolution = {
            'initial_business_type': None,
            'current_business_type': None,
            'business_growth_indicators': [],
            'evolving_needs': []
        }
        
        # Find first mention of business type
        for interaction in self.conversation_history:
            if 'business_type' in str(interaction.get('extracted_context', {})):
                evolution['initial_business_type'] = interaction['extracted_context'].get('business', {}).get('type', 'Unknown')
                break
                
        # Current business type
        if 'business_type' in self.extracted_facts:
            evolution['current_business_type'] = self.extracted_facts['business_type'].get('value', 'Unknown')
            
        return evolution
        
    def _analyze_engagement_pattern(self) -> Dict[str, Any]:
        """Analyze user engagement patterns"""
        if len(self.conversation_history) < 2:
            return {"status": "Insufficient data"}
            
        # Calculate time between interactions
        time_gaps = []
        for i in range(1, len(self.conversation_history)):
            gap = self.conversation_history[i]['timestamp'] - self.conversation_history[i-1]['timestamp']
            time_gaps.append(gap)
            
        avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        
        return {
            'average_time_between_interactions': f"{avg_gap/3600:.1f} hours",
            'engagement_frequency': 'High' if avg_gap < 3600 else 'Medium' if avg_gap < 86400 else 'Low',
            'session_consistency': len(set(i['session_id'] for i in self.conversation_history[-10:])),
            'question_complexity_trend': self._measure_complexity_trend()
        }
        
    def _measure_complexity_trend(self) -> str:
        """Measure if questions are becoming more complex over time"""
        if len(self.conversation_history) < 3:
            return "Insufficient data"
            
        # Simple complexity measure based on message length and keywords
        complexities = []
        for interaction in self.conversation_history:
            msg = interaction['user_message']
            complexity = len(msg.split()) + len([w for w in msg.lower().split() if w in ['how', 'what', 'why', 'when', 'where', 'which']])
            complexities.append(complexity)
            
        if len(complexities) >= 3:
            early_avg = sum(complexities[:3]) / 3
            recent_avg = sum(complexities[-3:]) / 3
            
            if recent_avg > early_avg * 1.2:
                return "Increasing (more detailed questions)"
            elif recent_avg < early_avg * 0.8:
                return "Decreasing (more focused questions)"
            else:
                return "Stable"
        
        return "Stable"
