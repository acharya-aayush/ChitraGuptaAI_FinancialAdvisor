"""
Enhanced Document Retrieval System for ChitraGupta
Provides multi-stage retrieval with semantic search, keyword matching, and context awareness
"""

import numpy as np
import faiss
import json
import math
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results"""
    chunk_id: int
    chunk: Dict[str, Any]
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    context_score: float = 0.0
    final_score: float = 0.0
    retrieval_method: str = "unknown"
    source_type: str = "unknown"

class EnhancedDocumentRetrieval:
    """Advanced retrieval with multiple strategies and intelligent ranking"""
    
    def __init__(self):
        self.faiss_index = None
        self.embeddings = None
        self.sentence_model = None
        self.cross_encoder = None
        self.legal_chunks = []
        
        # Legal document categories for context-aware filtering
        self.document_categories = {
            'tax': ['income tax', 'corporate tax', 'vat', 'withholding', 'tax rate', 'taxation', 'levy'],
            'business_setup': ['registration', 'license', 'permit', 'incorporation', 'establishment', 'formation'],
            'financial': ['loan', 'investment', 'capital', 'funding', 'finance', 'bank', 'credit'],
            'legal_compliance': ['penalty', 'audit', 'requirement', 'obligation', 'compliance', 'filing'],
            'operations': ['employee', 'salary', 'insurance', 'social security', 'provident fund'],
            'trade': ['import', 'export', 'customs', 'duty', 'tariff', 'trade license']
        }
        
        # Business domain expansions for better keyword matching
        self.query_expansions = {
            'vat': ['vat', 'value added tax', 'turnover tax', 'sales tax', 'service tax'],
            'tax': ['tax', 'taxation', 'levy', 'duty', 'assessment', 'withholding'],
            'business': ['business', 'company', 'firm', 'enterprise', 'corporation', 'organization'],
            'registration': ['registration', 'incorporation', 'formation', 'establishment', 'setup'],
            'license': ['license', 'permit', 'authorization', 'approval', 'clearance', 'certificate'],
            'threshold': ['threshold', 'limit', 'minimum', 'maximum', 'ceiling', 'floor'],
            'capital': ['capital', 'investment', 'funding', 'money', 'finance', 'equity'],
            'employee': ['employee', 'worker', 'staff', 'personnel', 'workforce', 'labor'],
            'salary': ['salary', 'wage', 'pay', 'compensation', 'remuneration', 'income'],
            'penalty': ['penalty', 'fine', 'punishment', 'sanction', 'violation', 'breach'],
            'audit': ['audit', 'inspection', 'examination', 'review', 'assessment', 'verification'],
            'filing': ['filing', 'submission', 'return', 'declaration', 'report', 'statement']
        }
        
        # Load retrieval components
        self.load_retrieval_models()
        
    def load_retrieval_models(self):
        """Load FAISS index, embeddings, and sentence transformer"""
        try:
            # Load FAISS index
            faiss_path = 'data/processed/faiss_index.bin'
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
                logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            else:
                logger.warning(f"FAISS index not found at {faiss_path}")
                
            # Load embeddings
            embeddings_path = 'data/processed/embeddings.npy'
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            else:
                logger.warning(f"Embeddings not found at {embeddings_path}")
                
            # Load legal chunks
            chunks_path = 'data/processed/document_chunks.json'
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.legal_chunks = json.load(f)
                logger.info(f"Loaded legal chunks: {len(self.legal_chunks)} documents")
            else:
                logger.warning(f"Document chunks not found at {chunks_path}")
                
            # Load sentence transformer with offline mode
            try:
                import os
                os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
                logger.info("Loaded sentence transformer model")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
                
            # Load cross-encoder for re-ranking with offline mode
            try:
                if CROSS_ENCODER_AVAILABLE:
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
                    logger.info("Loaded cross-encoder for re-ranking")
                else:
                    logger.warning("CrossEncoder not available - install with: pip install sentence-transformers")
                    self.cross_encoder = None
            except Exception as e:
                logger.warning(f"Cross-encoder not available (will skip re-ranking): {e}")
                self.cross_encoder = None
                
        except Exception as e:
            logger.error(f"Error loading retrieval models: {e}")
            
    def retrieve_documents_advanced(self, query: str, conversation_context: str = "", 
                                   user_profile: str = "") -> List[RetrievalResult]:
        """Multi-stage retrieval pipeline with intelligent ranking"""
        
        logger.info(f"Starting advanced retrieval for query: {query[:100]}")
        
        # Stage 1: Query analysis and intent detection
        query_analysis = self.analyze_query_intent(query, conversation_context)
        
        # Stage 2: Get optimal retrieval parameters
        retrieval_params = self.get_optimal_retrieval_params(query_analysis, conversation_context)
        
        # Stage 3: Multi-method retrieval
        all_results = []
        
        # Semantic similarity retrieval (if FAISS available)
        if self.faiss_index is not None and self.sentence_model is not None:
            semantic_results = self._semantic_retrieval(query, retrieval_params['top_k'])
            all_results.extend(semantic_results)
            
        # Enhanced keyword retrieval
        keyword_results = self._enhanced_keyword_retrieval(query, retrieval_params['top_k'])
        all_results.extend(keyword_results)
        
        # Context-aware retrieval
        if conversation_context:
            context_results = self._context_aware_retrieval(query, conversation_context, retrieval_params['top_k'])
            all_results.extend(context_results)
            
        # Stage 4: Deduplication and intelligent ranking
        unique_results = self._deduplicate_results(all_results)
        
        # Stage 4.5: Cross-Encoder Re-ranking (NEW - CRITICAL FIX)
        if self.cross_encoder is not None and len(unique_results) > 0:
            unique_results = self._rerank_with_cross_encoder(query, unique_results)
        
        # Stage 5: Context-aware scoring boost
        boosted_results = self._apply_context_boost(unique_results, query_analysis, conversation_context, user_profile)
        
        # Stage 6: Diversity and final ranking
        final_results = self._diversify_and_rank(boosted_results, query, retrieval_params)
        
        # Return top results
        top_results = final_results[:retrieval_params['top_k']]
        
        logger.info(f"Retrieved {len(top_results)} documents with scores: {[r.final_score for r in top_results[:3]]}")
        return top_results
    
    def _semantic_retrieval(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """FAISS-based semantic similarity retrieval"""
        try:
            if self.faiss_index is None or self.sentence_model is None:
                return []
                
            # Generate query embedding
            query_vector = self.sentence_model.encode([query])
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_vector, min(top_k, self.faiss_index.ntotal))
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.legal_chunks) and similarity > 0.3:  # Relevance threshold
                    result = RetrievalResult(
                        chunk_id=int(idx),
                        chunk=self.legal_chunks[idx],
                        similarity_score=float(similarity),
                        retrieval_method='semantic_faiss',
                        source_type=self._determine_source_type(self.legal_chunks[idx])
                    )
                    results.append(result)
            
            logger.info(f"Semantic retrieval found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    def _enhanced_keyword_retrieval(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Enhanced keyword-based retrieval with TF-IDF scoring"""
        query_lower = query.lower()
        
        # Expand query with synonyms and related terms
        expanded_terms = self._expand_query_terms(query_lower)
        
        # Add domain-specific weight multipliers
        term_weights = self._calculate_term_weights(expanded_terms)
        
        scored_chunks = []
        
        for idx, chunk in enumerate(self.legal_chunks):
            chunk_text = chunk.get('text', '').lower()
            
            if not chunk_text:
                continue
                
            # Calculate enhanced TF-IDF score
            score = self._calculate_tfidf_score(expanded_terms, term_weights, chunk_text, idx)
            
            # Positional scoring (earlier mentions matter more)
            position_bonus = self._calculate_position_bonus(expanded_terms, chunk_text)
            score += position_bonus
            
            if score > 0:
                result = RetrievalResult(
                    chunk_id=idx,
                    chunk=chunk,
                    keyword_score=score,
                    retrieval_method='enhanced_keyword',
                    source_type=self._determine_source_type(chunk)
                )
                scored_chunks.append(result)
        
        # Sort by keyword score
        scored_chunks.sort(key=lambda x: x.keyword_score, reverse=True)
        
        logger.info(f"Keyword retrieval found {len(scored_chunks)} results")
        return scored_chunks[:top_k]
    
    def _context_aware_retrieval(self, query: str, conversation_context: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents based on conversation context"""
        if not conversation_context:
            return []
            
        # Extract key terms from conversation context
        context_terms = self._extract_context_terms(conversation_context)
        
        # Find documents that match both query and context
        context_results = []
        
        for idx, chunk in enumerate(self.legal_chunks):
            chunk_text = chunk.get('text', '').lower()
            
            # Score based on context term matches
            context_score = 0
            for term in context_terms:
                if term in chunk_text:
                    context_score += 0.1 * (1 + chunk_text.count(term) * 0.1)
            
            # Also check query relevance
            query_terms = query.lower().split()
            query_matches = sum(1 for term in query_terms if term in chunk_text)
            
            if context_score > 0 and query_matches > 0:
                result = RetrievalResult(
                    chunk_id=idx,
                    chunk=chunk,
                    context_score=context_score,
                    retrieval_method='context_aware',
                    source_type=self._determine_source_type(chunk)
                )
                context_results.append(result)
        
        # Sort by context score
        context_results.sort(key=lambda x: x.context_score, reverse=True)
        
        logger.info(f"Context-aware retrieval found {len(context_results)} results")
        return context_results[:top_k]
    
    def _expand_query_terms(self, query: str) -> List[str]:
        """Expand query with business synonyms and related terms"""
        query_words = query.split()
        expanded = set(query_words)
        
        # Add synonyms from predefined expansions
        for word in query_words:
            if word in self.query_expansions:
                expanded.update(self.query_expansions[word])
        
        # Add morphological variations
        for word in query_words:
            # Add plural/singular variations
            if word.endswith('s') and len(word) > 3:
                expanded.add(word[:-1])  # Remove 's'
            else:
                expanded.add(word + 's')  # Add 's'
            
            # Add common business term patterns
            if word.endswith('ing'):
                expanded.add(word[:-3])  # Remove 'ing'
                expanded.add(word[:-3] + 'ed')  # Add 'ed'
        
        return list(expanded)
    
    def _calculate_term_weights(self, terms: List[str]) -> Dict[str, float]:
        """Calculate importance weights for different terms"""
        weights = {}
        
        # High importance terms
        high_importance = ['tax', 'vat', 'registration', 'license', 'threshold', 'penalty', 'audit']
        medium_importance = ['business', 'company', 'investment', 'capital', 'employee']
        
        for term in terms:
            if term in high_importance:
                weights[term] = 2.0
            elif term in medium_importance:
                weights[term] = 1.5
            else:
                weights[term] = 1.0
                
        return weights
    
    def _calculate_tfidf_score(self, terms: List[str], term_weights: Dict[str, float], 
                              chunk_text: str, chunk_idx: int) -> float:
        """Calculate TF-IDF score for a chunk"""
        score = 0
        chunk_words = chunk_text.split()
        
        for term in terms:
            if term in chunk_text:
                # Term frequency
                tf = chunk_text.count(term) / len(chunk_words) if chunk_words else 0
                
                # Inverse document frequency (approximation)
                df = sum(1 for chunk in self.legal_chunks if term in chunk.get('text', '').lower())
                idf = math.log(len(self.legal_chunks) / (1 + df)) if df > 0 else 0
                
                # Term weight
                weight = term_weights.get(term, 1.0)
                
                # Combined score
                score += tf * idf * weight
        
        return score
    
    def _calculate_position_bonus(self, terms: List[str], chunk_text: str) -> float:
        """Give bonus for terms appearing early in the document"""
        position_bonus = 0
        
        for term in terms:
            position = chunk_text.find(term)
            if position != -1:
                # Earlier positions get higher bonus (max 0.1)
                normalized_position = position / len(chunk_text)
                position_bonus += 0.1 * (1 - normalized_position)
        
        return position_bonus
    
    def _extract_context_terms(self, conversation_context: str) -> List[str]:
        """Extract key terms from conversation context"""
        context_lower = conversation_context.lower()
        
        # Extract business and legal terms mentioned in context
        important_terms = []
        
        # Look for business types
        business_types = ['clothing', 'electronics', 'restaurant', 'consultancy', 'fintech']
        for btype in business_types:
            if btype in context_lower:
                important_terms.append(btype)
        
        # Look for financial amounts (simplified)
        import re
        amounts = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?\s*(?:lakh|thousand|crore)', context_lower)
        if amounts:
            important_terms.extend(['investment', 'capital', 'funding'])
        
        # Look for legal/tax terms
        legal_terms = ['tax', 'vat', 'registration', 'license', 'audit', 'penalty', 'compliance']
        for term in legal_terms:
            if term in context_lower:
                important_terms.append(term)
        
        return important_terms
    
    def _determine_source_type(self, chunk: Dict[str, Any]) -> str:
        """Determine the type of document source"""
        metadata = chunk.get('metadata', {})
        text = chunk.get('text', '').lower()
        
        # Check metadata first
        if 'source_type' in metadata:
            return metadata['source_type']
        
        # Infer from content
        if any(word in text for word in ['article', 'section', 'rule']):
            return 'legal_document'
        elif any(word in text for word in ['rate', 'percentage', 'threshold']):
            return 'tax_rates'
        elif any(word in text for word in ['procedure', 'step', 'process']):
            return 'procedure_guide'
        else:
            return 'general_provision'
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate chunks and merge scores"""
        chunk_map = {}
        
        for result in results:
            chunk_id = result.chunk_id
            
            if chunk_id in chunk_map:
                # Merge scores from different retrieval methods
                existing = chunk_map[chunk_id]
                existing.similarity_score = max(existing.similarity_score, result.similarity_score)
                existing.keyword_score = max(existing.keyword_score, result.keyword_score)
                existing.context_score = max(existing.context_score, result.context_score)
                
                # Update retrieval method to include both
                if result.retrieval_method not in existing.retrieval_method:
                    existing.retrieval_method += f"+{result.retrieval_method}"
            else:
                chunk_map[chunk_id] = result
        
        return list(chunk_map.values())
    
    def _apply_context_boost(self, results: List[RetrievalResult], query_analysis: Dict[str, Any],
                           conversation_context: str, user_profile: str) -> List[RetrievalResult]:
        """Apply context-based score boosts"""
        
        for result in results:
            chunk_text = result.chunk.get('text', '').lower()
            boost = 0
            
            # Intent-based boost
            for intent in query_analysis.get('intents', []):
                if intent == 'specific_info' and any(word in chunk_text for word in ['article', 'section']):
                    boost += 0.2
                elif intent == 'process' and any(word in chunk_text for word in ['procedure', 'step']):
                    boost += 0.2
                elif intent == 'requirements' and any(word in chunk_text for word in ['must', 'shall', 'required']):
                    boost += 0.2
            
            # Domain-based boost
            for domain in query_analysis.get('domains', []):
                domain_keywords = self.document_categories.get(domain, [])
                matches = sum(1 for keyword in domain_keywords if keyword in chunk_text)
                boost += matches * 0.1
            
            # User profile boost (if business type matches)
            if user_profile and 'business_type' in user_profile.lower():
                business_type = self._extract_business_type_from_profile(user_profile)
                if business_type and business_type in chunk_text:
                    boost += 0.15
            
            # Conversation context boost
            if conversation_context:
                context_terms = self._extract_context_terms(conversation_context)
                context_matches = sum(1 for term in context_terms if term in chunk_text)
                boost += context_matches * 0.05
            
            result.context_score += boost
        
        return results
    
    def _extract_business_type_from_profile(self, user_profile: str) -> Optional[str]:
        """Extract business type from user profile"""
        profile_lower = user_profile.lower()
        business_types = ['clothing', 'electronics', 'restaurant', 'consultancy', 'fintech', 'retail']
        
        for btype in business_types:
            if btype in profile_lower:
                return btype
        return None
    
    def _diversify_and_rank(self, results: List[RetrievalResult], query: str, 
                           retrieval_params: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply final ranking with diversity considerations"""
        
        # Calculate final scores
        for result in results:
            final_score = 0
            
            # Combine different score types with weights
            if result.similarity_score > 0:
                final_score += result.similarity_score * 0.4
            if result.keyword_score > 0:
                final_score += result.keyword_score * 0.3
            if result.context_score > 0:
                final_score += result.context_score * 0.3
            
            # Source type bonus
            if result.source_type == 'legal_document':
                final_score += 0.1
            elif result.source_type == 'tax_rates':
                final_score += 0.05
            
            # Length penalty for very short chunks
            chunk_text = result.chunk.get('text', '')
            if len(chunk_text.split()) < 20:
                final_score *= 0.8
            
            result.final_score = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply diversity filter to avoid too many results from same source
        diverse_results = self._ensure_source_diversity(results, retrieval_params)
        
        return diverse_results
    
    def _ensure_source_diversity(self, results: List[RetrievalResult], 
                                retrieval_params: Dict[str, Any]) -> List[RetrievalResult]:
        """Ensure results come from different document sections"""
        diverse_results = []
        source_count = {}
        max_per_source = 3  # Maximum results per source type
        
        for result in results:
            source = result.source_type
            
            if source_count.get(source, 0) < max_per_source:
                diverse_results.append(result)
                source_count[source] = source_count.get(source, 0) + 1
        
        return diverse_results
    
    def _rerank_with_cross_encoder(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank results using cross-encoder for better relevance"""
        try:
            if not results or self.cross_encoder is None:
                return results
            
            # Prepare query-document pairs
            pairs = []
            for result in results:
                doc_text = result.chunk.get('text', '')[:1000]  # Limit to 1000 chars for cross-encoder
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Update final scores with cross-encoder scores
            for i, result in enumerate(results):
                # Combine existing score with cross-encoder score (weighted)
                result.final_score = (result.final_score * 0.3) + (float(ce_scores[i]) * 0.7)
            
            # Re-sort by updated scores
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(f"Cross-encoder re-ranked {len(results)} results. Top score: {results[0].final_score:.3f}")
            return results
            
        except Exception as e:
            logger.warning(f"Cross-encoder re-ranking failed: {e}. Using original scores.")
            return results
    
    def analyze_query_intent(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Analyze query to detect user intent and adjust retrieval strategy"""
        
        query_lower = query.lower()
        
        # Intent detection patterns
        intent_patterns = {
            'specific_info': ['what is', 'how much', 'when is', 'where is', 'which', 'define'],
            'comparison': ['vs', 'versus', 'difference', 'compare', 'better', 'between'],
            'process': ['how to', 'steps', 'procedure', 'process', 'method', 'way to'],
            'requirements': ['need', 'required', 'must have', 'should', 'necessary'],
            'planning': ['plan', 'strategy', 'roadmap', 'timeline', 'schedule'],
            'problem_solving': ['problem', 'issue', 'error', 'wrong', 'help', 'stuck']
        }
        
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_intents.append(intent)
        
        # Domain detection
        domain_keywords = {
            'tax': ['tax', 'vat', 'income tax', 'withholding', 'taxation'],
            'business_setup': ['registration', 'license', 'startup', 'company', 'incorporation'],
            'financial': ['loan', 'investment', 'capital', 'funding', 'finance', 'bank'],
            'legal_compliance': ['audit', 'filing', 'compliance', 'requirement', 'penalty'],
            'operations': ['employee', 'salary', 'insurance', 'social security'],
            'trade': ['import', 'export', 'customs', 'duty', 'trade']
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        # Query complexity analysis
        complexity = 'simple'
        if len(query.split()) > 15 or len(detected_intents) > 2:
            complexity = 'complex'
        elif len(query.split()) > 8 or len(detected_intents) > 1:
            complexity = 'medium'
        
        # Specificity analysis
        specificity = 'general'
        if any(char.isdigit() for char in query) or any(word in query_lower for word in ['article', 'section', 'rule']):
            specificity = 'specific'
        
        return {
            'intents': detected_intents,
            'domains': detected_domains,
            'complexity': complexity,
            'specificity': specificity,
            'query_length': len(query.split()),
            'contains_numbers': any(char.isdigit() for char in query)
        }
    
    def get_optimal_retrieval_params(self, query_analysis: Dict[str, Any], 
                                   conversation_context: str = "") -> Dict[str, Any]:
        """Determine optimal retrieval parameters based on query analysis"""
        
        params = {
            'top_k': 7,  # Default number of results
            'similarity_threshold': 0.6,
            'diversity_weight': 0.3,
            'context_weight': 0.2
        }
        
        # Adjust based on query complexity
        if query_analysis['complexity'] == 'complex':
            params['top_k'] = 10  # More documents for complex queries
            params['similarity_threshold'] = 0.5  # Lower threshold for broader search
        elif query_analysis['complexity'] == 'simple':
            params['top_k'] = 5  # Fewer documents for simple queries
            params['similarity_threshold'] = 0.7  # Higher threshold for precision
        
        # Adjust based on conversation length
        conversation_length = len(conversation_context.split()) if conversation_context else 0
        if conversation_length > 100:
            params['context_weight'] = 0.4  # Higher context weight for ongoing conversations
            params['top_k'] = min(params['top_k'] + 2, 12)  # Slightly more results
        
        # Adjust based on specificity
        if query_analysis['specificity'] == 'specific':
            params['similarity_threshold'] = 0.7  # Higher precision for specific queries
            params['diversity_weight'] = 0.1  # Less diversity, more precision
        
        # Adjust based on detected domains
        if len(query_analysis['domains']) > 2:
            params['top_k'] = min(params['top_k'] + 3, 12)  # More results for multi-domain queries
        
        return params
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        stats = {
            'total_documents': len(self.legal_chunks),
            'faiss_available': self.faiss_index is not None,
            'sentence_model_available': self.sentence_model is not None,
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'document_categories': list(self.document_categories.keys()),
            'query_expansion_terms': len(self.query_expansions)
        }
        
        if self.faiss_index:
            stats['faiss_index_size'] = self.faiss_index.ntotal
            
        return stats
