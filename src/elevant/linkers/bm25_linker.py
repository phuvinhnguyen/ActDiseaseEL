"""
Online BM25 entity linker using Wikidata API
No local database required - queries Wikidata API and uses BM25 for ranking
"""
from typing import Dict, Tuple, Optional, Any, List
import spacy
from spacy.tokens import Doc
import requests
from collections import defaultdict
import logging

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant import settings

logger = logging.getLogger("main." + __name__.split(".")[-1])


class BM25Linker(AbstractEntityLinker):
    """
    Online BM25 entity linker that:
    1. Detects entities using spaCy NER
    2. Queries Wikidata API for candidate entities
    3. Ranks candidates using BM25
    4. Caches results to minimize API calls
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Load SpaCy model for NER (no postprocessing to avoid database dependencies)
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "OnlineBM25")
        self.ner_identifier = "SpaCy"
        self.max_candidates = config.get("max_candidates", 10)
        self.use_cache = config.get("use_cache", True)
        self.api_timeout = config.get("api_timeout", 5)
        
        # BM25 parameters
        self.k1 = config.get("bm25_k1", 1.5)
        self.b = config.get("bm25_b", 0.75)
        
        # Cache for API results
        self.cache = {} if self.use_cache else None
        
        # NER tags to ignore
        self.ignore_tags = {"CARDINAL", "MONEY", "ORDINAL", "QUANTITY", "TIME", "DATE"}
        
        logger.info(f"Initialized OnlineBM25Linker with max_candidates={self.max_candidates}")
    
    def has_entity(self, entity_id: str) -> bool:
        """Always return True since we query online"""
        return True
    
    def search_wikidata(self, mention: str) -> List[Dict[str, Any]]:
        """Search Wikidata for entity candidates using the API"""
        # Check cache first
        if self.use_cache and mention in self.cache:
            return self.cache[mention]
        
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": mention,
            "language": "en",
            "limit": self.max_candidates,
            "format": "json"
        }
        
        # Add User-Agent header as required by Wikidata
        headers = {
            "User-Agent": "EntityLinkingResearch/1.0 (Educational/Research Project)"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.api_timeout)
            if response.ok:
                results = response.json().get("search", [])
                # Cache results
                if self.use_cache:
                    self.cache[mention] = results
                return results
            else:
                logger.warning(f"Wikidata API returned status {response.status_code} for '{mention}'")
        except Exception as e:
            logger.warning(f"Wikidata API error for '{mention}': {e}")
        
        return []
    
    def compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], 
                           avgdl: float, doc_len: int) -> float:
        """
        Compute BM25 score for a document given a query.
        
        BM25 formula:
        score = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        
        Where:
        - qi: query term
        - f(qi, D): frequency of qi in document D
        - |D|: document length
        - avgdl: average document length
        - k1, b: tuning parameters
        """
        score = 0.0
        
        # Count term frequencies in document
        doc_freq = defaultdict(int)
        for token in doc_tokens:
            doc_freq[token] += 1
        
        # Calculate BM25 for each query term
        for query_token in query_tokens:
            if query_token in doc_freq:
                # Term frequency
                tf = doc_freq[query_token]
                
                # IDF is simplified here (could be improved with corpus statistics)
                idf = 1.0
                
                # BM25 score for this term
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def rank_candidates_bm25(self, mention: str, candidates: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Rank candidates using BM25 scoring"""
        if not candidates:
            return []
        
        # Tokenize query (mention)
        query_tokens = mention.lower().split()
        
        # Prepare documents (label + description for each candidate)
        docs = []
        for cand in candidates:
            label = cand.get("label", "")
            description = cand.get("description", "")
            doc_text = f"{label} {description}".lower()
            docs.append(doc_text.split())
        
        # Calculate average document length
        avgdl = sum(len(doc) for doc in docs) / len(docs) if docs else 1.0
        
        # Score each candidate
        scored_candidates = []
        for i, cand in enumerate(candidates):
            doc_tokens = docs[i]
            doc_len = len(doc_tokens)
            score = self.compute_bm25_score(query_tokens, doc_tokens, avgdl, doc_len)
            scored_candidates.append((cand["id"], score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities in text using Wikidata API + BM25"""
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # Detect entity spans using spaCy NER
        for ent in doc.ents:
            if ent.label_ in self.ignore_tags:
                continue
            
            span = (ent.start_char, ent.end_char)
            mention = text[span[0]:span[1]]
            
            # Skip lowercase entities if uppercase flag is set
            if uppercase and mention.islower():
                continue
            
            # Skip if looks like a date
            if any(char.isdigit() for char in mention) and len(mention.split()) <= 3:
                continue
            
            # Query Wikidata for candidates
            candidates = self.search_wikidata(mention)
            
            if not candidates:
                # No candidates found - use NIL entity
                predictions[span] = EntityPrediction(span, "Q0", set())
                continue
            
            # Rank candidates with BM25
            ranked = self.rank_candidates_bm25(mention, candidates)
            
            # Select top candidate
            predicted_entity_id = ranked[0][0] if ranked else "Q0"
            candidate_ids = {cand_id for cand_id, _ in ranked}
            
            predictions[span] = EntityPrediction(span, predicted_entity_id, candidate_ids)
            
            logger.debug(f"Mention '{mention}' -> {predicted_entity_id} "
                        f"(from {len(candidates)} candidates)")
        
        return predictions


