"""
Improved BM25 entity linker using local entity database
Combines BM25 ranking with entity popularity from Wikipedia hyperlinks
"""
from typing import Dict, Tuple, Optional, Any, List, Set
import spacy
from spacy.tokens import Doc
from collections import defaultdict
import logging

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant import settings
from elevant.settings import NER_IGNORE_TAGS
from elevant.utils.dates import is_date
from elevant.utils.knowledge_base_mapper import UnknownEntity

logger = logging.getLogger("main." + __name__.split(".")[-1])


class BM25DatabaseLinker(AbstractEntityLinker):
    """
    BM25 entity linker with local database that:
    1. Detects entities using spaCy NER (enhanced with postprocessing)
    2. Gets candidate entities from local database (aliases, names, hyperlinks)
    3. Ranks candidates using BM25 + entity popularity
    4. Falls back to popularity-only when BM25 is inconclusive
    """
    
    def __init__(self, entity_database: EntityDatabase, config: Dict[str, Any]):
        self.entity_db = entity_database
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "BM25-DB")
        self.ner_identifier = "EnhancedSpacy"
        
        # BM25 parameters
        self.k1 = config.get("bm25_k1", 1.5)
        self.b = config.get("bm25_b", 0.75)
        
        # Ranking strategy
        self.use_popularity = config.get("use_popularity", True)
        self.popularity_weight = config.get("popularity_weight", 0.3)  # Mix BM25 with popularity
        
        # Load SpaCy model with enhanced NER (if available)
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        
        # Try to add NER postprocessor if available (optional for small databases)
        try:
            import elevant.ner.ner_postprocessing  # noqa: F401
            self.model.add_pipe("ner_postprocessor", after="ner")
            logger.info("Using enhanced NER with postprocessing")
        except (ImportError, ValueError) as e:
            logger.warning(f"NER postprocessor not available, using basic SpaCy NER: {e}")
            self.ner_identifier = "SpaCy"
        
        logger.info(f"Initialized BM25-DB Linker with popularity_weight={self.popularity_weight}")
    
    def has_entity(self, entity_id: str) -> bool:
        """Check if entity exists in database"""
        return self.entity_db.contains_entity(entity_id)
    
    def get_candidates(self, mention: str) -> Set[str]:
        """
        Get candidate entities from database using multiple strategies:
        1. Exact name match
        2. Alias match (Wikidata aliases)
        3. Hyperlink candidates (Wikipedia hyperlink statistics)
        """
        candidates = set()
        
        # Strategy 1: Get from name-to-entity mapping (Wikidata labels)
        name_candidates = self.entity_db.get_candidates(mention)
        if name_candidates:
            candidates.update(name_candidates)
        
        # Strategy 2: Get from hyperlink statistics (most popular candidates)
        hyperlink_candidates = self.entity_db.get_most_popular_candidate_for_hyperlink(mention)
        if hyperlink_candidates:
            candidates.update(hyperlink_candidates)
        
        # Strategy 3: Case-insensitive variants
        if not candidates:
            name_candidates_lower = self.entity_db.get_candidates(mention.lower())
            if name_candidates_lower:
                candidates.update(name_candidates_lower)
        
        return candidates
    
    def get_entity_description(self, entity_id: str) -> str:
        """
        Build a description for an entity from database information.
        Includes: entity name, aliases, and other metadata.
        """
        parts = []
        
        # Get entity name
        entity_name = self.entity_db.get_entity_name(entity_id)
        if entity_name:
            parts.append(entity_name)
        
        # Get aliases
        aliases = self.entity_db.get_aliases_for_entity(entity_id)
        if aliases:
            # Limit to top 5 aliases to avoid very long descriptions
            parts.extend(list(aliases)[:5])
        
        return " ".join(parts) if parts else ""
    
    def get_entity_popularity(self, entity_id: str) -> int:
        """Get entity popularity from sitelink count (Wikipedia language editions)"""
        return self.entity_db.get_sitelink_count(entity_id)
    
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
                # Using log(N/df) approximation, assuming reasonable corpus size
                idf = 1.0
                
                # BM25 score for this term
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                score += idf * (numerator / denominator)
        
        return score
    
    def rank_candidates(self, mention: str, candidates: Set[str]) -> List[Tuple[str, float]]:
        """
        Rank candidates using BM25 score combined with entity popularity.
        
        Final score = (1 - w) * BM25_normalized + w * Popularity_normalized
        where w is the popularity_weight parameter.
        """
        if not candidates:
            return []
        
        candidates_list = list(candidates)
        
        # Tokenize query (mention)
        query_tokens = mention.lower().split()
        
        # Prepare documents for each candidate
        docs = []
        for entity_id in candidates_list:
            description = self.get_entity_description(entity_id)
            doc_tokens = description.lower().split()
            docs.append(doc_tokens)
        
        # Calculate average document length for BM25
        avgdl = sum(len(doc) for doc in docs) / len(docs) if docs else 1.0
        
        # Calculate BM25 scores
        bm25_scores = []
        for i, entity_id in enumerate(candidates_list):
            doc_tokens = docs[i]
            doc_len = len(doc_tokens) if doc_tokens else 1
            bm25_score = self.compute_bm25_score(query_tokens, doc_tokens, avgdl, doc_len)
            bm25_scores.append(bm25_score)
        
        # Get popularity scores if enabled
        popularity_scores = []
        if self.use_popularity:
            for entity_id in candidates_list:
                popularity = self.get_entity_popularity(entity_id)
                popularity_scores.append(popularity)
        
        # Normalize and combine scores
        scored_candidates = []
        
        # Normalize BM25 scores (0-1 range)
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
        bm25_normalized = [score / max_bm25 for score in bm25_scores]
        
        if self.use_popularity and popularity_scores:
            # Normalize popularity scores (0-1 range)
            max_pop = max(popularity_scores) if max(popularity_scores) > 0 else 1.0
            pop_normalized = [score / max_pop for score in popularity_scores]
            
            # Combine scores
            for i, entity_id in enumerate(candidates_list):
                combined_score = (
                    (1 - self.popularity_weight) * bm25_normalized[i] +
                    self.popularity_weight * pop_normalized[i]
                )
                scored_candidates.append((entity_id, combined_score))
        else:
            # Use BM25 only
            for i, entity_id in enumerate(candidates_list):
                scored_candidates.append((entity_id, bm25_normalized[i]))
        
        # Sort by combined score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities in text using database + BM25 + popularity"""
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # Detect entity spans using enhanced spaCy NER
        for ent in doc.ents:
            if ent.label_ in NER_IGNORE_TAGS:
                continue
            
            span = (ent.start_char, ent.end_char)
            mention = text[span[0]:span[1]]
            
            # Skip lowercase entities if uppercase flag is set
            if uppercase and mention.islower():
                continue
            
            # Skip dates
            if is_date(mention):
                continue
            
            # Get candidates from database
            candidates = self.get_candidates(mention)
            
            if not candidates:
                # No candidates found - use NIL entity
                predicted_entity_id = UnknownEntity.NIL.value
                predictions[span] = EntityPrediction(span, predicted_entity_id, set())
                continue
            
            # Rank candidates with BM25 + popularity
            ranked = self.rank_candidates(mention, candidates)
            
            # Select top candidate
            predicted_entity_id = ranked[0][0] if ranked else UnknownEntity.NIL.value
            candidate_ids = {cand_id for cand_id, _ in ranked}
            
            predictions[span] = EntityPrediction(span, predicted_entity_id, candidate_ids)
            
            logger.debug(f"Mention '{mention}' -> {predicted_entity_id} "
                        f"(from {len(candidates)} candidates, score={ranked[0][1]:.3f})")
        
        return predictions

