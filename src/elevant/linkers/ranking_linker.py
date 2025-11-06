"""
Ranking-based entity linker using transformers LLM
Based on the ranking_system.py approach from EntityLinking/e2e/systems
"""
import logging
import json
import time
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant import settings
from elevant.settings import NER_IGNORE_TAGS
from elevant.utils.dates import is_date
from elevant.utils.knowledge_base_mapper import KnowledgeBaseMapper, UnknownEntity
import elevant.ner.ner_postprocessing  # import is needed so Python finds the custom factory

logger = logging.getLogger("main." + __name__.split(".")[-1])


@dataclass
class DetectedEntity:
    """Represents a detected entity in text"""
    text: str
    start_pos: int
    end_pos: int
    context_left: str
    context_right: str
    entity_type: str = "UNKNOWN"
    descriptions: List[str] = None


class RankingLinker(AbstractEntityLinker):
    """
    Ranking-based entity linker following ranking_system.py approach
    Uses transformers LLM instead of API
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        self.model.add_pipe("ner_postprocessor", after="ner")
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "Ranking LLM")
        self.ner_identifier = self.linker_identifier
        
        # LLM client
        from elevant.llm_client import LLMClient
        model_path = config.get("llm_model_path", None)
        self.llm_client = LLMClient(model_path) if model_path else None
        
        # Ensure required entity databases are loaded
        try:
            self.entity_db.load_entity_names()
            self.entity_db.load_alias_to_entities()
            self.entity_db.load_hyperlink_to_most_popular_candidates()
            self.entity_db.load_sitelink_counts()
        except Exception as e:
            logger.warning(f"Error while loading entity databases: {e}")

        # Ranking-specific parameters
        self.top_k = config.get("top_k", 5)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _detect_entities_with_descriptions(self, text: str, doc: Doc) -> List[DetectedEntity]:
        """Detect entities and generate descriptions using LLM"""
        detected_entities = []
        
        # First, detect entities using spaCy
        entities = []
        for ent in doc.ents:
            if ent.label_ in NER_IGNORE_TAGS:
                continue
            span = (ent.start_char, ent.end_char)
            snippet = text[span[0]:span[1]]
            if is_date(snippet):
                continue
            
            start_pos = max(0, span[0] - 50)
            end_pos = min(len(text), span[1] + 50)
            context_left = text[start_pos:span[0]]
            context_right = text[span[1]:end_pos]
            
            entities.append({
                'text': snippet,
                'start_pos': span[0],
                'end_pos': span[1],
                'context_left': context_left,
                'context_right': context_right,
                'type': ent.label_
            })
        
        # If LLM client available, try to enhance detection and get descriptions
        if self.llm_client and self.llm_client.model and len(entities) > 0:
            try:
                # Use LLM to detect entities and generate descriptions
                entity_list = "\n".join([f"- {e['text']} (type: {e['type']})" for e in entities])
                prompt = f"""
Analyze the following text and identify all named entities (people, places, organizations, products, etc.).
For each entity, provide:
1. The entity text
2. Generate 3 different descriptions of what this entity could be

Text: "{text}"

Detected entities:
{entity_list}

For each detected entity, generate 3 different descriptions that help identify this entity.

Return the results as a JSON list of objects with keys: "text", "descriptions".
The descriptions should be 3 different ways to describe this entity for entity linking purposes.
"""
                
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call(messages, max_tokens=2048)
                
                # Parse response
                try:
                    entity_descriptions = json.loads(response)
                    if isinstance(entity_descriptions, list):
                        # Map descriptions to entities
                        desc_map = {}
                        for item in entity_descriptions:
                            if 'text' in item and 'descriptions' in item:
                                desc_map[item['text']] = item.get('descriptions', [])
                        
                        # Add descriptions to entities
                        for ent in entities:
                            descriptions = desc_map.get(ent['text'], [])
                            if not descriptions:
                                descriptions = [f"Entity: {ent['text']}"]
                            
                            detected_entity = DetectedEntity(
                                text=ent['text'],
                                start_pos=ent['start_pos'],
                                end_pos=ent['end_pos'],
                                context_left=ent['context_left'],
                                context_right=ent['context_right'],
                                entity_type=ent['type'],
                                descriptions=descriptions[:3]
                            )
                            detected_entities.append(detected_entity)
                        
                        return detected_entities
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON")
            except Exception as e:
                logger.warning(f"Error in LLM entity detection: {e}")
        
        # Fallback: use spaCy entities without descriptions
        for ent in entities:
            detected_entity = DetectedEntity(
                text=ent['text'],
                start_pos=ent['start_pos'],
                end_pos=ent['end_pos'],
                context_left=ent['context_left'],
                context_right=ent['context_right'],
                entity_type=ent['type'],
                descriptions=[f"Entity: {ent['text']}"]
            )
            detected_entities.append(detected_entity)
        
        return detected_entities
    
    def _link_single_entity_iterative(self, entity: DetectedEntity) -> Optional[str]:
        """Link a single entity using iterative selection"""
        # Search for candidates using entity text and descriptions
        all_candidates = []
        
        # Search with entity text
        candidates = self.entity_db.get_candidates(entity.text)
        for entity_id in candidates:
            entity_name = self.entity_db.get_entity_name(entity_id)
            if entity_name:
                all_candidates.append({
                    'id': entity_id,
                    'title': entity_name,
                    'description': f"Entity: {entity_name}",
                    'score': self.entity_db.get_sitelink_count(entity_id)
                })
        
        # Search with descriptions if available
        if entity.descriptions:
            for desc in entity.descriptions:
                if len(desc) < 2:
                    continue
                desc_candidates = self.entity_db.get_candidates(desc)
                for entity_id in desc_candidates:
                    entity_name = self.entity_db.get_entity_name(entity_id)
                    if entity_name:
                        # Check if already in all_candidates
                        if not any(c['id'] == entity_id for c in all_candidates):
                            all_candidates.append({
                                'id': entity_id,
                                'title': entity_name,
                                'description': f"Entity: {entity_name}",
                                'score': self.entity_db.get_sitelink_count(entity_id)
                            })
        
        if not all_candidates:
            return None
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_candidates = []
        for candidate in sorted(all_candidates, key=lambda x: x.get('score', 0), reverse=True):
            if candidate['id'] not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate['id'])
                if len(unique_candidates) >= self.top_k * 2:
                    break
        
        # If few candidates, select directly
        if len(unique_candidates) <= 3:
            best_candidate = self._select_best_entity(entity, unique_candidates)
            return best_candidate['id'] if best_candidate else None
        
        # Iterative selection: reduce candidates step by step
        current_candidates = unique_candidates
        while len(current_candidates) > 3:
            current_candidates = self._reduce_candidates(entity, current_candidates)
            time.sleep(0.1)  # Small delay
        
        # Final selection
        best_candidate = self._select_best_entity(entity, current_candidates)
        return best_candidate['id'] if best_candidate else None
    
    def _reduce_candidates(self, entity: DetectedEntity, candidates: List[Dict]) -> List[Dict]:
        """Reduce candidates by half using LLM selection"""
        if len(candidates) <= 3:
            return candidates
        
        if not self.llm_client or not self.llm_client.model:
            # Fallback: return top 3
            return candidates[:3]
        
        # Take top 6 candidates for selection
        top_candidates = candidates[:6]
        
        try:
            prompt = f"""
Given entity "{entity.text}" in context: "{entity.context_left} {entity.text} {entity.context_right}"

Select the 3 most relevant entities from:
{chr(10).join([f"{i+1}. {c['title']}: {c['description'][:100]}..." for i, c in enumerate(top_candidates)])}

Return numbers only (e.g., "1,3,5"):
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=100)
            
            selected = []
            for num in response.split(','):
                try:
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(top_candidates):
                        selected.append(top_candidates[idx])
                except:
                    continue
            
            return selected if selected else top_candidates[:3]
        except:
            return top_candidates[:3]
    
    def _select_best_entity(self, entity: DetectedEntity, candidates: List[Dict]) -> Optional[Dict]:
        """Select the best entity from candidates using LLM"""
        if len(candidates) == 1:
            return candidates[0]
        
        if not self.llm_client or not self.llm_client.model:
            # Fallback: return highest scoring
            return max(candidates, key=lambda x: x.get('score', 0))
        
        try:
            prompt = f"""
Given entity "{entity.text}" in context: "{entity.context_left} {entity.text} {entity.context_right}"

Select the best match from:
{chr(10).join([f"{i+1}. {c['title']}: {c['description'][:100]}..." for i, c in enumerate(candidates)])}

Return the number (1-{len(candidates)}):
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=50)
            
            try:
                idx = int(response.strip()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
            except:
                pass
        except:
            pass
        
        # Fallback to highest scoring
        return max(candidates, key=lambda x: x.get('score', 0))
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities using ranking-based approach"""
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # Detect entities with descriptions
        detected_entities = self._detect_entities_with_descriptions(text, doc)
        
        # Link each detected entity
        for entity in detected_entities:
            entity_id = self._link_single_entity_iterative(entity)
            
            if entity_id is None:
                entity_id = UnknownEntity.NIL.value
            
            span = (entity.start_pos, entity.end_pos)
            
            # Get candidates for the span
            candidates = self.entity_db.get_candidates(entity.text)
            predictions[span] = EntityPrediction(span, entity_id, candidates)
        
        return predictions


