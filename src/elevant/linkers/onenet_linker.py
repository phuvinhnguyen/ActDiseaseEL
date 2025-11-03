"""
OneNet-style entity linker using transformers LLM
Based on the onenet_system.py approach from EntityLinking/e2e/systems
"""
import logging
import json
import re
import random
from typing import Dict, Tuple, Optional, Any, List

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


class OneNetLinker(AbstractEntityLinker):
    """
    OneNet-style entity linker following onenet_system.py approach
    Uses transformers LLM instead of API
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        self.model.add_pipe("ner_postprocessor", after="ner")
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "OneNet LLM")
        self.ner_identifier = self.linker_identifier
        
        # LLM client
        from elevant.llm_client import LLMClient
        model_path = config.get("llm_model_path", None)
        self.llm_client = LLMClient(model_path) if model_path else None
        
        # OneNet-specific parameters
        self.top_k = config.get("top_k", 5)
        self.shuffle_candidates = config.get("shuffle_candidates", True)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _detect_entities_simple(self, text: str, doc: Doc) -> List[Dict]:
        """Simple entity detection using spaCy"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in NER_IGNORE_TAGS:
                continue
            span = (ent.start_char, ent.end_char)
            snippet = text[span[0]:span[1]]
            if is_date(snippet):
                continue
            
            context_left = text[max(0, span[0] - 50):span[0]]
            context_right = text[span[1]:min(len(text), span[1] + 50)]
            
            entities.append({
                'text': snippet,
                'start_pos': span[0],
                'end_pos': span[1],
                'context_left': context_left,
                'context_right': context_right
            })
        
        return entities
    
    def _create_onenet_prompt(self, entity: Dict, candidates: List[Dict]) -> str:
        """Create OneNet-style prompt"""
        # Format context similar to OneNet
        context = f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}"
        context = ' '.join(context.split())  # Clean whitespace
        
        # Shuffle candidates like OneNet does if enabled
        if self.shuffle_candidates:
            shuffled_candidates = random.sample(candidates, len(candidates))
        else:
            shuffled_candidates = candidates
        
        content = f"Mention: {entity['text']}\n"
        content += f"Context: {context}\n"
        
        # Add candidates
        for i, candidate in enumerate(shuffled_candidates):
            entity_info = f"{candidate['title']}. {candidate['description'][:200]}..."
            content += f"Entity {i+1}: {entity_info}\n"
        
        content += "\nAnswer:"
        
        return content
    
    def _parse_onenet_response(self, response: str, candidates: List[Dict]) -> Optional[Dict]:
        """Parse OneNet LLM response"""
        response_lower = response.strip().lower()
        
        # Try to find entity name in response
        for candidate in candidates:
            candidate_name = candidate['title'].lower()
            if candidate_name in response_lower:
                return candidate
        
        # Try to parse JSON-like response
        json_match = re.search(r'\{.*\}', response)
        if json_match:
            try:
                json_str = json.loads(json_match.group())
                for key, value in json_str.items():
                    if isinstance(value, str):
                        for candidate in candidates:
                            if candidate['title'].lower() == value.lower():
                                return candidate
            except:
                pass
        
        # Try to find entity by number (Entity 1, Entity 2, etc.)
        number_match = re.search(r'entity\s*(\d+)', response_lower)
        if number_match:
            try:
                entity_num = int(number_match.group(1))
                if 1 <= entity_num <= len(candidates):
                    return candidates[entity_num - 1]
            except:
                pass
        
        return None
    
    def _link_single_entity_onenet(self, entity: Dict) -> Optional[str]:
        """Link a single entity using OneNet approach"""
        # Get candidates from database
        candidates = self.entity_db.get_candidates(entity['text'])
        
        if not candidates:
            return None
        
        # Convert to dict format
        candidate_dicts = []
        for entity_id in list(candidates)[:self.top_k]:
            entity_name = self.entity_db.get_entity_name(entity_id)
            if entity_name:
                candidate_dicts.append({
                    'id': entity_id,
                    'title': entity_name,
                    'description': f"Entity: {entity_name}",
                    'score': self.entity_db.get_sitelink_count(entity_id)
                })
        
        if not candidate_dicts:
            return None
        
        # Sort by score
        candidate_dicts.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # If no LLM client, return best candidate
        if not self.llm_client or not self.llm_client.model:
            return candidate_dicts[0]['id']
        
        # Create OneNet-style prompt
        prompt = self._create_onenet_prompt(entity, candidate_dicts)
        
        try:
            # Get LLM response
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=512)
            
            # Parse response and select entity
            selected_entity = self._parse_onenet_response(response, candidate_dicts)
            
            if selected_entity:
                return selected_entity['id']
            else:
                # Fallback to first candidate
                return candidate_dicts[0]['id']
                
        except Exception as e:
            logger.warning(f"Error in OneNet linking: {e}")
            # Fallback to first candidate
            return candidate_dicts[0]['id']
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities using OneNet approach"""
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # Detect entities
        detected_entities = self._detect_entities_simple(text, doc)
        
        # Link each detected entity
        for entity in detected_entities:
            entity_id = self._link_single_entity_onenet(entity)
            
            if entity_id is None:
                entity_id = UnknownEntity.NIL.value
            
            span = (entity['start_pos'], entity['end_pos'])
            
            # Get candidates for the span
            candidates = self.entity_db.get_candidates(entity['text'])
            predictions[span] = EntityPrediction(span, entity_id, candidates)
        
        return predictions


