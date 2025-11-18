"""
OneNet-style entity linker using transformers LLM
Simplified version with batch processing for faster inference
"""
import logging
import re
import random
import spacy
from typing import Dict, Tuple, Optional, Any, List

from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.utils.knowledge_base_mapper import UnknownEntity
from elevant.settings import LARGE_MODEL_NAME, NER_IGNORE_TAGS

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
        
        # Load spaCy model for NER
        try:
            self.model = spacy.load(LARGE_MODEL_NAME, disable=["lemmatizer"])
            logger.info(f"Loaded spaCy model: {LARGE_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.model = None
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "OneNet LLM")
        self.ner_identifier = "spaCy NER"
        
        # LLM client (only for entity linking, not NER)
        from elevant.llm_client import LLMClient
        model_path = config.get("llm_model_path", None)
        self.llm_client = LLMClient(model_path) if model_path else None
        
        # For Gemini API, check if model is available
        if self.llm_client and self.llm_client.use_gemini:
            if not self.llm_client.gemini_model:
                logger.warning("Gemini model not initialized. LLM features will be disabled.")
                self.llm_client = None
            else:
                logger.info(f"Gemini API initialized with model: {model_path}")
        
        # Ensure required entity databases are loaded
        try:
            self.entity_db.load_entity_names()
            self.entity_db.load_alias_to_entities()
            self.entity_db.load_hyperlink_to_most_popular_candidates()
            self.entity_db.load_sitelink_counts()
        except Exception as e:
            logger.warning(f"Error while loading entity databases: {e}")

        # OneNet-specific parameters
        self.top_k = config.get("top_k", 5)
        self.shuffle_candidates = config.get("shuffle_candidates", True)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _detect_entities_with_spacy(self, text: str, doc: Optional[Doc] = None) -> List[Dict]:
        """Use spaCy to detect entities"""
        if doc is None:
            if self.model is None:
                logger.warning("spaCy model not loaded, cannot detect entities")
                return []
            doc = self.model(text)
        
        entities = []
        for ent in doc.ents:
            # Skip ignored NER tags
            if ent.label_ in NER_IGNORE_TAGS:
                continue
            
            # Get aliases from entity database
            mention_text = ent.text
            aliases = [mention_text]  # Start with the mention itself
            entity_candidates = self.entity_db.get_candidates(mention_text)
            if entity_candidates:
                # Add aliases from the database for top candidates
                for entity_id in list(entity_candidates)[:3]:
                    entity_aliases = self.entity_db.get_entity_aliases(entity_id)
                    if entity_aliases:
                        aliases.extend(entity_aliases[:2])  # Add a couple aliases per candidate
            
            # Remove duplicates while preserving order
            seen = set()
            unique_aliases = []
            for alias in aliases:
                if alias.lower() not in seen:
                    seen.add(alias.lower())
                    unique_aliases.append(alias)
            aliases = unique_aliases[:5]  # Limit to 5 aliases
            
            entities.append({
                'text': mention_text,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'context_left': text[:ent.start_char],
                'context_right': text[ent.end_char:],
                'aliases': aliases,
                'link_entities': {},
                'confidence': 0.0,
                'candidates': []
            })
        
        return entities
    

    def _parse_linked_entity(self, output: str) -> tuple:
        """Parse entity ID and confidence from LLM output"""
        entity_id = None
        confidence = None
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if entity_id is None:
                if line.upper().startswith('ENTITY ID:'):
                    entity_id = line.split(':', 1)[1].strip()
                elif line.upper().startswith('ENTITY ID'):
                    parts = line.split(None, 2)
                    if len(parts) >= 3:
                        entity_id = parts[2].strip()
                elif 'ENTITY' in line.upper() and 'ID' in line.upper():
                    match = re.search(r'(?:ENTITY\s+ID[:\s]+|ID[:\s]+)([^\s]+)', line, re.IGNORECASE)
                    if match:
                        entity_id = match.group(1).strip()
            
            if confidence is None:
                if line.upper().startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.upper().startswith('CONFIDENCE'):
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        try:
                            confidence = float(parts[1].strip())
                        except ValueError:
                            pass
                elif 'CONFIDENCE' in line.upper():
                    match = re.search(r'(?:CONFIDENCE[:\s]+|CONF[:\s]+)([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        try:
                            confidence = float(match.group(1).strip())
                        except ValueError:
                            pass
            
            if entity_id is not None and confidence is not None:
                break
        
        if entity_id:
            entity_id = entity_id.strip()
            if entity_id.upper() in ['<NIL>', 'NIL', 'NONE', 'NULL', '']:
                entity_id = None
        
        if confidence is not None:
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.0
        else:
            confidence = 0.0
        
        return entity_id, confidence

    def _get_candidates_for_entities(self, entities: List[Dict]) -> List[Dict]:
        """Get candidates and link entities"""
        candidate_prompts = []
        
        for ent_idx, entity in enumerate(entities):
            entity_names = [entity['text']] + entity.get('aliases', [])
            candidates = set.union(*[self.entity_db.get_candidates(name) for name in entity_names])
            
            candidate_dicts = []
            if candidates:
                for entity_id in list(candidates)[:self.top_k]:
                    entity_name = self.entity_db.get_entity_name(entity_id)
                    description = self.entity_db.get_entity_description(entity_id)
                    if entity_name and entity_name != "Unknown":
                        candidate_dicts.append({
                            'id': entity_id,
                            'title': entity_name,
                            'description': description or f"Entity: {entity_name}",
                        })
            
            # Sort by score
            candidate_dicts.sort(key=lambda x: self.entity_db.get_sitelink_count(x['id']), reverse=True)
            entities[ent_idx]['candidates'] = candidate_dicts
            
            if candidate_dicts and self.llm_client:
                candidate_prompts.append(self._create_linking_prompt(entity, candidate_dicts))
            else:
                candidate_prompts.append(None)

        if any(p is not None for p in candidate_prompts) and self.llm_client:
            outputs = self.llm_client.call_batch([p for p in candidate_prompts if p is not None], max_tokens=512)
            linked_results = [self._parse_linked_entity(o) for o in outputs]
            
            output_idx = 0
            for i, prompt in enumerate(candidate_prompts):
                if prompt is not None:
                    entity_id, confidence = linked_results[output_idx]
                    output_idx += 1
                    if entity_id:
                        entities[i]['link_entities'] = {'id': entity_id}
                elif entities[i]['candidates']:
                    # Fallback to first candidate
                    entities[i]['link_entities'] = {'id': entities[i]['candidates'][0]['id']}

        return entities

    def _create_linking_prompt(self, entity: Dict, candidates: List[Dict]) -> List[Dict]:
        """Create OneNet-style prompt for Wikipedia entity linking"""
        context = f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}"
        context = ' '.join(context.split())
        
        # Shuffle candidates if enabled
        if self.shuffle_candidates:
            shuffled_candidates = random.sample(candidates, len(candidates))
        else:
            shuffled_candidates = candidates
        
        prompt = f"""KNOWLEDGE BASE: Wikipedia/Wikidata
TASK: Link Entity Mention to Wikipedia

=== ENTITY MENTION ===
Mention: {entity['text']}
Context: {context}

=== CANDIDATE ENTITIES ===
"""
        
        for i, candidate in enumerate(shuffled_candidates[:self.top_k]):
            candidate_id = candidate.get('id', 'N/A')
            candidate_title = candidate.get('title', 'Unknown')
            candidate_desc = candidate.get('description', '')[:100] if candidate.get('description') else ''
            prompt += f"{i+1}. {candidate_title} (ID: {candidate_id})"
            if candidate_desc:
                prompt += f"\n   Description: {candidate_desc}"
            prompt += "\n"
        
        prompt += f"""
=== YOUR TASK ===
Select the entity that best matches the mention in context.

CRITERIA:
1. Name match: Does the name match?
2. Context fit: Does it fit the context?
3. Entity type: Is it the right type?

=== REQUIRED OUTPUT FORMAT ===
You MUST output in this EXACT format (both fields required):
ENTITY ID: [candidate_id_from_list_above]
CONFIDENCE: [confidence_score_between_0.0_and_1.0]

Where:
- ENTITY ID: The exact ID from the candidate list (e.g., "Q12345")
- CONFIDENCE: A number between 0.0 and 1.0 indicating how confident you are:
  * 0.9-1.0: Very high confidence (exact match, clear context)
  * 0.7-0.9: High confidence (good match, some ambiguity)
  * 0.5-0.7: Medium confidence (partial match, some uncertainty)
  * 0.0-0.5: Low confidence (weak match, high uncertainty)

=== CRITICAL: FORMAT EXAMPLE ===
If candidate #2 is the best match and you're very confident:
ENTITY ID: Q12345
CONFIDENCE: 0.95

If candidate #1 is a good match but you're moderately confident:
ENTITY ID: Q67890
CONFIDENCE: 0.75

If no candidate matches well:
ENTITY ID: <NIL>
CONFIDENCE: 0.2

=== STRICT REQUIREMENTS ===
1. MUST output "ENTITY ID: " followed by the candidate ID or "<NIL>"
2. MUST output "CONFIDENCE: " followed by a number between 0.0 and 1.0
3. Both lines are REQUIRED
4. Use exact candidate ID from the list above
5. Confidence must be a valid float between 0.0 and 1.0
6. NO additional text, NO explanations, ONLY these two lines

=== OUTPUT NOW ===
"""
        
        return [{"role": "user", "content": prompt}]
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities using spaCy for NER and LLM for linking"""
        predictions = {}
        
        # Use spaCy for entity detection
        detected_entities = self._detect_entities_with_spacy(text, doc)
        
        if not detected_entities:
            return predictions
        
        # Use LLM for entity linking (batch processing)
        detected_entities = self._get_candidates_for_entities(detected_entities)
        
        for entity in detected_entities:
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = entity.get('link_entities', {}).get('id') or UnknownEntity.NIL.value
            candidates = {c['id'] for c in entity.get('candidates', [])}
            predictions[span] = EntityPrediction(span, entity_id, candidates)

        return predictions
