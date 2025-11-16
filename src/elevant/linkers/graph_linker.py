"""
Graph-based entity linker using transformers LLM
Simplified version with batch processing for faster inference
"""
import logging
import re
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


class GraphLinker(AbstractEntityLinker):
    """
    Graph-based entity linker following graph_system.py approach
    Uses transformers LLM instead of API
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        self.model = None
        # self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        # self.model.add_pipe("ner_postprocessor", after="ner")
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "Graph LLM")
        self.ner_identifier = self.linker_identifier
        
        # LLM client
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
        
        # Ensure required entity databases are loaded for candidate search and scoring
        # - Names for titles and alias aggregation
        # - Aliases and name-to-entity mappings for candidate generation
        # - Hyperlink candidates for popular mention->entity mappings (Wikipedia only)
        # - Sitelink counts for popularity-based scoring
        try:
            # Only load entity names if not already loaded (e.g., by custom KB)
            if not self.entity_db.entity_name_db:
                self.entity_db.load_entity_names()
            # Ensure name_to_entities_db is loaded for get_candidates()
            if not self.entity_db.name_to_entities_db:
                self.entity_db.load_name_to_entities()
            self.entity_db.load_alias_to_entities()
            
            # Try to load hyperlink candidates (Wikipedia-specific, may not exist for custom KB)
            try:
                self.entity_db.load_hyperlink_to_most_popular_candidates()
            except Exception as hyperlink_error:
                # This is expected for custom knowledge bases
                logger.debug(f"Hyperlink mappings not available (expected for custom KB): {hyperlink_error}")
            
            self.entity_db.load_sitelink_counts()
        except Exception as e:
            logger.warning(f"Error while loading entity databases: {e}")

        # Graph-specific parameters
        self.N_DESCRIPTIONS = config.get("n_descriptions", 3)
        self.K_SEARCH = config.get("k_search", 10)
        self.T_MAX = config.get("t_max", 5)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.7)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _detect_entities_with_llm(self, text: str) -> List[Dict]:
        """Use LLM to detect medical entities"""
        prompt = f"""KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Extract Medical Disease Entities

=== ABOUT DOID ===
DOID contains diseases, syndromes, infections, genetic disorders, cancers, and medical conditions.
DOID does NOT contain: people, places, organizations, dates, numbers, anatomical parts alone.

=== TEXT ===
{text}

=== YOUR TASK ===
Extract ALL disease/medical condition mentions that can be linked to DOID.

=== REQUIRED OUTPUT FORMAT ===
You MUST output each entity in this EXACT format (all fields required):
ENTITY: mention text | short surrounding text | alias 1, alias 2, alias 3

Where:
- mention text: The exact text as it appears in the document
- short surrounding text: A exact match short surrounding text that contains the mention text
- alias1,alias2,alias3: Comma-separated list of alternative names/synonyms (at least include the mention text itself)

=== CRITICAL: FORMAT EXAMPLE ===
If the text contains: "Patient diagnosed with agranulocytosis and leucopenia."
Then output:
ENTITY: agranulocytosis | In many cases, agranulocytosis is caused by chemotherapy. | agranulocytosis,agranulocytic angina

If "leucopenia" starts at character 50 and ends at character 60:
Then output:
ENTITY: Leucopenia | Leucopenia is a condition that occurs when the number of white blood cells in the body is too low. | leucopenia,leukopenia

=== STRICT REQUIREMENTS ===
1. EVERY line must start with "ENTITY: "
2. ALL fields are REQUIRED (mention | short surrounding text | aliases)
3. Use EXACT text from document (case-sensitive and detail specific, like copy from the text) for mention text and short surrounding text, this is the only way to find the exact position of the mention text in the text
4. Aliases must include at least the mention text itself, following DOID entity name format for exact match)
5. One entity per line, no blank lines between entities
6. If no entities found, output nothing

=== OUTPUT NOW ===
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.call(messages, max_tokens=512)
        
        entities = []
        for line in response.split('\n'):
            line = line.strip()
            if not line or not line.startswith('ENTITY:'): continue
            
            try:
                # Remove "ENTITY: " prefix and split by |
                parts = line.replace('ENTITY:', '').strip().split('|')
                if len(parts) < 3: continue
                
                mention_text = parts[0].strip()
                surrounding_text = parts[1].strip()
                aliases = [i.strip() for i in parts[2].strip().split(',') if i.strip()]

                # find position of surrounding text in text
                if len(text.split(mention_text)) > 2:
                    start_pos_surrounding = text.find(surrounding_text)
                    start_pos = text.find(mention_text, start_pos_surrounding - 1)
                else:
                    start_pos = text.find(mention_text)

                entities.append({
                    'text': mention_text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(mention_text),
                    'context_left': text[:start_pos],
                    'context_right': text[start_pos + len(mention_text):],
                    'aliases': aliases,
                    'link_entities': {},
                    'confidence': 0.0,
                    'candidates': []
                })
            except Exception as e:
                logger.warning(f"Error parsing entity line '{line}': {e}")
                continue
        
        return entities
    

    def _parse_linked_entity(self, output: str) -> tuple:
        """Parse entity ID and confidence from LLM output
        
        Returns:
            tuple: (entity_id, confidence) where entity_id is str or None, confidence is float 0.0-1.0
        """
        entity_id = None
        confidence = None
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try to find ENTITY ID (case-insensitive, handle variations)
            if entity_id is None:
                # Try different formats
                if line.upper().startswith('ENTITY ID:'):
                    entity_id = line.split(':', 1)[1].strip()
                elif line.upper().startswith('ENTITY ID'):
                    # Handle "ENTITY ID" without colon
                    parts = line.split(None, 2)
                    if len(parts) >= 3:
                        entity_id = parts[2].strip()
                elif 'ENTITY' in line.upper() and 'ID' in line.upper():
                    # Try to extract ID from various formats
                    match = re.search(r'(?:ENTITY\s+ID[:\s]+|ID[:\s]+)([^\s]+)', line, re.IGNORECASE)
                    if match:
                        entity_id = match.group(1).strip()
            
            # Try to find CONFIDENCE (case-insensitive, handle variations)
            if confidence is None:
                # Try different formats
                if line.upper().startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.upper().startswith('CONFIDENCE'):
                    # Handle "CONFIDENCE" without colon
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        try:
                            confidence = float(parts[1].strip())
                        except ValueError:
                            pass
                elif 'CONFIDENCE' in line.upper():
                    # Try to extract confidence from various formats
                    match = re.search(r'(?:CONFIDENCE[:\s]+|CONF[:\s]+)([0-9.]+)', line, re.IGNORECASE)
                    if match:
                        try:
                            confidence = float(match.group(1).strip())
                        except ValueError:
                            pass
            
            # Stop if we found both
            if entity_id is not None and confidence is not None:
                break
        
        # Validate and normalize
        if entity_id:
            entity_id = entity_id.strip()
            # Handle <NIL> or None cases
            if entity_id.upper() in ['<NIL>', 'NIL', 'NONE', 'NULL', '']:
                entity_id = None
        
        # Validate confidence
        if confidence is not None:
            try:
                confidence = float(confidence)
                # Clamp to 0.0-1.0 range
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.0
        else:
            confidence = 0.0
        
        return entity_id, confidence


    def _get_candidates_for_entities(self, entities: List[Dict]) -> List[Dict]:
        """Get candidates for all entities efficiently"""
        candidate_prompts = []
        confirmed_entities = [i for i in entities if i['confidence'] > self.HIGH_CONFIDENCE_THRESHOLD]
        unconfirmed_entities = [i for i in entities if i['confidence'] <= self.HIGH_CONFIDENCE_THRESHOLD]
        for ent_idx, entity in enumerate(unconfirmed_entities):
            entity_names = [entity['text']] + entity['aliases']
            candidates = set.union(*[self.entity_db.get_candidates(entity_name) for entity_name in entity_names])
            
            candidate_dicts = []
            if candidates:
                for entity_id in list(candidates)[:self.K_SEARCH]:
                    entity_name = self.entity_db.get_entity_name(entity_id)
                    description = self.entity_db.get_entity_description(entity_id)
                    aliases = self.entity_db.get_entity_aliases(entity_id)
                    if entity_name and entity_name != "Unknown":
                        candidate_dicts.append({
                            'id': entity_id,
                            'title': entity_name,
                            'description': description,
                            'aliases': aliases,
                        })
            
            unconfirmed_entities[ent_idx]['candidates'] = candidate_dicts
            candidate_prompts.append(self._create_linking_prompt(entity, candidate_dicts, confirmed_entities or None))

        # Only call batch if we have prompts to process
        if candidate_prompts and self.llm_client:
            outputs = self.llm_client.call_batch(candidate_prompts)
            linked_results = [self._parse_linked_entity(i) for i in outputs]
        else:
            linked_results = [(None, 0.0)] * len(unconfirmed_entities)

        for i in range(len(unconfirmed_entities)):
            confidence = linked_results[i][1]
            if confidence and confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                continue
            entity_id = linked_results[i][0]
            if entity_id:
                unconfirmed_entities[i]['confidence'] = confidence
                entity_canonical_name = self.entity_db.get_entity_name(entity_id)
                entity_aliases = self.entity_db.get_entity_aliases(entity_id)
                entity_description = self.entity_db.get_entity_description(entity_id)
                unconfirmed_entities[i]['link_entities'] = {
                    'id': entity_id,
                    'title': entity_canonical_name,
                    'description': entity_description,
                    'aliases': entity_aliases,
                }

        return confirmed_entities + unconfirmed_entities

    
    def _create_linking_prompt(self, entity: Dict, candidates: List[Dict], other_entities: List[Dict] = None) -> List[Dict]:
        """Create disambiguation prompt for a single entity with context"""
        context = f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}"
        context = ' '.join(context.split())
        
        prompt = f"""KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Link Medical Mention to DOID Disease

=== MEDICAL MENTION ===
Mention: {entity['text']}
Context: {context}"""
        
        # Add information about other entities detected (for context)
        if other_entities:
            other_entities_info = []
            for e in other_entities:
                if isinstance(e, dict) and e.get('text') != entity['text']:
                    link_info = e.get('link_entities', {})
                    if link_info.get('id'):
                        other_entities_info.append(
                            f"{e['text']} is linked to {link_info.get('title', 'Unknown')} ({link_info['id']}): {link_info.get('description', '')[:100]}"
                        )
            if other_entities_info:
                prompt += f"\nOther medical entities in text:\n- " + "\n- ".join(other_entities_info[:5])
        
        prompt += "\n\n=== CANDIDATE DISEASES ===\n"
        
        for i, candidate in enumerate(candidates[:self.T_MAX]):
            candidate_id = candidate.get('id', 'N/A')
            candidate_title = candidate.get('title', 'Unknown')
            candidate_desc = candidate.get('description', '')[:100] if candidate.get('description') else ''
            prompt += f"{i+1}. {candidate_title} (ID: {candidate_id})"
            if candidate_desc:
                prompt += f"\n   Description: {candidate_desc}"
            prompt += "\n"
        
        prompt += f"""
=== YOUR TASK ===
Select the disease that best matches the mention in context.

CRITERIA:
1. Name match: Does the name match the medical term?
2. Specificity: Is it the right level of detail?
3. Context fit: Does it fit the clinical context?

=== REQUIRED OUTPUT FORMAT ===
You MUST output in this EXACT format (both fields required):
ENTITY ID: [candidate_id_from_list_above]
CONFIDENCE: [confidence_score_between_0.0_and_1.0]

Where:
- ENTITY ID: The exact ID from the candidate list (e.g., "DOID:12345")
- CONFIDENCE: A number between 0.0 and 1.0 indicating how confident you are:
  * 0.9-1.0: Very high confidence (exact match, clear context)
  * 0.7-0.9: High confidence (good match, some ambiguity)
  * 0.5-0.7: Medium confidence (partial match, some uncertainty)
  * 0.0-0.5: Low confidence (weak match, high uncertainty)

=== CRITICAL: FORMAT EXAMPLE ===
If candidate #2 is the best match and you're very confident:
ENTITY ID: DOID:12345
CONFIDENCE: 0.95

If candidate #1 is a good match but you're moderately confident:
ENTITY ID: DOID:67890
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
        """Predict entities using simplified graph-based approach with batch processing"""        
        predictions = {}
        
        detected_entities = self._detect_entities_with_llm(text)
        
        if not detected_entities:
            return predictions
        
        detected_entities = self._get_candidates_for_entities(detected_entities)
        detected_entities = self._get_candidates_for_entities(detected_entities)
        
        for entity in detected_entities:
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = entity.get('link_entities', {}).get('id') or UnknownEntity.NIL.value
            candidates = {c['id'] for c in entity.get('candidates', [])}
            predictions[span] = EntityPrediction(span, entity_id, candidates)

        return predictions


