"""
Graph-based entity linker using transformers LLM
Simplified version with batch processing for faster inference
"""
import logging
import re
import spacy
from typing import Dict, Tuple, Optional, Any, List

from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.utils.knowledge_base_mapper import UnknownEntity
from elevant.settings import LARGE_MODEL_NAME, NER_IGNORE_TAGS

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
        
        # Load spaCy model for NER
        try:
            self.model = spacy.load(LARGE_MODEL_NAME, disable=["lemmatizer"])
            logger.info(f"Loaded spaCy model: {LARGE_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.model = None
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "Graph LLM")
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

        # Graph-specific parameters
        self.N_DESCRIPTIONS = config.get("n_descriptions", 3)
        self.K_SEARCH = config.get("k_search", 10)
        self.T_MAX = config.get("t_max", 5)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.7)
        
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
                        aliases.extend(list(entity_aliases)[:2])  # Add a couple aliases per candidate
            
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
        """Get candidates for all entities efficiently"""
        candidate_prompts = []
        confirmed_entities = [i for i in entities if i['confidence'] > self.HIGH_CONFIDENCE_THRESHOLD]
        unconfirmed_entities = [i for i in entities if i['confidence'] <= self.HIGH_CONFIDENCE_THRESHOLD]
        for ent_idx, entity in enumerate(unconfirmed_entities):
            entity_names = [entity['text']] + entity.get('aliases', [])
            candidates = set()
            
            # Get candidates for all entity names and aliases
            for entity_name in entity_names:
                candidates.update(self.entity_db.get_candidates(entity_name))
            
            # Also try to get candidates using hyperlink database (better coverage for partial names)
            mention_text = entity['text']
            hyperlink_candidates = self.entity_db.get_most_popular_candidate_for_hyperlink(mention_text)
            if hyperlink_candidates:
                candidates.update(hyperlink_candidates)
            
            # Try lowercase version too
            if mention_text != mention_text.lower():
                lowercase_candidates = self.entity_db.get_candidates(mention_text.lower())
                candidates.update(lowercase_candidates)
                lowercase_hyperlink = self.entity_db.get_most_popular_candidate_for_hyperlink(mention_text.lower())
                if lowercase_hyperlink:
                    candidates.update(lowercase_hyperlink)
            
            # Rank candidates by popularity (sitelink count)
            candidate_scores = []
            for entity_id in candidates:
                sitelink_count = self.entity_db.get_sitelink_count(entity_id)
                candidate_scores.append((entity_id, sitelink_count))
            
            # Sort by popularity (descending)
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            candidate_dicts = []
            for entity_id, _ in candidate_scores[:self.K_SEARCH]:
                entity_name = self.entity_db.get_entity_name(entity_id)
                description = self.entity_db.get_entity_description(entity_id)
                aliases = self.entity_db.get_entity_aliases(entity_id)
                if entity_name and entity_name != "Unknown":
                    candidate_dicts.append({
                        'id': entity_id,
                        'title': entity_name,
                        'description': description,
                        'aliases': list(aliases) if aliases else [],
                    })
            
            unconfirmed_entities[ent_idx]['candidates'] = candidate_dicts
            candidate_prompts.append(self._create_linking_prompt(entity, candidate_dicts, confirmed_entities or None))

        # Only call batch if we have prompts to process
        if candidate_prompts and self.llm_client:
            outputs = sum([self.llm_client.call_batch(candidate_prompts[i:i+8]) for i in range(0, len(candidate_prompts), 8)], [])
            linked_results = [self._parse_linked_entity(i) for i in outputs]
        else:
            linked_results = [(None, 0.0)] * len(unconfirmed_entities)

        for i in range(len(unconfirmed_entities)):
            confidence = linked_results[i][1]
            entity_id = linked_results[i][0]
            
            # If confidence is too low, skip linking but keep the entity for second pass
            if confidence and confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                # Don't link yet, will try again in second pass
                continue
            
            if entity_id:
                unconfirmed_entities[i]['confidence'] = confidence or 0.5
                entity_canonical_name = self.entity_db.get_entity_name(entity_id)
                entity_aliases = self.entity_db.get_entity_aliases(entity_id)
                entity_description = self.entity_db.get_entity_description(entity_id)
                unconfirmed_entities[i]['link_entities'] = {
                    'id': entity_id,
                    'title': entity_canonical_name,
                    'description': entity_description or "",
                    'aliases': list(entity_aliases) if entity_aliases else [],
                }
            elif unconfirmed_entities[i].get('candidates'):
                # If no entity selected but we have candidates, use the most popular one as fallback
                # This helps with cases where LLM doesn't output properly
                best_candidate = unconfirmed_entities[i]['candidates'][0]
                unconfirmed_entities[i]['confidence'] = 0.4  # Low confidence fallback
                unconfirmed_entities[i]['link_entities'] = {
                    'id': best_candidate['id'],
                    'title': best_candidate['title'],
                    'description': best_candidate.get('description', ''),
                    'aliases': best_candidate.get('aliases', []),
                }

        return confirmed_entities + unconfirmed_entities

    
    def _create_linking_prompt(self, entity: Dict, candidates: List[Dict], other_entities: List[Dict] = None) -> List[Dict]:
        """Create disambiguation prompt for a single entity with context"""
        context = f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}"
        context = ' '.join(context.split())
        
        prompt = f"""KNOWLEDGE BASE: Wikipedia/Wikidata
TASK: Link Entity Mention to Wikipedia

=== ENTITY MENTION ===
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
                prompt += f"\nOther entities in text:\n- " + "\n- ".join(other_entities_info[:5])
        
        prompt += "\n\n=== CANDIDATE ENTITIES ===\n"
        
        if not candidates:
            prompt += "No candidates found. Output <NIL>.\n"
        else:
            for i, candidate in enumerate(candidates[:self.T_MAX]):
                candidate_id = candidate.get('id', 'N/A')
                candidate_title = candidate.get('title', 'Unknown')
                candidate_desc = candidate.get('description', '')[:150] if candidate.get('description') else ''
                sitelink_count = self.entity_db.get_sitelink_count(candidate_id) if candidate_id != 'N/A' else 0
                
                prompt += f"{i+1}. {candidate_title} (ID: {candidate_id})"
                if sitelink_count > 0:
                    prompt += f" [Popularity: {sitelink_count:,} Wikipedia links]"
                if candidate_desc:
                    prompt += f"\n   Description: {candidate_desc}"
                # Show aliases if available
                candidate_aliases = candidate.get('aliases', [])
                if candidate_aliases and len(candidate_aliases) > 1:
                    alias_str = ', '.join(candidate_aliases[:3])
                    prompt += f"\n   Also known as: {alias_str}"
                prompt += "\n"
        
        prompt += f"""
=== YOUR TASK ===
Select the entity that best matches the mention in context.

IMPORTANT NOTES:
- The mention "{entity['text']}" might be a PARTIAL NAME (e.g., "Steve" could refer to "Steve Jobs", "Steve Ballmer", etc.)
- Consider the FULL CONTEXT to disambiguate
- Popular entities (with many Wikipedia links) are often more likely to be correct
- If the mention is a single word that appears in a longer entity name, that entity might be correct

CRITERIA (in order of importance):
1. Context fit: Does the entity make sense in the given context?
2. Name match: Does the name contain or match the mention?
3. Entity type: Is it the right type (person, organization, location, etc.)?
4. Popularity: More popular entities are often more likely

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

=== PARTIAL NAME EXAMPLES ===
Example 1: Mention "Steve" in context "Steve Jobs was CEO of Apple"
- If candidate list includes "Steve Jobs (Q19837)" with high popularity, select it
- Even if "Steve (Q63978595)" appears first, "Steve Jobs" is correct based on context

Example 2: Mention "Stanford" in context "dropped out of Stanford to join Microsoft"
- This refers to "Stanford University (Q41506)", not just "Stanford (Q2789084)" location
- Look for entities that match the context (educational institution in this case)

Example 3: Mention "Apple" in context "CEO of Apple"
- This likely refers to "Apple Inc. (Q312)" company, not other entities named "Apple"
- Consider entity type and context

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
        # Second pass for low-confidence entities
        detected_entities = self._get_candidates_for_entities(detected_entities)
        
        for entity in detected_entities:
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = entity.get('link_entities', {}).get('id') or UnknownEntity.NIL.value
            candidates = {c['id'] for c in entity.get('candidates', [])}
            predictions[span] = EntityPrediction(span, entity_id, candidates)

        return predictions