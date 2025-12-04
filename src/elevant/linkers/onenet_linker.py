"""
OneNet-style entity linker using transformers LLM
Simplified version with batch processing for faster inference.

This implementation keeps the original OneNet idea:
- Use an LLM to detect mentions in text
- Generate candidates for each mention
- Optionally use the LLM again to pick the best candidate

But instead of getting candidates from `EntityDatabase`, it now follows
the graph-based linker approach and derives candidates directly from
the DOID ontology via `find_near_matches_for_span`.
"""
import logging
import re
import random
from typing import Dict, Tuple, Optional, Any, List

from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.utils.knowledge_base_mapper import UnknownEntity

# Re‑use ontology utilities from the graph-based linker
from elevant.linkers.graph_linker import parse_obo_file, find_near_matches_for_span

logger = logging.getLogger("main." + __name__.split(".")[-1])


class OneNetLinker(AbstractEntityLinker):
    """
    OneNet-style entity linker following onenet_system.py approach
    Uses transformers LLM instead of API.

    Entity candidates are discovered using the DOID ontology and
    `find_near_matches_for_span` (same mechanism as `GraphLinker`),
    not via `EntityDatabase.get_candidates`.
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any],
                 obo_path: str = '/media/volume/LLMRag2/.local/HumanDiseaseOntology/src/ontology/doid-merged.obo'):
        self.entity_db = entity_database
        self.model = None
        self.obo_path = obo_path
        
        # Get config variables
        self.linker_identifier = config.get("linker_name", "OneNet LLM")
        self.ner_identifier = self.linker_identifier
        
        # LLM client
        from elevant.llm_client import LLMClient
        model_path = config.get("llm_model_path", None)
        self.llm_client = LLMClient(model_path) if model_path else None
        
        # For Gemini API, check if model is available
        if self.llm_client and getattr(self.llm_client, "use_gemini", False):
            if not getattr(self.llm_client, "gemini_model", None):
                logger.warning("Gemini model not initialized. LLM features will be disabled.")
                self.llm_client = None
            else:
                logger.info(f"Gemini API initialized with model: {model_path}")
        
        # Load ontology matcher from OBO file (same as GraphLinker)
        try:
            self.entities_matcher = parse_obo_file(self.obo_path)
        except Exception as e:
            logger.warning(f"Error while parsing OBO ontology at {self.obo_path}: {e}")
            self.entities_matcher = None
        
        # OneNet-specific parameters
        self.top_k = config.get("top_k", 5)
        self.shuffle_candidates = config.get("shuffle_candidates", True)
        
    def has_entity(self, entity_id: str) -> bool:
        # Keep compatibility with existing `EntityDatabase` checks
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
- short surrounding text: An exact match short surrounding text that contains the mention text
- alias1,alias2,alias3: Comma-separated list of alternative names/synonyms (at least include the mention text itself)

=== CRITICAL: FORMAT EXAMPLE ===
If the text contains: "Patient diagnosed with agranulocytosis and leucopenia."
Then output:
ENTITY: agranulocytosis | In many cases, agranulocytosis is caused by chemotherapy. | agranulocytosis,agranulocytic angina

If "leucopenia" is mentioned:
Then output:
ENTITY: leucopenia | Leucopenia is a condition that occurs when the number of white blood cells in the body is too low. | leucopenia,leukopenia

=== STRICT REQUIREMENTS ===
1. EVERY line must start with "ENTITY: "
2. ALL fields are REQUIRED (mention | short surrounding text | aliases)
3. Use EXACT text from document (case-sensitive and detail specific, like copy from the text) for mention text and short surrounding text, this is the only way to find the exact position of the mention text in the text
4. Aliases must include at least the mention text itself, following DOID entity name format for exact match
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
                parts = line.replace('ENTITY:', '').strip().split('|')
                if len(parts) < 3: continue
                
                mention_text = parts[0].strip()
                surrounding_text = parts[1].strip()
                aliases = [i.strip() for i in parts[2].strip().split(',') if i.strip()]

                # Find position of surrounding text in text
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
        """
        Get candidates and link entities.

        This version mirrors the GraphLinker behaviour for candidate discovery:
        - Use `find_near_matches_for_span` over the DOID ontology
        - Do NOT use `self.entity_db.get_candidates`
        - Keep the OneNet pipeline structure (LLM can still re-rank/pick)
        """
        candidate_prompts = []
        
        if not self.entities_matcher:
            logger.warning("Ontology matcher not initialized; no candidates will be generated.")
        
        # Unpack ontology matcher (see `graph_linker.parse_obo_file`)
        if self.entities_matcher:
            _, term_to_entities, sym_spell = self.entities_matcher
        else:
            term_to_entities, sym_spell = {}, None
        
        for ent_idx, entity in enumerate(entities):
            entity_names = [entity['text']] + entity.get('aliases', [])
            candidate_dicts: List[Dict[str, Any]] = []
            
            # Use ontology-based fuzzy matching to get DOID candidates
            if term_to_entities and sym_spell:
                seen_ids = set()
                for name in entity_names:
                    name = name.strip()
                    if not name:
                        continue
                    try:
                        near_matches = find_near_matches_for_span(
                            name,
                            term_to_entities,
                            sym_spell,
                            top_k=self.top_k,
                            min_confidence=0.7,
                            min_similarity=0.7,
                        )
                    except Exception as e:
                        logger.debug(f"Error in find_near_matches_for_span for '{name}': {e}")
                        continue
                    
                    for cand in near_matches:
                        cand_id = cand.get('id')
                        if not cand_id or cand_id in seen_ids:
                            continue
                        seen_ids.add(cand_id)
                        title = cand.get('name', '')
                        description = cand.get('def', '')
                        synonyms = cand.get('synonyms', [])
                        if synonyms:
                            description = (description + "\nSynonyms: " + ", ".join(synonyms)).strip()
                        if title:
                            candidate_dicts.append({
                                'id': cand_id,
                                'title': title,
                                'description': description or f"Entity: {title}",
                            })
            
            entities[ent_idx]['candidates'] = candidate_dicts
            
            if candidate_dicts and self.llm_client:
                candidate_prompts.append(self._create_linking_prompt(entity, candidate_dicts))
            else:
                candidate_prompts.append(None)

        if any(p is not None for p in candidate_prompts) and self.llm_client:
            outputs = self.llm_client.call_batch(
                [p for p in candidate_prompts if p is not None],
                max_tokens=512
            )
            linked_results = [self._parse_linked_entity(o) for o in outputs]
            
            output_idx = 0
            for i, prompt in enumerate(candidate_prompts):
                if prompt is not None:
                    entity_id, confidence = linked_results[output_idx]
                    output_idx += 1
                    if entity_id:
                        entities[i]['link_entities'] = {'id': entity_id}
                elif entities[i]['candidates']:
                    # Fallback to first candidate if no LLM decision
                    entities[i]['link_entities'] = {'id': entities[i]['candidates'][0]['id']}

        return entities

    def _create_linking_prompt(self, entity: Dict, candidates: List[Dict]) -> List[Dict]:
        """Create OneNet-style prompt for DOID entity linking"""
        context = f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}"
        context = ' '.join(context.split())
        
        # Shuffle candidates if enabled
        if self.shuffle_candidates:
            shuffled_candidates = random.sample(candidates, len(candidates))
        else:
            shuffled_candidates = candidates
        
        prompt = f"""KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Link Medical Mention to DOID Disease

=== MEDICAL MENTION ===
Mention: {entity['text']}
Context: {context}

=== CANDIDATE DISEASES ===
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
        """Predict entities using OneNet approach with batch processing"""
        predictions = {}
        
        detected_entities = self._detect_entities_with_llm(text)
        
        if not detected_entities:
            return predictions
        
        detected_entities = self._get_candidates_for_entities(detected_entities)
        
        for entity in detected_entities:
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = entity.get('link_entities', {}).get('id') or UnknownEntity.NIL.value
            candidates = {c['id'] for c in entity.get('candidates', [])}
            predictions[span] = EntityPrediction(span, entity_id, candidates)

        return predictions
