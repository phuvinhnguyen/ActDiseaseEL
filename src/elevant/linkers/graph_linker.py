"""
Graph-based entity linker using transformers LLM
Based on the graph_system.py approach from EntityLinking/e2e/systems
"""
import logging
import re
import time
from typing import Dict, Tuple, Optional, Any, List, Set
from dataclasses import dataclass, field
from collections import defaultdict

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
class GraphNode:
    """Represents a node in the entity graph"""
    entity_text: str
    start_pos: int
    end_pos: int
    context_left: str
    context_right: str
    descriptions: List[str] = field(default_factory=list)
    entity_id: Optional[str] = None
    entity_title: Optional[str] = None
    confidence: float = 0.0
    status: str = "pending"  # pending, high_confidence, done
    candidates: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class GraphLinker(AbstractEntityLinker):
    """
    Graph-based entity linker following graph_system.py approach
    Uses transformers LLM instead of API
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        self.model.add_pipe("ner_postprocessor", after="ner")
        
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
        self.K_SEARCH = config.get("k_search", 5)
        self.T_MAX = config.get("t_max", 5)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.9)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _build_entity_graph(self, text: str, doc: Doc):
        """Build entity graph from text using LLM and spaCy"""
        graph_nodes = {}
        
        # Non-healthcare entity labels to filter out for DOID
        NON_HEALTHCARE_LABELS = {
            "GPE",      # Geopolitical entity (countries, cities, states)
            "LOC",      # Location (non-geopolitical locations)
            "PERSON",   # People
            "ORG",      # Organizations (unless healthcare-related, but hard to filter)
            "DATE",     # Dates
            "TIME",     # Times
            "CARDINAL", # Numbers
            "ORDINAL",  # Ordinal numbers
            "MONEY",    # Money
            "QUANTITY", # Quantities
            "PERCENT",  # Percentages
            "EVENT",    # Events
            "FAC",      # Facilities (buildings, airports, etc.)
        }
        
        # First, detect entities using spaCy
        entities = []
        for ent in doc.ents:
            if ent.label_ in NER_IGNORE_TAGS:
                continue
            # Filter out non-healthcare entities for DOID
            if ent.label_ in NON_HEALTHCARE_LABELS:
                continue
            span = (ent.start_char, ent.end_char)
            snippet = text[span[0]:span[1]]
            if is_date(snippet):
                continue
            
            # Additional filtering: skip if it looks like a location or person name
            snippet_lower = snippet.lower()
            if any(word in snippet_lower for word in ['street', 'road', 'avenue', 'park', 'garden', 'hospital', 'clinic']):
                # Skip if it's clearly a location (unless it's a disease name with these words)
                if not any(disease_word in snippet_lower for disease_word in ['disease', 'syndrome', 'cancer', 'tumor']):
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
        
        # If LLM client available, try to enhance detection and get relations
        if self.llm_client and (self.llm_client.model or self.llm_client.gemini_model):
            try:
                # Use LLM to detect additional entities and relations
                prompt = f"""
KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Medical Entity Detection for Disease Linking

=== ABOUT DOID ===
The Human Disease Ontology (DOID) is a medical knowledge base containing:
• Diseases: diabetes, cancer, influenza, tuberculosis, malaria
• Medical conditions: hypertension, asthma, arthritis, obesity
• Syndromes: Down syndrome, metabolic syndrome, SARS
• Infectious diseases: COVID-19, HIV, hepatitis, pneumonia
• Genetic disorders: cystic fibrosis, hemophilia, sickle cell disease
• Mental health: depression, schizophrenia, anxiety disorders
• Allergies: peanut allergy, drug allergies, food allergies
• Cancers: breast cancer, lung cancer, leukemia, lymphoma

DOID does NOT contain:
✗ People (doctors, patients, researchers)
✗ Places (hospitals, clinics, cities, countries)
✗ Organizations (WHO, medical institutions)
✗ Dates, times, numbers, measurements
✗ Anatomical parts alone (heart, liver, lung) - only with disease context
✗ Isolated symptoms (fever, pain, cough) - only as defined conditions
✗ Medical procedures or drugs alone
✗ Legal or administrative terms

=== YOUR TASK ===
Extract healthcare terms and disease mentions from the text that can be linked to DOID entries.

TEXT:
{text}

=== WHAT TO EXTRACT ===
Focus on disease-related entities:
1. Complete disease names: "type 2 diabetes mellitus", "rheumatoid arthritis"
2. Medical conditions: "chronic kidney disease", "heart failure"
3. Infections: "streptococcal infection", "viral pneumonia"
4. Syndromes: "irritable bowel syndrome", "carpal tunnel syndrome"
5. Cancers with specificity: "stage II breast cancer", "non-small cell lung cancer"
6. Mental health conditions: "major depressive disorder", "bipolar disorder"
7. Allergic conditions: "penicillin allergy", "latex hypersensitivity"

=== OUTPUT FORMAT ===
For each healthcare entity found:
ENTITY: [exact_text_from_document] | [5-10 words of context]

For relationships between diseases:
RELATION: [disease1] -> [disease2] | [relationship_type]

=== EXAMPLES ===
ENTITY: type 2 diabetes mellitus | Patient diagnosed with type 2 diabetes mellitus in 2020
ENTITY: hypertension | comorbid conditions include hypertension and hyperlipidemia
ENTITY: seasonal allergies | History of seasonal allergies and asthma
RELATION: diabetes mellitus -> diabetic neuropathy | complication

=== REQUIREMENTS ===
• Extract only disease/condition names, not people, places, or organizations
• Use exact text as it appears in the document
• Provide 5-10 words of surrounding context
• If no healthcare entities found, output nothing
• No explanations or additional text
"""
                
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call(messages, max_tokens=1024)
                
                # Parse additional entities from LLM response
                llm_entities = self._parse_llm_entities(response, text)
                
                # Merge with spaCy entities (avoid duplicates)
                existing_texts = {e['text'] for e in entities}
                for llm_ent in llm_entities:
                    if llm_ent['text'] not in existing_texts:
                        entities.append(llm_ent)
                        existing_texts.add(llm_ent['text'])
                        
            except Exception as e:
                logger.warning(f"Error in LLM entity detection: {e}")
        
        # Create graph nodes
        for i, entity_data in enumerate(entities):
            node_id = f"entity_{i}"
            node = GraphNode(
                entity_text=entity_data['text'],
                start_pos=entity_data['start_pos'],
                end_pos=entity_data['end_pos'],
                context_left=entity_data['context_left'],
                context_right=entity_data['context_right']
            )
            graph_nodes[node_id] = node
        
        return graph_nodes
    
    def _parse_llm_entities(self, response: str, text: str) -> List[Dict]:
        """Parse entities from LLM response"""
        entities = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITY:'):
                entity_part = line[7:].strip()
                if '|' in entity_part:
                    entity_text, context_window = entity_part.split('|', 1)
                    entity_text = entity_text.strip()
                    context_window = context_window.strip()
                    
                    # Find entity position using context matching
                    position_info = self._find_entity_position(text, entity_text, context_window)
                    if position_info:
                        entities.append({
                            'text': entity_text,
                            'start_pos': position_info['start_pos'],
                            'end_pos': position_info['end_pos'],
                            'context_left': position_info['context_left'],
                            'context_right': position_info['context_right']
                        })
        
        return entities
    
    def _find_entity_position(self, text: str, entity_text: str, context_window: str) -> Optional[Dict]:
        """Find entity position in text using entity text and context window"""
        entity_lower = entity_text.lower()
        text_lower = text.lower()
        
        # Try exact entity match first
        start_pos = text_lower.find(entity_lower)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            context_left = text[max(0, start_pos - 50):start_pos]
            context_right = text[end_pos:min(len(text), end_pos + 50)]
            return {
                'start_pos': start_pos,
                'end_pos': end_pos,
                'context_left': context_left,
                'context_right': context_right
            }
        
        return None
    
    def _generate_descriptions(self, node: GraphNode, text: str):
        """Generate descriptions for entity using LLM"""
        if not self.llm_client or (not self.llm_client.model and not self.llm_client.gemini_model):
            # Fallback descriptions
            node.descriptions = [
                f"The entity: {node.entity_text}",
                f"Information about {node.entity_text}",
                f"Details regarding {node.entity_text}"
            ]
            return
        
        try:
            prompt = f"""
KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Generate Medical Search Queries

=== ENTITY TO SEARCH ===
Entity: "{node.entity_text}"
Context: "...{node.context_left} {node.entity_text} {node.context_right}..."

=== YOUR TASK ===
Generate {self.N_DESCRIPTIONS} different medical search queries to find this entity in DOID.
DOID contains diseases, syndromes, infections, genetic disorders, cancers, and medical conditions.

=== SEARCH QUERY GUIDELINES ===
1. Medical terminology (formal names): "diabetes mellitus", "myocardial infarction"
2. Common names (lay terms): "heart attack", "sugar disease"
3. Specific variants: "type 2 diabetes", "non-insulin-dependent diabetes"
4. Category + specificity: "chronic kidney disease", "bacterial pneumonia"
5. Include disease type when relevant: "syndrome", "disorder", "disease", "cancer"

=== EXAMPLES ===
Entity: "Type 2 Diabetes"
DESCRIPTION 1: type 2 diabetes mellitus
DESCRIPTION 2: diabetes mellitus type 2
DESCRIPTION 3: non-insulin-dependent diabetes mellitus

Entity: "breast cancer"
DESCRIPTION 1: breast carcinoma
DESCRIPTION 2: malignant breast neoplasm
DESCRIPTION 3: breast cancer

Entity: "food allergy"
DESCRIPTION 1: food allergy
DESCRIPTION 2: food hypersensitivity
DESCRIPTION 3: allergic reaction to food

=== OUTPUT FORMAT ===
Generate {self.N_DESCRIPTIONS} search queries:
DESCRIPTION 1: [primary medical term]
DESCRIPTION 2: [alternative medical term or synonym]
DESCRIPTION 3: [related term or specification]

Focus on medical terminology that would appear in DOID.
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=256)
            
            # Parse descriptions
            descriptions = self._parse_descriptions(response)
            if descriptions:
                node.descriptions = descriptions[:self.N_DESCRIPTIONS]
            else:
                node.descriptions = [
                    f"The entity: {node.entity_text}",
                    f"Information about {node.entity_text}",
                    f"Details regarding {node.entity_text}"
                ]
                
        except Exception as e:
            logger.warning(f"Error generating descriptions: {e}")
            node.descriptions = [f"Entity: {node.entity_text}"]
    
    def _parse_descriptions(self, response: str) -> List[str]:
        """Parse descriptions from LLM response"""
        descriptions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^(DESCRIPTION\s+\d+|\d+\.?)\s*:', line, re.IGNORECASE):
                colon_idx = line.find(':')
                if colon_idx != -1:
                    description = line[colon_idx + 1:].strip()
                    if description:
                        descriptions.append(description)
            elif line and not line.startswith('DESCRIPTION') and len(line) > 10:
                descriptions.append(line)
        
        return descriptions[:self.N_DESCRIPTIONS]
    
    def _normalize_entity_name(self, entity_text: str) -> List[str]:
        """
        Generate normalized variations of entity name for better candidate matching.
        Handles different languages, naming conventions, and common variations.
        """
        normalized_names = [entity_text]  # Always include original
        
        # Basic normalization
        entity_lower = entity_text.lower().strip()
        if entity_lower != entity_text:
            normalized_names.append(entity_lower)
        
        # Remove common punctuation and special characters
        entity_cleaned = re.sub(r'[^\w\s-]', ' ', entity_text)
        entity_cleaned = ' '.join(entity_cleaned.split())  # Clean whitespace
        if entity_cleaned != entity_text:
            normalized_names.append(entity_cleaned)
        
        # Try common medical term variations
        # Example: "Type 2 Diabetes" -> "type 2 diabetes mellitus", "diabetes mellitus type 2"
        if any(term in entity_lower for term in ['diabetes', 'cancer', 'syndrome', 'disease', 'disorder']):
            # Add "disease" suffix if not present
            if not any(suffix in entity_lower for suffix in ['disease', 'syndrome', 'disorder', 'cancer']):
                normalized_names.append(f"{entity_text} disease")
            
            # Try reordering for "Type X" patterns
            type_match = re.match(r'(type\s+\d+)\s+(.*)', entity_lower, re.IGNORECASE)
            if type_match:
                type_part, rest = type_match.groups()
                normalized_names.append(f"{rest} {type_part}")
        
        # Try LLM-based name normalization if available
        if self.llm_client and (self.llm_client.model or self.llm_client.gemini_model):
            try:
                prompt = f"""
ENTITY: "{entity_text}"

TASK: Suggest 2-3 alternative names or variations for this entity that might appear in a medical ontology.

INSTRUCTIONS:
1. Consider medical terminology variations
2. Include both formal and common names
3. Consider language variations (Latin, English)
4. Keep it short - just the alternative names

OUTPUT FORMAT:
ALT 1: [alternative name]
ALT 2: [alternative name]
ALT 3: [alternative name]

Example for "Type 2 Diabetes":
ALT 1: type 2 diabetes mellitus
ALT 2: diabetes mellitus type 2
ALT 3: non-insulin-dependent diabetes
"""
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.call(messages, max_tokens=128)
                
                # Parse alternative names
                for line in response.strip().split('\n'):
                    if line.strip().startswith('ALT'):
                        match = re.match(r'ALT\s+\d+:\s*(.+)', line.strip(), re.IGNORECASE)
                        if match:
                            alt_name = match.group(1).strip()
                            if alt_name and len(alt_name) > 2:
                                normalized_names.append(alt_name)
            except Exception as e:
                logger.debug(f"Error in LLM name normalization: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in normalized_names:
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                unique_names.append(name)
        
        logger.debug(f"Normalized '{entity_text}' to {len(unique_names)} variations: {unique_names[:3]}...")
        return unique_names
    
    def _search_candidates(self, node: GraphNode):
        """Search for entity candidates"""
        # Normalize entity name to handle language/naming variations
        normalized_names = self._normalize_entity_name(node.entity_text)
        
        # Use normalized names, original text, and descriptions to search
        search_queries = normalized_names + node.descriptions
        all_candidates = []
        
        for query in search_queries:
            if len(query) < 2:
                continue
            
            # Get candidates from entity database
            candidates = self.entity_db.get_candidates(query)
            
            # Convert to dict format with title and description
            for entity_id in candidates:
                entity_name = self.entity_db.get_entity_name(entity_id)
                if entity_name and entity_name != "Unknown":
                    all_candidates.append({
                        'id': entity_id,
                        'title': entity_name,
                        'description': f"Entity: {entity_name}",
                        'score': self.entity_db.get_sitelink_count(entity_id)
                    })
                else:
                    logger.debug(f"  Skipping candidate {entity_id}: entity_name={entity_name}")
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_candidates = []
        for candidate in sorted(all_candidates, key=lambda x: x.get('score', 0), reverse=True):
            if candidate['id'] not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate['id'])
                if len(unique_candidates) >= self.K_SEARCH * 2:
                    break
        
        node.candidates = unique_candidates
        if not unique_candidates:
            # Check if entity looks like a healthcare term
            entity_lower = node.entity_text.lower()
            healthcare_keywords = [
                'disease', 'syndrome', 'cancer', 'tumor', 'tumour', 'carcinoma', 'sarcoma',
                'diabetes', 'fever', 'pain', 'inflammation', 'infection', 'virus', 'bacteria',
                'disorder', 'condition', 'illness', 'symptom', 'sign', 'diagnosis', 'treatment',
                'therapy', 'medication', 'drug', 'pathology', 'lesion', 'malformation', 'anomaly'
            ]
            is_healthcare_term = any(keyword in entity_lower for keyword in healthcare_keywords)
            
            if is_healthcare_term:
                # This might be a healthcare entity, so log as warning
                logger.warning(f"No candidates found for healthcare entity '{node.entity_text}' with queries: {search_queries}")
                # Debug: check what get_candidates returns
                for query in search_queries:
                    if len(query) >= 2:
                        candidates = self.entity_db.get_candidates(query)
                        logger.warning(f"  Query '{query}' -> {len(candidates)} candidates from get_candidates()")
                        if candidates:
                            sample_id = list(candidates)[0]
                            entity_name = self.entity_db.get_entity_name(sample_id)
                            logger.warning(f"    Sample candidate: {sample_id} -> name={entity_name}")
            else:
                # Not a healthcare term, so just debug log (expected to have no candidates)
                logger.debug(f"No candidates found for non-healthcare entity '{node.entity_text}' (expected)")
        else:
            logger.debug(f"Found {len(unique_candidates)} candidates for entity '{node.entity_text}'")
    
    def _select_best_candidate(self, node: GraphNode) -> Optional[str]:
        """Select best candidate using LLM ranking"""
        if not node.candidates:
            return None
        
        if len(node.candidates) == 1:
            return node.candidates[0]['id']
        
        if not self.llm_client or (not self.llm_client.model and not self.llm_client.gemini_model):
            # Fallback: select highest scoring candidate
            best = max(node.candidates, key=lambda x: x.get('score', 0))
            return best['id']
        
        try:
            # Use LLM to rank candidates
            prompt = f"""
KNOWLEDGE BASE: Human Disease Ontology (DOID)
TASK: Medical Entity Disambiguation

=== ENTITY MENTION ===
Mention: "{node.entity_text}"
Clinical Context: "...{node.context_left} {node.entity_text} {node.context_right}..."

=== CANDIDATE DISEASES FROM DOID ===
"""
            
            for i, candidate in enumerate(node.candidates[:self.T_MAX]):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:150]}\n"
            
            prompt += f"""
=== YOUR TASK ===
Select the DOID disease entry that best matches the entity mention based on the clinical context.

=== SELECTION CRITERIA ===
1. **Exact Terminology Match**: Does the candidate name match the medical term used?
   Example: "type 2 diabetes mellitus" matches better than general "diabetes"

2. **Specificity Level**: Is the candidate at the right level of detail?
   Example: "bacterial pneumonia" vs. "pneumonia" vs. "respiratory infection"

3. **Clinical Context**: Does the candidate fit the medical context provided?
   Example: "chronic" vs. "acute" forms, "juvenile" vs. "adult-onset"

4. **Medical Synonyms**: Consider alternative medical terminology
   Example: "myocardial infarction" = "heart attack"

=== DISEASE HIERARCHY EXAMPLE ===
Respiratory Disease (general)
  ├─ Pneumonia (more specific)
  │   ├─ Bacterial pneumonia (most specific)
  │   └─ Viral pneumonia (most specific)
  └─ Bronchitis

Choose the most specific match that fits the context.

=== OUTPUT FORMAT ===
BEST: [number]

Output only the number (1-{min(self.T_MAX, len(node.candidates))}) of the best DOID candidate.
"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.call(messages, max_tokens=128)
            
            # Parse best index
            best_index = self._parse_best_index(response)
            if best_index is not None and 1 <= best_index <= len(node.candidates[:self.T_MAX]):
                return node.candidates[best_index - 1]['id']
            
            # Fallback to highest scoring
            best = max(node.candidates, key=lambda x: x.get('score', 0))
            return best['id']
            
        except Exception as e:
            logger.warning(f"Error in LLM candidate selection: {e}")
            # Fallback to highest scoring
            best = max(node.candidates, key=lambda x: x.get('score', 0))
            return best['id']
    
    def _parse_best_index(self, response: str) -> Optional[int]:
        """Parse best candidate index from LLM response"""
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if 'BEST:' in line.upper():
                line = line[line.upper().index('BEST:') + 5:].strip()
            
            numbers = re.findall(r'\d+', line)
            if numbers:
                try:
                    return int(numbers[0])
                except ValueError:
                    continue
        
        return None
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities using graph-based approach"""
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # Build entity graph
        graph_nodes = self._build_entity_graph(text, doc)
        
        # Process each node
        for node_id, node in graph_nodes.items():
            # Generate descriptions
            self._generate_descriptions(node, text)
            
            # Search for candidates
            self._search_candidates(node)
            
            # Select best candidate
            entity_id = self._select_best_candidate(node)
            
            if entity_id is None:
                entity_id = UnknownEntity.NIL.value
            
            span = (node.start_pos, node.end_pos)
            candidates = {c['id'] for c in node.candidates}
            predictions[span] = EntityPrediction(span, entity_id, candidates)
        
        return predictions


