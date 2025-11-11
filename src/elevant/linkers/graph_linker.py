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
        # - Hyperlink candidates for popular mention->entity mappings
        # - Sitelink counts for popularity-based scoring
        try:
            # Only load entity names if not already loaded (e.g., by custom KB)
            if not self.entity_db.entity_name_db:
                self.entity_db.load_entity_names()
            # Ensure name_to_entities_db is loaded for get_candidates()
            if not self.entity_db.name_to_entities_db:
                self.entity_db.load_name_to_entities()
            self.entity_db.load_alias_to_entities()
            self.entity_db.load_hyperlink_to_most_popular_candidates()
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
TEXT: {text}

CONTEXT: You are a medical ontology expert working with the Human Disease Ontology (DOID). DOID is a comprehensive hierarchical controlled vocabulary for human diseases. Your task is to extract ONLY entities that can be mapped to DOID entries.

WHAT IS DOID?
The Human Disease Ontology (DOID) contains:
- Disease names (e.g., "type 2 diabetes mellitus", "breast cancer", "Alzheimer's disease")
- Medical conditions (e.g., "hypertension", "asthma", "arthritis")
- Syndromes (e.g., "Down syndrome", "metabolic syndrome")
- Infectious diseases (e.g., "tuberculosis", "malaria", "COVID-19")
- Genetic disorders (e.g., "cystic fibrosis", "sickle cell anemia")
- Mental health conditions (e.g., "depression", "schizophrenia", "anxiety disorder")
- Allergies and hypersensitivities (e.g., "peanut allergy", "drug allergy")
- Cancers and tumors (e.g., "angiosarcoma", "lymphoma", "glioblastoma")

INSTRUCTIONS:
1. Extract ONLY entities that are likely DOID entries:
   ✓ Disease names (specific and general)
   ✓ Medical conditions and disorders
   ✓ Syndromes and symptom complexes
   ✓ Infectious diseases
   ✓ Allergies and hypersensitivity conditions
   ✓ Cancer types and tumors
   ✓ Genetic/hereditary conditions
   ✓ Mental health disorders
   
2. EXCLUDE non-disease entities:
   ✗ People names (e.g., "Dr. Smith", "THOMSON")
   ✗ Locations/Places (e.g., "Kew Gardens", "Surrey", "Leven Park")
   ✗ Organizations (e.g., "WHO", "hospital names")
   ✗ Dates and times (e.g., "1891", "37 years")
   ✗ Numbers and measurements
   ✗ Anatomical parts UNLESS part of disease name (e.g., "heart" alone is ✗, but "heart disease" is ✓)
   ✗ Symptoms UNLESS they are a defined condition (e.g., "fever" alone is ✗, but "rheumatic fever" is ✓)
   ✗ Legal/regulatory terms (e.g., "Dangerous Drugs Acts")
   ✗ General medical procedures (e.g., "surgery", "vaccination" without disease context)

3. Prefer complete disease names:
   - Good: "type 2 diabetes mellitus", "acute bronchitis", "peanut allergy"
   - Avoid: "diabetes" alone if text says "diabetes mellitus"

OUTPUT FORMAT:
For each DOID-likely entity:
ENTITY: [entity_text] | [context_window]

For disease relationships:
RELATION: [entity1_text] -> [entity2_text] | [relation_type]

EXAMPLES:
ENTITY: type 2 diabetes mellitus | diagnosed with type 2 diabetes mellitus and hypertension
ENTITY: hypertension | diabetes mellitus and hypertension in elderly patients
ENTITY: aspirin allergy | patient developed aspirin allergy after treatment
ENTITY: angiosarcoma | rare vascular cancer angiosarcoma was detected

IMPORTANT:
- Match text exactly as it appears
- Context should be 5-10 words surrounding the entity
- ONLY output entities mappable to DOID (diseases, conditions, syndromes)
- If no DOID-relevant entities found, output nothing
- No explanations, only ENTITY and RELATION lines
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
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

TASK: Generate {self.N_DESCRIPTIONS} different search queries to find this entity in the Human Disease Ontology (DOID). Each description should help match this entity to a disease, medical condition, or health-related term in DOID.

CONTEXT: DOID (Human Disease Ontology) contains diseases, medical conditions, syndromes, infections, allergies, and health disorders. Your descriptions should help find the correct DOID entry.

INSTRUCTIONS:
1. Think about what type of disease/condition this might be
2. Consider medical terminology and common names
3. Include relevant category terms (e.g., "diabetes", "cancer", "syndrome", "disease", "disorder")
4. Be specific but also consider variations

OUTPUT FORMAT:
DESCRIPTION 1: [medical description or search term]
DESCRIPTION 2: [alternative medical term or category]
DESCRIPTION 3: [related disease terminology]

EXAMPLES:
For entity "type 2 diabetes":
DESCRIPTION 1: type 2 diabetes mellitus
DESCRIPTION 2: non-insulin-dependent diabetes
DESCRIPTION 3: adult-onset diabetes

For entity "peanut allergy":
DESCRIPTION 1: peanut allergy
DESCRIPTION 2: peanut hypersensitivity
DESCRIPTION 3: legume allergy peanut

Make each description medically relevant and helpful for finding the correct DOID entry.
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
    
    def _search_candidates(self, node: GraphNode):
        """Search for entity candidates"""
        # Use entity text and descriptions to search
        search_queries = [node.entity_text] + node.descriptions
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
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

TASK: Select the best matching disease/condition from the Human Disease Ontology (DOID) for the entity above.

CANDIDATES FROM DOID:
"""
            
            for i, candidate in enumerate(node.candidates[:self.T_MAX]):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:150]}\n"
            
            prompt += f"""
INSTRUCTIONS:
1. Consider the medical context and exact wording
2. Match based on disease terminology and specificity
3. Prefer exact medical terminology matches
4. Consider disease hierarchy (specific vs. general conditions)
5. If the entity mentions a specific subtype, choose the specific DOID entry

RANKING CRITERIA:
- Exact match of disease name (highest priority)
- Semantic similarity in medical terminology
- Correct level of specificity (e.g., "type 2 diabetes mellitus" vs. "diabetes mellitus")
- Context appropriateness

OUTPUT FORMAT:
BEST: [number]

Only output the number (1-{min(self.T_MAX, len(node.candidates))}) of the best matching DOID candidate.
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


