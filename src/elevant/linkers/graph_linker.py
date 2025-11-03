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
        
        # First, detect entities using spaCy
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
        
        # If LLM client available, try to enhance detection and get relations
        if self.llm_client and self.llm_client.model:
            try:
                # Use LLM to detect additional entities and relations
                prompt = f"""
TEXT: {text}

INSTRUCTIONS:
1. Identify all important entities (people, organizations, locations, products, events, etc.)
2. For each entity, provide the entity text and a short context window around it
3. Identify relationships between entities

OUTPUT FORMAT:
For each entity, output:
ENTITY: [entity_text] | [context_window]

For each relation, output:
RELATION: [entity1_text] -> [entity2_text] | [relation_type]

Example:
ENTITY: Apple | technology company Apple Inc. is headquartered
ENTITY: California | headquartered in Cupertino, California
RELATION: Apple -> California | located_in

IMPORTANT:
- Entity text must match exactly what appears in the text
- Context window should be 5-10 words around the entity
- Only output entities and relations, no other text
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
        if not self.llm_client or not self.llm_client.model:
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

INSTRUCTIONS:
Generate {self.N_DESCRIPTIONS} different descriptions for this entity. Each description should be a short phrase that helps identify what this entity is.

OUTPUT FORMAT:
DESCRIPTION 1: [description1]
DESCRIPTION 2: [description2]
DESCRIPTION 3: [description3]

Example:
DESCRIPTION 1: A technology company
DESCRIPTION 2: A fruit company  
DESCRIPTION 3: A multinational corporation

Make each description distinct and informative.
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
                if entity_name:
                    all_candidates.append({
                        'id': entity_id,
                        'title': entity_name,
                        'description': f"Entity: {entity_name}",
                        'score': self.entity_db.get_sitelink_count(entity_id)
                    })
        
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
    
    def _select_best_candidate(self, node: GraphNode) -> Optional[str]:
        """Select best candidate using LLM ranking"""
        if not node.candidates:
            return None
        
        if len(node.candidates) == 1:
            return node.candidates[0]['id']
        
        if not self.llm_client or not self.llm_client.model:
            # Fallback: select highest scoring candidate
            best = max(node.candidates, key=lambda x: x.get('score', 0))
            return best['id']
        
        try:
            # Use LLM to rank candidates
            prompt = f"""
ENTITY: "{node.entity_text}"
CONTEXT: "...{node.context_left} {node.entity_text} {node.context_right}..."

CANDIDATES:
"""
            
            for i, candidate in enumerate(node.candidates[:self.T_MAX]):
                prompt += f"{i+1}. {candidate['title']} - {candidate['description'][:150]}\n"
            
            prompt += f"""
INSTRUCTIONS:
Select the best candidate for the entity above.

OUTPUT FORMAT:
BEST: [number]

Only output the number of the best candidate.
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


