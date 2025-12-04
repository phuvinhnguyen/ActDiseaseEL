"""
Graph-based entity linker using transformers LLM
Simplified version with batch processing for faster inference
"""
from typing import Dict, Tuple, Optional, Any, List
from spacy.tokens import Doc
from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.utils.knowledge_base_mapper import UnknownEntity
import ahocorasick, difflib, re, nltk
from symspellpy import SymSpell, Verbosity
from functools import cache
from nltk.corpus import stopwords
import re

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def build_entity_matcher(entities: Dict) -> Tuple[Any, Dict, Any]:
    all_terms = []
    term_to_entities = {}  # Map term to list of entities
    
    for entity_id, entity in entities.items():
        # Add main name
        name = entity['name']
        if name.lower() not in term_to_entities:
            term_to_entities[name.lower()] = []
        term_to_entities[name.lower()].append(entity)
        all_terms.append(name)
        
        # Add synonyms
        for synonym in entity.get('synonyms', []):
            if synonym.lower() not in term_to_entities:
                term_to_entities[synonym.lower()] = []
            term_to_entities[synonym.lower()].append(entity)
            all_terms.append(synonym)
    
    sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=5)
    for term in all_terms:
        sym_spell.create_dictionary_entry(term, 1)

    automaton = ahocorasick.Automaton()
    for term in term_to_entities.keys():
        automaton.add_word(term, term)
    automaton.make_automaton()
    
    return automaton, term_to_entities, sym_spell


@cache
def parse_obo_file(obo_path):
    """Parse OBO file to extract entity information with descriptions"""
    ontology = {}
    
    with open(obo_path, 'r', encoding='utf-8') as f:
        current_term = None
        in_term = False
        
        for line in f:
            line = line.strip()
            
            # Start of a new term
            if line == '[Term]':
                if current_term and 'id' in current_term:
                    # Save previous term
                    ontology[current_term['id']] = current_term
                current_term = {'synonyms': [], 'xrefs': []}
                in_term = True
                continue
            
            # End of term section
            if line.startswith('[') and line != '[Term]':
                if current_term and 'id' in current_term:
                    ontology[current_term['id']] = current_term
                in_term = False
                current_term = None
                continue
            
            if not in_term or not current_term:
                continue
            
            # Parse term fields
            if line.startswith('id: '):
                current_term['id'] = line[4:].strip()
            
            elif line.startswith('name: '):
                current_term['name'] = line[6:].strip()
            
            elif line.startswith('def: '):
                # Extract definition (remove quotes and references)
                def_match = re.match(r'def: "(.+?)"', line)
                if def_match:
                    current_term['def'] = def_match.group(1)
            
            elif line.startswith('synonym: '):
                # Extract synonym (remove quotes)
                syn_match = re.match(r'synonym: "(.+?)"', line)
                if syn_match:
                    current_term['synonyms'].append(syn_match.group(1))
            
            elif line.startswith('alt_id: '):
                if 'alt_ids' not in current_term:
                    current_term['alt_ids'] = []
                current_term['alt_ids'].append(line[8:].strip())
            
            elif line.startswith('xref: '):
                current_term['xrefs'].append(line[6:].strip())
        
        # Don't forget the last term
        if current_term and 'id' in current_term:
            ontology[current_term['id']] = current_term
    
    return build_entity_matcher(ontology)

def find_near_matches_for_span(span_text: str, term_to_entities: Dict, 
                              sym_spell: Any, top_k: int = 10, 
                              min_confidence: float = 0.7, 
                              min_similarity: float = 0.7) -> List[Dict]:
    span_text_lower = span_text.lower()
    results = []
    
    if len(span_text_lower.strip()) < 3 or span_text_lower in stopwords.words('english'):
        return []
    
    sym_matches = sym_spell.lookup(span_text_lower, Verbosity.ALL, max_edit_distance=3)
    for match in sym_matches:
        term = match.term
        distance = match.distance
        
        # Calculate confidence score
        max_len = max(len(span_text), len(term))
        confidence = 1.0 - (distance / max_len) if max_len > 0 else 0
        
        # Only keep high confidence matches
        if confidence >= min_confidence and term in term_to_entities:
            for entity in term_to_entities[term]:
                entity_info = entity.copy()
                entity_info['match_type'] = 'symspell'
                entity_info['edit_distance'] = distance
                entity_info['matched_term'] = term
                entity_info['confidence_score'] = confidence
                results.append(entity_info)
    
    # Method 2: Difflib with similarity ratio (apply min_similarity threshold)
    all_terms = list(term_to_entities.keys())
    diff_matches = difflib.get_close_matches(span_text_lower, all_terms, 
                                           n=top_k*2, cutoff=min_similarity)
    
    for matched_term in diff_matches:
        ratio = difflib.SequenceMatcher(None, span_text_lower, matched_term).ratio()
        # Additional filtering by our min_similarity threshold
        if ratio >= min_similarity:
            for entity in term_to_entities[matched_term]:
                # Check if this entity is already in results
                existing_ids = [r['id'] for r in results]
                if entity['id'] not in existing_ids:
                    entity_info = entity.copy()
                    entity_info['match_type'] = 'difflib'
                    entity_info['similarity_ratio'] = ratio
                    entity_info['matched_term'] = matched_term
                    entity_info['confidence_score'] = ratio
                    results.append(entity_info)
    
    # Method 3: Partial matching with higher quality requirements
    partial_count = 0
    for term, entity_list in term_to_entities.items():
        if span_text_lower in term or term in span_text_lower:
            # Calculate confidence for partial matches
            if span_text_lower in term:
                confidence = len(span_text_lower) / len(term)
            else:
                confidence = len(term) / len(span_text_lower)
            
            # Only keep high quality partial matches
            if confidence >= min_confidence:
                for entity in entity_list:
                    existing_ids = [r['id'] for r in results]
                    if entity['id'] not in existing_ids:
                        entity_info = entity.copy()
                        entity_info['match_type'] = 'partial'
                        entity_info['matched_term'] = term
                        entity_info['confidence_score'] = confidence
                        results.append(entity_info)
                        partial_count += 1
    
    # Remove duplicates and sort by confidence score
    unique_results = {}
    for result in results:
        if result['id'] not in unique_results or result['confidence_score'] > unique_results[result['id']]['confidence_score']:
            unique_results[result['id']] = result
    
    # Final filtering by confidence and return top_k
    high_confidence_results = [r for r in unique_results.values() if r['confidence_score'] >= min_confidence]
    final_results = sorted(high_confidence_results, key=lambda x: x['confidence_score'], reverse=True)[:top_k]
    return final_results

def extract_spans_with_entities(text: str, entities_matcher: Dict, top_k: int = 10, 
                               min_span_length: int = 2, max_span_length: int = 4,
                               min_confidence: float = 0.7,
                               min_similarity: float = 0.7) -> Dict:
    automaton, term_to_entities, sym_spell = entities_matcher
    
    # Split text into words and track their positions
    words = []
    word_positions = []  # (start_char, end_char) for each word
    
    # Tokenize text while preserving positions
    word_pattern = re.compile(r'\S+')
    for match in word_pattern.finditer(text):
        word = match.group()
        start = match.start()
        end = match.end()
        words.append(word)
        word_positions.append((start, end))
    
    spans_info = {}
    total_spans_checked = 0
    
    # Generate n-grams of different lengths
    for n in range(min_span_length, max_span_length + 1):
        for i in range(len(words) - n + 1):
            span_words = words[i:i + n]
            span_text = ' '.join(span_words)
            
            # Skip spans that are mostly stopwords
            non_stopwords = [w for w in span_words if w.lower() not in stopwords.words('english')]
            if len(non_stopwords) == 0:
                continue
            
            # Get exact character positions from word boundaries
            # Start at the beginning of first word, end at the end of last word
            start_char = word_positions[i][0]  # Start of first word
            end_char = word_positions[i + n - 1][1]  # End of last word
            
            # Extract the actual text at this position (includes spaces between words)
            actual_text = text[start_char:end_char]
            
            # Normalize both for comparison (handle multiple spaces)
            actual_words = actual_text.split()
            span_words_list = span_text.split()
            
            # Verify we have the same words in the same order (case-insensitive)
            if len(actual_words) != len(span_words_list):
                continue
            if any(aw.lower() != sw.lower() for aw, sw in zip(actual_words, span_words_list)):
                continue
            
            # Use the normalized span_text for entity matching (single spaces)
            # But store actual_text for display/context
            span_text_for_matching = ' '.join(actual_words)  # Normalized version
            
            span_key = (start_char, end_char)
            
            # Only process if we haven't seen this exact span before
            if span_key not in spans_info:
                total_spans_checked += 1
                # Find near matches for this span with quality filtering
                # Use normalized text for matching
                near_entities = find_near_matches_for_span(
                    span_text_for_matching, term_to_entities, sym_spell, 
                    top_k=top_k, 
                    min_confidence=min_confidence,
                    min_similarity=min_similarity
                )
                
                if near_entities:  # Only include spans that have high-quality matches
                    # Store the actual text from the original (preserves original spacing)
                    spans_info[span_key] = {
                        'span_text': span_text_for_matching,  # Normalized for consistency
                        'start_char': start_char,
                        'end_char': end_char,
                        'entities': near_entities
                    }
    
    return spans_info

def find_entity_spans_in_text(text: str, obo_path: str, top_k: int = 10, 
                             min_span_length: int = 2, max_span_length: int = 4,
                             min_confidence: float = 0.7,
                             min_similarity: float = 0.7) -> Dict:
    # Parse OBO file
    entities = parse_obo_file(obo_path)
    
    # Extract spans with high-quality entities - work directly with original text
    # This ensures positions are correct relative to the original text
    spans_result = extract_spans_with_entities(
        text, entities, 
        top_k=top_k, 
        min_span_length=min_span_length, 
        max_span_length=max_span_length,
        min_confidence=min_confidence,
        min_similarity=min_similarity
    )
    
    return spans_result

class GraphLinker(AbstractEntityLinker):
    """
    Graph-based entity linker following graph_system.py approach
    Uses transformers LLM instead of API
    """
    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any],
                 obo_path: str = '/media/volume/LLMRag2/.local/HumanDiseaseOntology/src/ontology/doid-merged.obo',
                 verbose: bool = True,
                 multilingual: bool = True
                 ):
        self.entity_db = entity_database
        self.model = None
        self.obo_path = obo_path
        self.verbose = verbose
        self.multilingual = multilingual

        # Get config variables
        self.linker_identifier = config.get("linker_name", "Graph LLM")
        self.ner_identifier = self.linker_identifier
        
        # LLM client
        from elevant.llm_client import LLMClient
        model_path = config.get("llm_model_path", None)
        self.llm_client = LLMClient(model_path) if model_path else None

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
                pass
            
            self.entity_db.load_sitelink_counts()
        except Exception as e:
            pass

        # Graph-specific parameters
        self.N_DESCRIPTIONS = config.get("n_descriptions", 3)
        self.K_SEARCH = config.get("k_search", 10)
        self.T_MAX = config.get("t_max", 5)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.7)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _llm_ner(self, text: str) -> List[Dict]:
        text_short_chunks = [(i, text[i:i+600]) for i in range(0, len(text), 550)]
        
        ner_prompts = '''
You are a DOID disambiguation expert.  
For every disease or syndrome mentioned in the text, pick the best English name(s) from DOID/MeSH/ICD-10.

## TEXT  
"{text}"

## RESPONSE FORMAT RULES  
CRITICAL: You MUST follow this format:
- Include reasoning lines to explain your thought process about identifying and mapping entities
- After reasoning, output each entity on a separate line in the exact format: ENTITY: <exact substring from text>: <English name 1>, <English name 2>, ...
- You may interleave reasoning with ENTITY lines, but each ENTITY line must be correctly formatted
- Use the EXACT substring from the text (don't modify spelling/capitalization)
- Provide English equivalent names separated by commas
- If no diseases found, output nothing (empty response)

## STEP-BY-STEP PROCESSING:
1. Read the text carefully and identify ALL disease/syndrome mentions
2. For each mention, extract the EXACT substring as it appears in text
3. Map to appropriate English medical terms from DOID/MeSH/ICD-10
4. Output reasoning followed by ENTITY lines in the required format

## EXAMPLES - PAY CLOSE ATTENTION TO FORMAT:

**Example 1:**
TEXT: "Den här patienten lider av både diabetes, högt blodtryck och mild depression."
OUTPUT:
I know that in this Swedish text, "diabetes" and "mild depression" are similar to same name in English, so I will output the same name for both.
ENTITY: diabetes: diabetes
ENTITY: mild depression: mild depression
"högt blodtryck" is a Swedish word for hypertension (high blood pressure), so I will output the English name for it.
ENTITY: högt blodtryck: hypertension, high blood pressure  

**Example 2:**
TEXT: "On diagnostique chez le patient un infarctus du myocarde, une insuffisance cardiaque et un asthme sévère."
OUTPUT:
ENTITY: infarctus du myocarde: myocardial infarction, heart attack
ENTITY: insuffisance cardiaque: heart failure, cardiac failure
ENTITY: asthme sévère: severe asthma

**Example 3:**
TEXT: "The patient suffers from MI and CHF."
OUTPUT:
I understand that "MI" is a common medical abbreviation for myocardial infarction, which is also known as heart attack. "CHF" stands for congestive heart failure, and it can be referred to as chronic heart failure.
ENTITY: MI: myocardial infarction, heart attack
ENTITY: CHF: congestive heart failure, chronic heart failure

**Example 4:**
TEXT: "No significant medical history noted."
OUTPUT:
[After analyzing the text, I find no disease mentions, so I output nothing]

## YOUR TASK:
Now process the following text. Output ONLY in the specified format, no other text:

TEXT: "{text}"
'''
        
        prompts = [[
            {"role": "system", "content": "You are a DOID NER expert. For every disease or syndrome mentioned in the text, detect and pick the best English name(s) from DOID/MeSH/ICD-10. Output in the specified format."},
            {"role": "user", "content": ner_prompts.format(text=text)}
            ] for _, text in text_short_chunks]
        responses = self.llm_client.call_batch(prompts)
        detected_entities = []
        for (i, text), response in zip(text_short_chunks, responses):
            for line in response.split('\n'):
                line = line.strip()
                if not line or not line.startswith('ENTITY:'): 
                    continue
                try:
                    mention, entity_english_names = line.replace('ENTITY:', '', 1).strip().split(':')
                    start_pos = i + text.find(mention)
                    end_pos = start_pos + len(mention)

                    entities = find_entity_spans_in_text(entity_english_names, self.obo_path, top_k=20, min_span_length=2, max_span_length=4, min_confidence=0.6, min_similarity=0.6)
                    entities = sum([i['entities'] for i in list(entities.values())], [])[:20]
                    
                    entity_info = {
                        'text': mention,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'context_left': text[max(0, start_pos - 500):start_pos],
                        'context_right': text[end_pos:min(len(text), end_pos + 500)],
                        'link_entities': {},
                        'confidence': 0.0,
                        'candidates': entities
                    }
                    detected_entities.append(entity_info)
                except Exception as e:
                    continue
        return detected_entities
    
    def _detect_entities_with_llm(self, text: str) -> List[Dict]:
        """Use LLM to detect medical entities from text"""
        span_dict = find_entity_spans_in_text(
            text, self.obo_path, 
            top_k=20, 
            min_span_length=2, 
            max_span_length=3, 
            min_confidence=0.8, 
            min_similarity=0.8
        )

        self._log(f"Span dictionary ({len(span_dict.values())}): {span_dict}"[:500])
        
        # Convert to list for easier indexing
        spans_list = list(span_dict.values())
        
        if not spans_list: return []
        
        # Format spans for prompt
        spans_text = '\n'.join([
            f"{idx+1}. Positions {span['start_char']}-{span['end_char']}: \"{span['span_text']}\""
            for idx, span in enumerate(spans_list)
        ])
        
        # Create strict prompt
        prompt = f"""You validate disease mentions for the DOID knowledge base.

TEXT
-----
{text}

CANDIDATE SPANS
---------------
{spans_text if spans_text else 'No candidates'}

TASK
----
1. Select ONLY spans that clearly refer to diseases, syndromes, or medical conditions.
2. Ignore spans that refer to people, places, procedures, dates, numbers, or anatomy alone.
3. Your respond must contain many lines (or one line) that starts with 'INDEX:' followed by the index of valid spans and the corrected (normalized) text of that span (for example, if span is "this", the text after valid index should be the name of entity related to "this" in English).
4. If no spans qualify, do not provide a line with 'INDEX:' in your answer.

ALLOWED OUTPUT (STRICT)
-----------------------
Example 1:
I believe 1 ("this"),3 ("herpas zostar"), and 5 ("pseudomonas") are diseases.
INDEX: 1: infectious mononucleosis
INDEX: 3: herpes zoster
INDEX: 5: pseudomonas

Example 2:
After reading the text, I confirm that 1 ("this disease") mentions to szymczak's syndrome.
INDEX: 1: szymczak's syndrome

Example 3:
After reading this Swedish paper, I believe 1 ("denna sjukdom"), 2 ("the disease"), 3 ("infektion"), 4 ("influensa"), and 5 ("lunginflammation") are related to DOID.
INDEX: 1: pneumonia
INDEX: 2: infection
INDEX: 3: infection
INDEX: 4: influenza
INDEX: 5: lung inflammation

Different output formats are not allowed.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.call(messages)
        detected_entities = []
        for line in response.split('\n'):
            line = line.strip()
            if not line or not line.startswith('INDEX:'): 
                continue
            
            try:
                # Extract index numbers
                index_str = line.replace('INDEX:', '').strip()
                idx, index_text = index_str.split(':')
            
                if 1 <= idx <= len(spans_list):
                    span = spans_list[idx - 1]
                    specific_spans = find_entity_spans_in_text(index_text, self.obo_path, top_k=20, min_span_length=2, max_span_length=3, min_confidence=0.8, min_similarity=0.8)
                    specific_spans_list = sum([i['entities'] for i in list(specific_spans.values())], [])[:10]
                    entity_info = {
                        'text': span['span_text'],
                        'start_pos': span['start_char'],
                        'end_pos': span['end_char'],
                        'context_left': text[max(0, span['start_char'] - 500):span['start_char']],
                        'context_right': text[span['end_char']:min(len(text), span['end_char'] + 500)],
                        'link_entities': {},
                        'confidence': 0.0,
                        'candidates': specific_spans_list + span['entities']
                    }
                    detected_entities.append(entity_info)
                else:
                    continue
            except Exception as e:
                continue
        
        return detected_entities
    
    def _display_candidates(self, candidates: List[Dict]) -> str:
        text = ''
        def_format = 'Definition: {definition}'
        synonyms_format = 'Synonyms: {synonyms}'
        cands = [
            {
                'name': cand['name'],
                'def': def_format.format(definition=cand.get('def', '')) + '\n' + synonyms_format.format(synonyms=cand.get('synonyms', ''))
            } for cand in candidates
            ]
        if len(candidates) > 30:
            for i, cand in enumerate(cands): text += f'{i+1}. {cand['name']} ({cand['def'][:30]})\n'
        elif len(candidates) > 10:
            for i, cand in enumerate(cands): text += f'{i+1}. {cand['name']}:\n{cand['def'].split('\n')[0]}\n'
        else:
            for i, cand in enumerate(cands): text += f'{i+1}. {cand['name']}:\n{cand['def']}\n'

        return text

    def _parse_linked_entity(self, output: str) -> Dict:
        entities = {}
        for line in output.split('\n'):
            if line.startswith('ENTITY:'):
                try:
                    content = line.replace('ENTITY:', '', 1).strip()
                    parts = [p.strip() for p in content.split(':') if p.strip()]
                    if len(parts) < 2:
                        continue
                    index = int(parts[0])
                    confidence = float(parts[1])
                    entities[index - 1] = confidence
                except Exception:
                    continue
        return entities

    def _get_candidates_for_entities(self, entities: List[Dict]) -> List[Dict]:
        """Get candidates for all entities efficiently"""
        
        candidate_prompts = []
        confirmed_entities = [i for i in entities if i['confidence'] > self.HIGH_CONFIDENCE_THRESHOLD]
        unconfirmed_entities = [i for i in entities if i['confidence'] <= self.HIGH_CONFIDENCE_THRESHOLD]
        
        self._log(f"Confirmed entities: {len(confirmed_entities)}")
        self._log(f"Unconfirmed entities: {len(unconfirmed_entities)}")

        for ent_idx, entity in enumerate(unconfirmed_entities):
            prompt = self._create_linking_prompt(entity, entity['candidates'], confirmed_entities or None)
            candidate_prompts.append(prompt)

        # Only call batch if we have prompts to process
        if candidate_prompts and self.llm_client:
            responses = self.llm_client.call_batch(candidate_prompts)
            linked_results = [self._parse_linked_entity(i) for i in responses]
        else:
            linked_results = [{} for _ in range(len(unconfirmed_entities))]

        for i in range(len(unconfirmed_entities)):
            result = linked_results[i]
            
            indexes = list(result.keys())
            if indexes:
                unconfirmed_entities[i]['candidates'] = [unconfirmed_entities[i]['candidates'][index] for index in indexes if 0 <= index < len(unconfirmed_entities[i]['candidates'])]
            
            for index, confidence in result.items():
                if confidence > unconfirmed_entities[i]['confidence']:
                    unconfirmed_entities[i]['confidence'] = confidence
                    if 0 <= index < len(unconfirmed_entities[i]['candidates']):
                        unconfirmed_entities[i]['link_entities'] = unconfirmed_entities[i]['candidates'][index]
        return confirmed_entities + unconfirmed_entities

    
    def _create_linking_prompt(self, entity: Dict, candidates: List[Dict], other_entities: List[Dict] = None) -> List[Dict]:        
        context = ' '.join(f"{entity['context_left']} ###{entity['text']}### {entity['context_right']}".split())
        top_candidates = candidates[:self.T_MAX]
        candidates_text = self._display_candidates(top_candidates)
        other_entities_text = ''
        if other_entities:
            other_info = []
            for e in other_entities:
                if isinstance(e, dict) and e.get('text') != entity['text']:
                    link = e.get('link_entities', {})
                    if link.get('id'):
                        other_info.append(f"- {e['text']} -> {link.get('name', 'Unknown')} ({link.get('def', link.get('synonyms', ''))[:100]}...)")
            other_entities_text = f"\nLINKED ENTITIES IN TEXT:\n" + '\n'.join(other_info[:3]) + "\n" if other_info else ''
        
        prompt = f"""You are a DOID disambiguation expert. Pick the best candidate disease.

## MENTION
"{entity['text']}"

## CONTEXT
{context}
{other_entities_text}
## CANDIDATE DISEASES (1-based index)
{candidates_text if candidates_text else 'No candidates provided'}

## RESPONSE RULES
1. Output one line per accepted candidate.
2. Each line MUST match exactly: ENTITY: <index>: <confidence>
3. Confidence must be between 0.0 and 1.0. Higher = better match.
4. If no candidates fit, output nothing.
5. If you are unsure about the candidate, out low confidence score.
6. You should output many candidates if you think they are related to the mention.
7. If "No candidates provided", output nothing.

## EXAMPLE OUTPUT:
**Example Output 1:**
I confirm that the mention is "agranulocytic angina", which is entity 2.
ENTITY: 2: 1.0

**Example Output 2:**
I found that "itis" (5th entity) is a common suffix for many diseases and I am not sure about the exact disease, so I output low confidence score.
ENTITY: 5: 0.5

**Example Output 3:**
I found that "itis" (1), "phagia" (6), "angina" (9), "fissure" (4) are all related to diseases, I will need more information about them to decide which one is the best candidate with high confidence, so I output multiple candidates (with entity I want to know more about) with low confidence scores.
ENTITY: 1: 0.5
ENTITY: 6: 0.4
ENTITY: 9: 0.3
ENTITY: 4: 0.35
"""
        return [
            {"role": "system", "content": "You are a DOID disambiguation expert. Pick the best candidate disease."},
            {"role": "user", "content": prompt}
            ]
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        predictions = {}
        
        # Step 1: Detect entities
        if self.multilingual:
            print("Using multilingual mode")
            detected_entities = self._llm_ner(text)
        else:
            print("Using non-multilingual mode")
            detected_entities = self._detect_entities_with_llm(text)
        
        if not detected_entities:
            return predictions
        
        for _ in range(3): detected_entities = self._get_candidates_for_entities(detected_entities)
        
        for i, entity in enumerate(detected_entities):
            if entity['confidence'] < self.HIGH_CONFIDENCE_THRESHOLD:
                continue
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = entity.get('link_entities', {}).get('id') or UnknownEntity.NIL.value
            candidates = {c['id'] for c in entity.get('candidates', [])}            
            predictions[span] = EntityPrediction(span, entity_id, candidates)
            
        return predictions


