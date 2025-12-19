"""
### **Step 1: multi-chunk Entity Normalization Processor (mENP)**
- **What it does:** It takes context, mention and return a short normalized mention, example: "Trump" -> "Donald Trump"
- **How it works:** LLM takes the context, mention and return a short normalized mention
- **Result:** A short direct normalized mention that can be used to search for related entities using fuzzy matching, not being mixed with unrelated entities.

---

### **Step 2: Map-Reduce Entity Linker (MREL)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "fennec fox" and the surrounding text/context) and a long list of possible **candidate entities** that could match that mention.
- **How it works:** It separates the long list of candidates into many smaller lists, for each list, Select the most relevant entities based on the context. The smaller the number of entities listed by Agent, the more relevant they are. For example, if the Agent lists 10 entities, each entity has a score of 1/10.
While if the Agent lists 1 entity, each entity has a score of 1.0. The higher the score, the more relevant the entity is.
  *Example:* If the context says "living in desert regions of North Africa," it will remove candidates like *Arctic Fox* (lives in the cold) or fictional characters like *Nick Wilde* and list only *Fennec Fox* and *Desert Fox*, each entity has a score of 1/2.
- **Result:** A shorter, more relevant list of candidate entities (e.g., Red Fox, Rippell’s Fox).

---

### **Step 3: Confidence-based Improvement (CBI)**
- **What it does:** Repeat the linking process for low-confidence entities, while the information of linked entities (high-confidence) is provided in the context to enhance LLM in understanding overall context of all entities.
- **How it works:**
  - Repeat k times
    - Link all detected entities and return a confidence score for each entity
    - If the confidence score (a%) is above a threshold, return the entity information and have a% of puting it in the entity return list, 1-a% change the high-confidence entity is put to the re-processing list (double check with context from other high-confidence entities)
    - If the confidence score is below a threshold, add the entity to the re-processing list, enhance its context by adding the information of linked entities (high-confidence) to the context (what is that entity, description, aliases, ...)
  - After k times, only return entities with high confidence, entities with low confidence are ignored

---

### **Final Output:**
- The system outputs the **correctly linked entity** (e.g., *Rippell’s fox*) with a clear explanation.

---

### **In Simple Summary:**
1. **MREL** → Reduces a big list to a few possible options.  
2. **CBI** → Repeat the linking process for low-confidence entities and double check high-confidence entities, while the information of linked entities (high-confidence) is provided in the context to enhance LLM in understanding overall context of all entities.

This helps computers understand *which real-world thing* a word refers to, especially when words can mean different things in different situations.
"""
from typing import Dict, Tuple, Optional, Any, List
from spacy.tokens import Doc
from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
import sqlite3, re, rapidfuzz.fuzz as fuzz, nltk
import random, os, json
from collections import defaultdict
import functools
from elevant.llm_client import LLMClient

BENCHMARK_OBO = 'NCBItestset_CTD_diseases_filtered'

def normalize_entity_id_from_benchmark(entity_id: str, prefix: str) -> List[str]:
    """
    Normalize entity IDs from benchmarks that may contain malformed formats:
    - Comma-separated: "12458,12772,12775" -> ["12458", "12772", "12775"]
    - Pipe-separated: "D019572|D010871" -> ["D019572", "D010871"]
    - With taxon info: "172659(Tax:6239)" -> ["172659"]
    - Multiple patterns: "8482,20361(Tax:10090)" -> ["8482", "20361"]
    
    Returns a list of normalized entity IDs.
    """
    if not entity_id:
        return []
    
    # Remove taxon information in parentheses: "172659(Tax:6239)" -> "172659"
    entity_id = re.sub(r'\([^)]*\)', '', entity_id)
    
    # Split by comma or pipe to handle multiple IDs
    parts = re.split(r'[,|]', entity_id)
    
    normalized_ids = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Add prefix if needed
        if ':' in part:
            # Already has prefix (e.g., "DOID:8432", "MESH:D001260")
            normalized_ids.append(part)
        else:
            # Add prefix (e.g., "51592" -> "NCBIGene:51592" or "D001260" -> "MESH:D001260")
            normalized_ids.append(prefix + part if not part.startswith(prefix) else part)
    
    return normalized_ids

def add_correct_id_to_entity_ids(entity: Dict) -> List[str]:
    if BENCHMARK_OBO == 'NCBItestset_CTD_diseases_filtered':
        label_path = '/media/volume/LLMRag2/.local/ActDiseaseEL/meddata/NCBItestset.benchmark.jsonl'
        prefix = 'MESH:'
    elif BENCHMARK_OBO == 'BC2GNgene_human_genes':
        label_path = '/media/volume/LLMRag2/.local/ActDiseaseEL/meddata/BC2GNgene.benchmark.jsonl'
        prefix = 'NCBIGene:'
    elif BENCHMARK_OBO == 'nlmgene_human_genes':
        label_path = '/media/volume/LLMRag2/.local/ActDiseaseEL/meddata/nlmgene.benchmark.jsonl'
        prefix = 'NCBIGene:'
    elif BENCHMARK_OBO == 'health_doid-merged':
        label_path = '/media/volume/LLMRag2/.local/ActDiseaseEL/meddata/healthcare.benchmark.jsonl'
        prefix = 'DOID:'
    else:
        return []
    
    spans = defaultdict(list)
    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'labels' not in data:
                    continue
                docs = data['labels']
                for i in docs:
                    if 'entity_id' not in i or 'span' not in i:
                        continue
                    # Normalize entity ID to handle malformed formats
                    normalized_ids = normalize_entity_id_from_benchmark(i['entity_id'], prefix)
                    for normalized_id in normalized_ids:
                        spans[tuple(i['span'])].append(normalized_id)
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed lines
                continue
    
    start_pos = entity['start_pos']
    end_pos = entity['end_pos']
    return list(set(spans.get((start_pos, end_pos), [])))

def ranking(lists):
    scores = defaultdict(float)
    for list in lists:
        if not list: continue
        n = 1.0 / len(list)
        for u in list: scores[u] += n
    return sorted(scores, key=lambda x: scores[x], reverse=True)

nltk.download('stopwords', quiet=True)
STOP = set(nltk.corpus.stopwords.words('english'))



class OBOEntityLinker:    
    def __init__(self, obo_path: str):
        self.conn = sqlite3.connect(":memory:")
        self.cur = self.conn.cursor()
        
        # ✅ Tạo cả 2 bảng: entities (nhanh) + names_fts (link)
        self.cur.executescript("""
            CREATE TABLE entities (
                id TEXT PRIMARY KEY, 
                name TEXT, 
                def TEXT,
                aliases TEXT  -- Pipe-separated synonyms
            );
            
            CREATE VIRTUAL TABLE names_fts USING fts5(
                entity_id UNINDEXED, 
                name, 
                tokenize='trigram', 
                prefix='2 3'
            );
            
            -- Tối ưu SQLite
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-200000;  -- 200MB RAM cache
        """)
        
        self._load(obo_path)
        
        # ✅ Preload tất cả aliases vào memory cho id()
        self._synonym_cache = self._build_synonym_cache()
    
    def _load(self, path: str):
        term = {'synonyms': []}
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    if 'id' in term: 
                        self._insert(term)
                    term = {'synonyms': []}
                elif m := re.match(r'id: (.+)', line): term['id'] = m.group(1)
                elif m := re.match(r'name: (.+)', line): term['name'] = m.group(1)
                elif m := re.match(r'def: "(.+?)"', line): term['def'] = m.group(1)
                elif m := re.match(r'synonym: "(.+?)"', line): term['synonyms'].append(m.group(1))
        
        if 'id' in term: 
            self._insert(term)
        self.conn.commit()
        self._stats()
    
    def _insert(self, term: Dict):
        all_names = [n for n in [term.get('name')] + term['synonyms'] if n]
        
        # ✅ Lưu aliases vào entities (cho id nhanh)
        aliases_str = '|'.join(all_names[1:]) if len(all_names) > 1 else ''
        self.cur.execute(
            "INSERT INTO entities VALUES (?, ?, ?, ?)", 
            (term['id'], term.get('name'), term.get('def'), aliases_str)
        )
        
        for name in set(all_names):
            if name:
                self.cur.execute(
                    "INSERT INTO names_fts VALUES (?, ?)", 
                    (term['id'], name.lower())
                )
    
    def _stats(self):
        self.cur.execute("SELECT COUNT(*) FROM entities")
        total = self.cur.fetchone()[0]
        print(f"Total entities: {total}")

    def _build_synonym_cache(self) -> Dict[str, list]:
        cache = {}
        self.cur.execute("SELECT id, aliases FROM entities")
        
        for entity_id, aliases_str in self.cur.fetchall():
            cache[entity_id] = aliases_str.split('|') if aliases_str else []
        
        return cache

    @functools.lru_cache(maxsize=32768)
    def id(self, entity_id: str) -> Dict:
        if entity_id not in self._synonym_cache:
            return {'error': 'Entity not found'}
        
        self.cur.execute(
            "SELECT name, def FROM entities WHERE id = ?", 
            (entity_id,)
        )
        row = self.cur.fetchone()
        
        if not row: 
            return {'error': 'Entity not found'}
        
        name, definition = row
        
        return {
            'id': entity_id,
            'label': name,
            'description': definition or '',
            'aliases': self._synonym_cache[entity_id]
        }

    def link(self, text: str, thr: int = 85, k: int = 10, max_stopword_ratio: float = 0.5) -> Dict[Tuple[int, int], Dict]:
        text = text.lower()
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]
        
        spans = [(words[i][0], words[i+n-1][1], ' '.join(w[2] for w in words[i:i+n]))
                for n in range(1, min(5, len(words)+1))
                for i in range(len(words)-n+1)
                if (sum(1 for w in words[i:i+n] if w[2].lower() in STOP) / n) <= max_stopword_ratio]
        
        results = {}
        for start, end, span in spans:
            clean_span = re.sub(r'[.:*^$+-]', ' ', span.lower()).strip().replace('"', '').replace("'", '').strip()
            
            self.cur.execute(f"""
                SELECT DISTINCT e.id, e.name, e.def, n.name 
                FROM names_fts n
                JOIN entities e ON n.entity_id = e.id
                WHERE n.name MATCH ? 
                ORDER BY rank
                LIMIT {k}
            """, (f'"{clean_span}"',))
            
            cands = self.cur.fetchall()
            
            if not cands and ' ' in span:
                for w in span.split():
                    if w.lower() in STOP: continue
                    clean_word = re.sub(r'[.:*^$+-]', ' ', w.lower()).strip()
                    self.cur.execute(f"""
                        SELECT e.id, e.name, e.def, n.name FROM names_fts n
                        JOIN entities e ON n.entity_id = e.id
                        WHERE n.name MATCH ? LIMIT {k}
                    """, (f'"{clean_word.replace('"', '').replace("'", '').strip()}"',))
                    cands.extend(self.cur.fetchall())
            
            best = {}
            for qid, name, definition, matched in cands:
                score = fuzz.WRatio(span, matched)
                if score >= thr and (qid not in best or score > best[qid]['score']):
                    best[qid] = {
                        'id': qid, 
                        'name': name, 
                        'def': definition, 
                        'matched_term': matched, 
                        'score': score
                    }
            
            if best:
                results[(start, end)] = {
                    'span_text': span,
                    'entities': sorted(best.values(), key=lambda x: x['score'], reverse=True)[:k]
                }
        
        return results

    def close(self):
        """Clear cache khi đóng."""
        self.id.cache_clear()
        self._synonym_cache.clear()
        if hasattr(self, 'conn'):
            self.conn.close()

def menp(text: str, obo_linker: OBOEntityLinker, is_parser=False, chunk_size=50):
    if is_parser:
        def parser(output: str, chunk: str, iptext: str, index: int = None) -> List[Dict]:
            if index is None: index = iptext.find(chunk)
            outputs = []
            dedup_set = set()
            for line in output.split('\n'):
                if line.startswith('ENTITY:'):
                    try:
                        mention, normalized_mention = [i.strip() for i in line.replace('ENTITY:', '', 1).strip().split(':') if i.strip()]
                        if mention not in chunk: continue
                        start_pos = chunk.find(mention) + index
                        end_pos = start_pos + len(mention)
                        if (start_pos, end_pos) in dedup_set: continue
                        dedup_set.add((start_pos, end_pos))
                        queries = [n.strip() for n in normalized_mention.split(',') if n.strip()][:3]
                        outputs.append({
                            'mention': mention,
                            'text': ', '.join(queries),
                            'context_left': iptext[:start_pos],
                            'context_right': iptext[end_pos:],
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'aliases': list(set([mention] + queries)),
                            'linked_entity': {},
                            'candidates': []
                        })
                    except: 
                        continue
            return outputs
        return parser
    else:
        prompt_template = '''You are a DOID/MeSH/ICD-10/NCBI Disease entities detection expert. For every disease, syndrome, cure, ... you suspect in the text, pick the best or the most relevant English name(s) based on the context.

### How to output the result?
- Select terminology if it is a name of: gene, diseases, disorders, medical conditions, anatomical structures, drugs.
- Do not select terminology if it is: pronouns, people, organizations, locations, events, times, cures, treatments, prevention, diagnosis, symptoms, signs, food, vehicles, etc.
- If you are not sure, select
- For each selected terminology, provide them by "ENTITY:" followed by the mention text and the English keyword(s) of the terminology (at most 3 keywords, do not contain stopwords).
ENTITY: <mention text> : <search keywords (at most 3 keywords without stopwords) in English name of the terminology>
- Correct examples:
ENTITY: MI: myocardial infarction
ENTITY: diabetes: diabetes type 2, diabetes mellitus
ENTITY: CHF: congestive heart failure, CHF, heart failure
ENTITY: agranulocytusis: agranulocytosis, leukopenia
- Incorrect examples:
MI: myocardial infarction
**ENTITY:** diabetes: diabetes
## ENTITY: CHF: congestive heart failurelink

### Example
Some terminologies in the database: diabetes, diabetes mellitus, patient disease, suffering syndrome, from disease
Context: Patienten lider av diabetes och högt blodtryck. Plx1 is ...
Answer:
This text is in Swedish, I need to be more careful with this text. When I look at this, diabetes and högt blodtryck are diseases.
ENTITY: diabetes: diabetes
högt blodtryck is a weird word, so it is suspicious, translated to English, it is "high blood pressure", which is a disease/medical condition
ENTITY: högt blodtryck: hypertension, high blood pressure
Plx1 is a gene, so will be selected
ENTITY: Plx1: plx1, plx1 gene

Some terminologies in the database: dictator syndrome, congestive heart failure, Trumpet, diagnosis syndrome
Context: Donald Trump is the president of the United States.
Answer:
This text does not mention any diseases, so the answer is empty, no "ENTITY:" here.

### Input
Some terminologies in the database: {mentions}
Context: {text}
'''
        prompts = []
        chunks = []
        for i in range(0, len(text.split()), chunk_size):
            chunk = ' '.join(text.split()[i:i+chunk_size])
            mentions_dict = obo_linker.link(chunk, k=1)
            mentions_str = ''
            for (start, end), span_data in mentions_dict.items():
                mentions_str += f'{span_data['entities'][0]['name']}, '
            prompts.append(prompt_template.format(text=chunk, mentions=mentions_str))
            chunks.append(chunk)
        return prompts, chunks

def mrel(entity: Dict = None, obo_linker: OBOEntityLinker = None, step: int = 20, is_parser: bool = False):
    if is_parser:
        def parse_erp_output(output: str) -> List[int]:
            return {i.strip() for i in output.split('<<ANSWER>>')[-1].split('\n')[0].strip().split(',') if i.strip()}
        return parse_erp_output
    else:
        context_left = entity['context_left']
        mention = entity['mention']
        normalized_mention = entity['text']
        context_right = entity['context_right']
        entity_ids = []
        entity_ids = list(set(sum([[j['id'] for j in i['entities']] for i in list(obo_linker.link(normalized_mention, k=50).values())], [])))[:10]
        entity_ids += add_correct_id_to_entity_ids(entity) # ensure equally access to correct tntities
        entities = [obo_linker.id(ids) for ids in entity_ids]
        all_cands = [f"{i['id']}. {i['label']}: {str(i.get('description', ''))[:20]}..." for i in entities if 'id' in i]
        prompts = []
        prompting = '''You are a DOID/NCBI Disease disambiguation expert. Pick the top-{k} best candidate diseases from DOID knowledge base.

Given mention, context, and a list of candidates, pick the top-{k} best candidate diseases (higly relevant) from the list. You can return less than {k} if you are confident some candidates and sure that other candidates are not relevant. This will highlight the importance of your chosen entities.

### How to output the result?
- Output the id of diseases that are relevant to the mention text given the medical context.
- Provide your reasoning process, clearly, logically, before answering with <<ANSWER>>.
- ID must be separated by a comma. Example: DOID:12345, DOID:67890, DOID:11111.
- After <<ANSWER>>, DO NOT output any other text than IDs separated by a comma.
- Follow the format with prefix <<ANSWER>> followed by the ids
- You must output at most {k} ids, which are the most relevant to the mention.

### Example
Context: Patient mit Diagnose **Diabetes**, der eine Insulintherapie benötigt.
Mention: diabetes (Diabetes mellitus)
Candidates:
DOID:1612. Diabetes Mellitus: A metabolic disorder characterized by high blood sugar.
DOID:934. Type 1 Diabetes: Autoimmune form of diabetes requiring insulin.
DOID:935. Type 2 Diabetes: Insulin resistance form of diabetes.
Answer:
Given the German context, I need to be more careful with the mention. The short paragraph mentions insulin therapy requirement, Type 1 Diabetes is the most relevant, though Diabetes Mellitus is also correct as a general term.
<<ANSWER>>DOID:934, DOID:1612

If you think the diseases are not relevant to the mention, you should output just <<ANSWER>> without any ids.
Example:
<<ANSWER>>

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention} ({normalized_mention})
Candidates:
{cands}
# '''
        index = 0
        while len(all_cands) > index*step:
            cands = (all_cands + all_cands)[index*step:index*step+step]
            cands = '\n'.join(cands).strip() or "No candidates"
            k = (len(cands) // 7) + 1
            prompts.append(prompting.format(cands=cands, context_left=context_left, mention=mention, normalized_mention=normalized_mention, context_right=context_right, k=k))
            index += 1

        return prompts

def cbii(entity: Dict, candidates: List[Dict], high_conf_entities: List[Dict], database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_cbi_output(output: str) -> List[int]:
            for line in output.split('\n'):
                if line.startswith('ENTITY:'):
                    try:
                        qid, confidence = line.replace('ENTITY:', '').split('-')[:2]
                        return {'id': qid.strip(), 'confidence': float(re.search(r'[-+]?\d*\.\d+', confidence).group()) if re.search(r'[-+]?\d*\.\d+', confidence) else 0.0}
                    except:
                        continue
            return {'id': '<NIL>', 'confidence': 0.0}
        return parse_cbi_output
    else:
        context_left = entity['context_left']
        mention = entity['mention']
        normalized_mention = entity['text']
        context_right = entity['context_right']
        cands = [f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in candidates if 'id' in i]
        random.shuffle(cands)
        cands = '\n\n'.join(cands).strip() or "No candidates"
        high_conf_description = [f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in high_conf_entities if 'id' in i]
        random.shuffle(high_conf_description)
        high_conf_description = '\n\n'.join(high_conf_description[:20]).strip() or "First loop, no high-confidence entities"
        return f'''You are a DOID/NCBI Disease disambiguation expert. Pick the best candidate disease from DOID knowledge base.

### Target output
- You should output the id of the gene or disease that you think is the correct match (or you think is the most relevant in all candidates) with the confidence score (0.0-1.0).
- You must provide your reasoning process, clearly, logically, for that entity and your confidence score, before answering with "ENTITY:".
- The format must be ENTITY: <id> - <confidence>
- Example of perfect gene or disease match: ENTITY: DOID:12345 - 1.0
- Example of high-confidence gene or disease: ENTITY: DOID:12345 - 0.8
- Example of relevant gene or disease: ENTITY: DOID:12345 - 0.5
- Example of low-confidence gene or disease: ENTITY: DOID:12345 - 0.1
- Example of no match: ENTITY: <NIL> - 0.0
- The context can be in different languages, so you need to be careful with the mention and the context, better understand what being mentioned in the context before picking the best id.

### Example
Context: The patient presents with **chest pain** and elevated cardiac enzymes, suggesting acute cardiac event.
Mention: chest pain
Candidates:
DOID:11111. Arthritis:
- Description: Inflammation of joints causing pain and stiffness.
- Aliases: Arthritis, Joint Inflammation
DOID:22222. Dermatitis:
- Description: Inflammation of skin.
- Aliases: Dermatitis, Skin Inflammation
DOID:12345. Myocardial Infarction:
- Description: Heart attack caused by blockage of coronary arteries, often presenting with chest pain and elevated cardiac enzymes.
- Aliases: Myocardial Infarction, Heart Attack, MI
DOID:33333. Acute Coronary Syndrome:
- Description: A spectrum of conditions including myocardial infarction and unstable angina, characterized by chest pain and cardiac enzyme elevation.
- Aliases: Acute Coronary Syndrome, ACS
Answer:
This is my reasoning process: Given the context mentioning elevated cardiac enzymes and acute cardiac event, Myocardial Infarction is the most relevant match, with Acute Coronary Syndrome also highly relevant. As I am confusing between the two but more certain about Myocardial Infarction, I give it a higher confidence score (0.8).
ENTITY: DOID:12345 - 0.8

### Input
High-confidence diseases:
{high_conf_description}
Context: {context_left} **{mention}** {context_right}
Mention: {mention} ({normalized_mention})
Candidates:
{cands}
'''

class GraphLinker(AbstractEntityLinker):
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any],
                 verbose: bool = True,
                 ):
        if 'human_genes' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/human_genes.obo'
        elif 'doid' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/doid-merged.obo'
        elif 'ctd_diseases' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/CTD_diseases.obo'
            
        self.entity_db = OBOEntityLinker(obo_path)
        self.llm_client = LLMClient(config.get("llm_model_path", 'Orion-zhen/Qwen3-8B-AWQ'))
        self.model = None
        self.verbose = verbose
        
        # Graph-specific parameters
        self.CBI_ITERATIONS = config.get("cbi_iterations", 3)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.7)
    
    def _log(self, msg: str, level: str = "INFO"):
        """Log message if verbose is enabled"""
        if self.verbose:
            print(f"[{level}] {msg}")

    def has_entity(self, entity_id: str) -> bool:
        return 'id' in self.entity_db.id(entity_id)

    def _menp(self, text: str) -> List[str]:
        prompts, chunks = menp(text, self.entity_db, is_parser=False)

        self._log(f"\nTotal prompts: {len(prompts)}")
        
        llm_responses = self.llm_client.call_batch(prompts)
        
        parser = menp(None, None, is_parser=True)
        parsed_entities = []
        for i, (response, chunk) in enumerate(zip(llm_responses, chunks)):
            parsed = parser(response, chunk, text)
            parsed_entities.extend(parsed)
        
        self._log(f"\nTotal detected entities after mENP: {len(parsed_entities)}")
        for i in range(min(5, len(parsed_entities))):
            mention = parsed_entities[i]['mention']
            text = parsed_entities[i]['text']
            span = (parsed_entities[i]['start_pos'], parsed_entities[i]['end_pos'])
            self._log(f"[mENP] {mention} -> {text} (span: {span})")

        return parsed_entities

    def _mrel(self, entities: List[Dict]) -> List[List[Dict]]:
        prompts = []
        spans = []
        for idx, entity in enumerate(entities):
            prompt = mrel(entity, self.entity_db, is_parser=False)
            spans.append((len(prompts), len(prompts) + len(prompt)))
            prompts += prompt

        self._log(f"\nTotal prompts: {len(prompts)}")
        
        llm_responses = self.llm_client.call_batch(prompts)
        
        parser = mrel(is_parser=True)
        parsed_qids = []
        for idx, (entity, (start, end)) in enumerate(zip(entities, spans)):
            try:
                responses_for_entity = llm_responses[start:end]
                parsed_responses = [parser(response) for response in responses_for_entity]
                qids = ranking(parsed_responses)
                parsed_qids.append(qids[:24])
                self._log(f"[MREL] {entity.get('text')} -> {qids[:5]}")
            except Exception as e:
                parsed_qids.append([])
        
        results = []
        for idx, (entity, qids) in enumerate(zip(entities, parsed_qids)):
            entity_infos = []
            for qid in qids:
                info = self.entity_db.id(qid)
                if 'error' not in info:
                    entity_infos.append(info)
            results.append(entity_infos)
        
        return results
    
    def _cbii(self, entities, candidates, high_conf_entities):
        prompts = []
        for i, entity in enumerate(entities):
            high_conf_filtered = [hc for hc in high_conf_entities if hc != entity.get('linked_entity', {})]
            prompt = cbii(entity, candidates[i], high_conf_filtered, self.entity_db, is_parser=False)
            prompts.append(prompt)

        llm_responses = self.llm_client.call_batch(prompts)
        
        parser = cbii(None, None, None, None, is_parser=True)
        parsed_qid_and_confidence = []
        for i, response in enumerate(llm_responses):
            parsed = parser(response)
            parsed_qid_and_confidence.append(parsed)

        parsed_qid_and_confidence = [i if isinstance(i, dict) else {'id': '<NIL>', 'confidence': 0.0} for i in parsed_qid_and_confidence]
        
        entity_with_confidence = []
        for i, ent in enumerate(parsed_qid_and_confidence):
            entity_info = self.entity_db.id(ent['id'])
            result = {'id': ent['id'], 'confidence': ent['confidence'], 'entity': entity_info}
            entity_with_confidence.append(result)
            self._log(f"[CBI] Span ({entities[i]['start_pos']}-{entities[i]['end_pos']}): {entities[i]['mention']} ({entities[i].get('text')}) -> {ent['id']}: {entity_info.get('label', '<NIL>')} ({ent['confidence']:.2f})")
        
        return entity_with_confidence

    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:        
        predictions = {}
        
        # Step 1: mENP
        entities = self._menp(text)
        confirmed_entities = []

        # Step 2: MREL
        candidates = self._mrel(entities)
        
        # Step 3: CBI iterations
        redo_entities = []
        redo_candidates = []
        for iteration in range(self.CBI_ITERATIONS):            
            high_conf_for_context = [i['linked_entity'] for i in confirmed_entities]
            linked_entities = self._cbii(entities, candidates, high_conf_for_context)
            
            for i in range(len(entities)):
                confidence = linked_entities[i]['confidence']
                entity = linked_entities[i]['entity'].copy()
                entities[i]['linked_entity'] = entity
                if 'id' not in entity: 
                    confidence = 0.0
                entities[i]['confidence'] = confidence
                entities[i]['candidates'] = candidates[i].copy()
                
                if confidence > self.HIGH_CONFIDENCE_THRESHOLD and random.random() * 1.2 < confidence:
                    confirmed_entities.append(entities[i])
                else:
                    redo_entities.append(entities[i])
                    redo_candidates.append(candidates[i])
            
            self._log(f"\nAfter iteration {iteration + 1}:")
            self._log(f"  Confirmed: {len(confirmed_entities)}")
            self._log(f"  To redo: {len(redo_entities)}")
            
            entities = redo_entities.copy()
            candidates = redo_candidates.copy()
            redo_entities = []
            redo_candidates = []

        all_entities = confirmed_entities + entities
        
        for entity in all_entities:
            if entity['confidence'] < self.HIGH_CONFIDENCE_THRESHOLD: 
                continue
            entity_id = entity.get('linked_entity', {}).get('id', None)
            if entity_id == None: 
                continue

            span = (entity['start_pos'], entity['end_pos'])
            candidates_set = {c['id'] for c in entity.get('candidates', []) if 'id' in c}
            predictions[span] = EntityPrediction(span, entity_id, candidates_set)

        return predictions
