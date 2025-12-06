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
from elevant.utils.knowledge_base_mapper import UnknownEntity
import sqlite3, re, rapidfuzz.fuzz as fuzz, nltk
import random
from collections import defaultdict
from functools import cache
from elevant.llm_client import LLMClient

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
        self.cur.executescript("""
            CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT, def TEXT);
            CREATE VIRTUAL TABLE names_fts USING fts5(entity_id UNINDEXED, name, tokenize='trigram', prefix='2 3');
        """)
        self._load(obo_path)
    
    def _load(self, path: str):
        term = {'synonyms': []}
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '[Term]':
                    if 'id' in term: self._insert(term)
                    term = {'synonyms': []}
                elif m := re.match(r'id: (.+)', line): term['id'] = m.group(1)
                elif m := re.match(r'name: (.+)', line): term['name'] = m.group(1)
                elif m := re.match(r'def: "(.+?)"', line): term['def'] = m.group(1)
                elif m := re.match(r'synonym: "(.+?)"', line): term['synonyms'].append(m.group(1))
        if 'id' in term: self._insert(term)
        self.conn.commit()
        self._stats()
    
    def _insert(self, term: Dict):
        self.cur.execute("INSERT INTO entities VALUES (?, ?, ?)", (term['id'], term.get('name'), term.get('def')))
        for name in [n for n in [term.get('name')] + term['synonyms'] if n]:
            self.cur.execute("INSERT INTO names_fts VALUES (?, ?)", (term['id'], name.lower()))
    
    def _stats(self):
        self.cur.execute("SELECT COUNT(*), id, name FROM entities LIMIT 1")
        total, top_id, top_name = self.cur.fetchone()
        print(f"✅ Loaded {total} items. Top: {top_id} → {top_name}")

    @cache
    def id(self, entity_id: str) -> Dict:
        self.cur.execute("SELECT name, def FROM entities WHERE id = ?", (entity_id,))
        row = self.cur.fetchone()
        if not row: return {}
        name, definition = row
        
        self.cur.execute("SELECT name FROM names_fts WHERE entity_id = ?", (entity_id,))
        all_names = [r[0] for r in self.cur.fetchall()]
        
        synonyms = [n for n in all_names if n != name]
        
        return {
            'id': entity_id,
            'label': name,
            'description': definition or '',
            'aliases': synonyms
        } or {'error': 'Entity not found'}

    def link(self, text: str, thr: int = 90, k: int = 10, max_stopword_ratio: float = 0.5) -> Dict[Tuple[int, int], Dict]:
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]
        
        spans = [(words[i][0], words[i+n-1][1], ' '.join(w[2] for w in words[i:i+n]))
                for n in range(1, min(5, len(words)+1))
                for i in range(len(words)-n+1)
                if (sum(1 for w in words[i:i+n] if w[2].lower() in STOP) / n) <= max_stopword_ratio]
        
        results = {}
        for start, end, span in spans:
            clean_span = re.sub(r'[.:*^$+-]', ' ', span.lower()).strip()
            self.cur.execute(f"""
                SELECT e.id, e.name, e.def, n.name FROM names_fts n
                JOIN entities e ON n.entity_id = e.id
                WHERE n.name MATCH ? LIMIT {k}
            """, (f'"{clean_span.replace('"', '').replace("'", '').strip()}"',))
            cands = self.cur.fetchall()
            
            # Fallback cho multi-word spans, không cần cho từ đơn
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
                    best[qid] = {'id': qid, 'name': name, 'def': definition, 'matched_term': matched, 'score': score}
            
            if best:
                results[(start, end)] = {
                    'span_text': span,
                    'entities': sorted(best.values(), key=lambda x: x['score'], reverse=True)[:k]
                }
        
        return results

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
                        end_pos = start_pos + len(mention) + index
                        if (start_pos, end_pos) in dedup_set: continue
                        dedup_set.add((start_pos, end_pos))
                        outputs.append({
                            'mention': mention,
                            'text': normalized_mention,
                            'context_left': iptext[:start_pos],
                            'context_right': iptext[end_pos:],
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'aliases': [normalized_mention, mention],
                            'linked_entity': {},
                            'candidates': []
                        })
                    except: 
                        continue
            return outputs
        return parser
    else:
        prompt_template = '''You are a DOID/MeSH/ICD-10 entities detection expert. For every disease, syndrome, cure, ... you suspect in the text, pick the best English name(s).

Output is a list of entities, each entity is a tuple of mention and the normalized mention.

### How to output the result?
- For each completed suspected term, if they are (or likely to be) diseases, disorders, medical conditions, anatomical structures, chemicals, drugs, organisms, or procedures, pick them even you might not be sure or they are historical/obsolete terms. Ensure this process is short and concise. **BEFORE** Each "ENTITY:", you must provide your reasoning process and a type of the entity.
- You must reason carefully BEFORE picking the entity, not AFTER. This means that before each "ENTITY:", you must provide your reasoning process and a type of the entity.
- The answer must be short, concise, and include some terms that are **WRONG** without "ENTITY:" to ensure that you are fair and not hallucinating.
- Correct examples:
ENTITY: MI: myocardial infarction
ENTITY: diabetes: diabetes
ENTITY: CHF: congestive heart failure
- Incorrect examples:
MI: myocardial infarction
**ENTITY:** diabetes: diabetes
## ENTITY: CHF: congestive heart failure

### Example
Naive terms: diabetes, diabetes mellitus, patient disease, suffering syndrome, from disease
Context: Patienten lider av diabetes och högt blodtryck.
Answer:
This text is in Swedish, I need to be more careful with this text. When I look at this, diabetes and högt blodtryck are diseases.
ENTITY: diabetes: diabetes
högt blodtryck is a weird word, so it is suspicious, translated to English, it is "high blood pressure", which is a disease/medical condition
ENTITY: högt blodtryck: hypertension, high blood pressure
Patienten is a person, an ordinary word, not in the biomedical system, so I dont use "ENTITY:" here

Naive terms: myocardial infarction, congestive heart failure, case disease, diagnosis syndrome
Context: **MI** and **CHF** were diagnosed in this case. Type of 000
Answer:
MI (myocardial infarction) is a disease/medical condition
ENTITY: MI: myocardial infarction, heart attack
CHF (congestive heart failure) is also a disease/medical condition
ENTITY: CHF: congestive heart failure, chronic heart failure
diagnosed is a verb, not in the biomedical system, so no "ENTITY:" here
Type of (ordinary term) is a descriptor, not an entity in the biomedical system, so no "ENTITY:" here
000 is weird, but it has no meaning but just a number, so no "ENTITY:" here

Naive terms: dictator syndrome, congestive heart failure, Trumpet, diagnosis syndrome
Context: Donald Trump is the president of the United States.
Answer:
This text does not mention any diseases, so the answer is empty.

### Input
Naive terms: {mentions}
**DO NOT TRUST NAIVE TERMS BLINDLY, THEY CAN BE WRONG MOST OF THE TIME AS THEY USE FUZZY MATCHING, TRUST THE CONTEXT AND YOUR KNOWLEDGE**
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

def mrel(entity: Dict = None, obo_linker: OBOEntityLinker = None, step: int = 24, is_parser: bool = False):
    if is_parser:
        def parse_erp_output(output: str) -> List[int]:
            return {i.strip() for i in output.split('<<ANSWER>>')[-1].split('\n')[0].strip().split(',') if i.strip()}
        return parse_erp_output
    else:
        context_left = entity['context_left']
        mention = entity['mention']
        normalized_mention = entity['text']
        context_right = entity['context_right']
        entity_ids = list(set(sum([[j['id'] for j in i['entities']] for i in list(obo_linker.link(normalized_mention, k=50).values())], [])))[:250]
        entities = [obo_linker.id(ids) for ids in entity_ids]
        all_cands = [f"{i['id']}. {i['label']}: {str(i.get('description', ''))[:20]}..." for i in entities if 'id' in i]
        prompts = []
        prompting = '''You are a DOID disambiguation expert. Pick the best candidate disease from DOID knowledge base.

Input is a disease mention like "myocardial infarction", the surrounding text/context and a long list of possible **candidate diseases** that could match that mention.
The smaller the number of diseases listed by you, the more relevant they are. For example, if you lists 10 diseases, each disease has a score of 1/10.
If you lists 1 disease, that disease has a score of 1.0. The higher the score, the more relevant the disease is.
  *Example:* If the context says "chest pain and elevated cardiac enzymes," it will remove candidates like *Arthritis* (not cardiac-related) or *Dermatitis* (skin condition) and list only *Myocardial Infarction* and *Acute Coronary Syndrome*, each disease has a score of 1/2.
- **Your output:** A shorter, more relevant list of candidate diseases from DOID (e.g., Myocardial Infarction, Acute Coronary Syndrome).

### How to output the result?
- You should output the id of diseases that are relevant to the mention text given the medical context.
- You MUST provide your reasoning process, clearly, logically, before answering with <<ANSWER>>.
- Each id must be separated by a comma.
- Example: DOID:12345, DOID:67890, DOID:11111
- Follow the format with prefix <<ANSWER>> followed by the ids (Example: <<ANSWER>>DOID:12345, DOID:67890, DOID:11111)
- You must output at most 10 ids, which are the most relevant to the mention.

### Example
Context: The patient presents with **chest pain** and elevated cardiac enzymes, suggesting acute cardiac event.
Mention: chest pain (myocardial infarction)
Candidates:
DOID:12345. Myocardial Infarction: A heart attack caused by blockage of coronary arteries.
DOID:67890. Angina Pectoris: Chest pain due to reduced blood flow to heart.
DOID:11111. Arthritis: Inflammation of joints.
DOID:22222. Dermatitis: Inflammation of skin.
DOID:33333. Acute Coronary Syndrome: A spectrum of conditions including MI and unstable angina.
Answer:
This is my reasoning process: Given the context mentioning elevated cardiac enzymes and acute cardiac event, both Myocardial Infarction and Acute Coronary Syndrome are highly relevant, not just Myocardial Infarction alone. Angina is also possible but less likely given the enzyme elevation.
<<ANSWER>>DOID:12345, DOID:33333, DOID:67890

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
            prompts.append(prompting.format(cands=cands, context_left=context_left, mention=mention, normalized_mention=normalized_mention, context_right=context_right))
            index += 1
        
        index = 0
        while len(all_cands) > index*step*2:
            cands = (all_cands + all_cands)[index*step*2:index*step*2+step*2]
            cands = '\n'.join(cands).strip() or "No candidates"
            prompts.append(prompting.format(cands=cands, context_left=context_left, mention=mention, normalized_mention=normalized_mention, context_right=context_right))
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
        return f'''You are a DOID disambiguation expert. Pick the best candidate disease from DOID knowledge base.

### Target output
- You should output the id of the disease that you think is the correct match (or you think is the most relevant in all candidates) with the confidence score (0.0-1.0).
- You must provide your reasoning process, clearly, logically, for that entity and your confidence score, before answering with "ENTITY:".
- The format must be ENTITY: <id> - <confidence>
- Example of perfect disease match: ENTITY: DOID:12345 - 1.0
- Example of high-confidence disease: ENTITY: DOID:12345 - 0.8
- Example of relevant disease: ENTITY: DOID:12345 - 0.5
- Example of low-confidence disease: ENTITY: DOID:12345 - 0.1
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
                 obo_path: str = '/media/volume/LLMRag2/.local/HumanDiseaseOntology/src/ontology/doid-merged.obo',
                 verbose: bool = True,
                 ):
        self.entity_db = OBOEntityLinker(obo_path)
        self.llm_client = LLMClient(config.get("llm_model_path", None))
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
        
        llm_responses = self.llm_client.call_batch(prompts)
        
        parser = menp(None, None, is_parser=True)

        print('--------------------------------<PROMPTS>--------------------------------')
        print(prompts[0])
        print('--------------------------------<LLM RESPONSES>--------------------------------')
        print(llm_responses[0])
        print('--------------------------------<PARSED ENTITIES>--------------------------------')
        print(parser(llm_responses[0], chunks[0], text)[0]['mention'])
        print(parser(llm_responses[0], chunks[0], text)[0]['text'])
        print(parser(llm_responses[0], chunks[0], text)[0]['start_pos'])
        print(parser(llm_responses[0], chunks[0], text)[0]['end_pos'])
        print('--------------------------------<END OF DEBUG>--------------------------------')
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
        for ent in parsed_qid_and_confidence:
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
