"""
### **Step 0: Entity Normalization Processor (ENP)**
- **What it does:** It takes context, mention and return a short normalized mention, example: "Trump" -> "Donald Trump"
- **How it works:** LLM takes the context, mention and return a short normalized mention
- **Result:** A short direct normalized mention that can be used to search for related entities using fuzzy matching, not being mixed with unrelated entities.

---

### **Step 1: Map-Reduce Entity Linker (MREL)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "fennec fox" and the surrounding text/context) and a long list of possible **candidate entities** that could match that mention.
- **How it works:** It separates the long list of candidates into many smaller lists, for each list, Select the most relevant entities based on the context. The smaller the number of entities listed by Agent, the more relevant they are. For example, if the Agent lists 10 entities, each entity has a score of 1/10.
While if the Agent lists 1 entity, each entity has a score of 1.0. The higher the score, the more relevant the entity is.
  *Example:* If the context says "living in desert regions of North Africa," it will remove candidates like *Arctic Fox* (lives in the cold) or fictional characters like *Nick Wilde* and list only *Fennec Fox* and *Desert Fox*, each entity has a score of 1/2.
- **Result:** A shorter, more relevant list of candidate entities (e.g., Red Fox, Rippell’s Fox).

---

### **Step 2: Confidence-based Improvement (CBI)**
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
1. **EMRP** → Reduces a big list to a few possible options.  
2. **CBI** → Repeat the linking process for low-confidence entities and double check high-confidence entities, while the information of linked entities (high-confidence) is provided in the context to enhance LLM in understanding overall context of all entities.

This helps computers understand *which real-world thing* a word refers to, especially when words can mean different things in different situations.
"""
import logging
import random
import spacy
from typing import Dict, Tuple, Optional, Any, List
from spacy.tokens import Doc
from elevant.llm_client import LLMClient
from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.settings import LARGE_MODEL_NAME, NER_IGNORE_TAGS
from functools import cache
import requests
import re
from collections import defaultdict

def ranking(lists):
    scores = defaultdict(float)
    for list in lists:
        if not list: continue
        n = 1.0 / len(list)
        for u in list: scores[u] += n
    return sorted(scores, key=lambda x: scores[x], reverse=True)

logger = logging.getLogger("main." + __name__.split(".")[-1])

@cache
def get_entity_info(qid: str) -> dict:
    """Get English entity info from Wikidata ID"""
    qid = qid.strip().upper()
    try:
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "format": "json",
                "languages": "en",
                "props": "labels|descriptions|aliases|claims"
            },
            headers={"User-Agent": "WikidataBot/1.0"},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        
        if "entities" not in data or qid not in data["entities"]:
            logger.warning(f"[GET_ENTITY_INFO] Entity {qid} not found")
            return {"error": "Entity not found"}
        
        entity = data["entities"][qid]
        if "missing" in entity:
            logger.warning(f"[GET_ENTITY_INFO] Entity {qid} is missing")
            return {"error": "Entity is missing"}
        
        return {
            "id": entity["id"],
            "label": entity.get("labels", {}).get("en", {}).get("value", ""),
            "description": entity.get("descriptions", {}).get("en", {}).get("value", ""),
            "aliases": [a["value"] for a in entity.get("aliases", {}).get("en", [])],
            "instance_of": [
                claim["mainsnak"]["datavalue"]["value"]["id"]
                for claim in entity.get("claims", {}).get("P31", [])
                if "datavalue" in claim.get("mainsnak", {})
            ]
        }
    except Exception as e:
        return {"error": str(e)}

def enp(entity: Dict = None, is_parser: bool = False):
    if is_parser:
        def parse_enp_output(output: str) -> str:
            return output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
        return parse_enp_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
    return f'''### **Step 0: Entity Normalization Processor (ENP)**
- **What it does:** It takes context, mention and return a short normalized mention, example: "Trump" -> "Donald Trump"
- **How it works:** LLM takes the context, mention and return a short normalized mention
- **Result:** A short direct normalized mention that can be used to search for related entities using fuzzy matching, not being mixed with unrelated entities.

### How to output the result?
- You should output the normalized mention that can be used to search for related entities using fuzzy matching, not being mixed with unrelated entities.
- Use your knowledge to determine the most relevant normalized mention, what entity is mentioned in the context and which way to write them correctly.
- If you don't know the correct normalized mention, you should output <<ANSWER>> with the original mention without modification (or with minor modification to correct the spelling).
- Follow the format with prefix <<ANSWER>> followed by the normalized mention (Example: <<ANSWER>>Donald Trump)

### Example
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Answer: <<ANSWER>>Fennec Fox

Context: **Steve** invented the Apple computer.
Mention: Steve
Answer:
They are talking about Steve Jobs.
<<ANSWER>>Steve Jobs

Context: **Dolnal-trump** is a president of the United States.
Mention: Dolnal-trump
Answer:
They are talking about Donald Trump.
<<ANSWER>>Donald Trump

Context: **hr234sdfn4353** in the document presents ...
Mention: hr234sdfn4353
Answer:
I dont know what hr234sdfn4353 is given the context, so I will repeat the original mention.
<<ANSWER>>hr234sdfn4353

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
'''

def mrel(entity: Dict = None, database: EntityDatabase = None, step: int = 24, is_parser: bool = False):
    if is_parser:
        def parse_erp_output(output: str) -> List[int]:
            return {i.strip() for i in output.split('<<ANSWER>>')[-1].split('\n')[0].strip().split(',') if i.strip()}
        return parse_erp_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        entity_ids = list(database.get_candidates(entity['text']))
        entities = [get_entity_info(ids) for ids in entity_ids]
        all_cands = [f"{i['id']}. {i['label']}: {i['description'][:20]}..." for i in entities if 'id' in i]
        prompts = []
        prompting = '''### **Step 1: Map-Reduce Entity Linker (MREL)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "fennec fox" and the surrounding text/context) and a long list of possible **candidate entities** that could match that mention.
- **How it works:** It separates the long list of candidates into many smaller lists, for each list, Select the most relevant entities based on the context. The smaller the number of entities listed by Agent, the more relevant they are. For example, if the Agent lists 10 entities, each entity has a score of 1/10.
While if the Agent lists 1 entity, each entity has a score of 1.0. The higher the score, the more relevant the entity is.
  *Example:* If the context says "living in desert regions of North Africa," it will remove candidates like *Arctic Fox* (lives in the cold) or fictional characters like *Nick Wilde* and list only *Fennec Fox* and *Desert Fox*, each entity has a score of 1/2.
- **Result:** A shorter, more relevant list of candidate entities (e.g., Red Fox, Rippell’s Fox).

### How to output the result?
- You should output the id of entities that are relevant to the mention text given the context.
- Each id must be separated by a comma.
- Example: Q023454, Q320004, Q32454, Q1234
- Follow the format with prefix <<ANSWER>> followed by the ids (Example: <<ANSWER>>Q023454, Q320004, Q32454, Q1234)
- You must output at most 10 ids, which are the most relevant to the mention.

### Example
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Candidates:
Q32454. Arctic Fox: A fox that lives in the Arctic regions.
Q1234. Red Fox: A fox with red fur.
Q654767. Rippell's Fox: A fox that lives in the Rippell's regions.
Q123456. Nick Wilde: A character in the movie "Harari".
Q023454. Fennec Fox: A fox that lives in the Fennec regions.
Q320004. Desert Fox: A fox that lives in the Sahara Desert.
Answer:
Even though Fennec Fox is the most relevant entity, I cannot ignore possibilities of Desert Fox. As a result, relevant entities are Q023454, Q320004
<<ANSWER>>Q023454, Q320004

Context: Steve invented the Apple computer.
Mention: Steve Jobs
Candidates:
Q19837. Steve Jobs: Inventor of the Apple computer.
Q19838. Steve Wozniak: Inventor of the Apple computer.
Q19839. Steve: A character in the movie "Harari".
Answer:
Given the context and knowing that Steve Jobs is the inventor of the Apple computer, I believe the correct match is Q19837, do not need to consider other entities.
<<ANSWER>>Q19837

If you think the entities are not relevant to the mention, you should output just <<ANSWER>> without any ids.
Example:
<<ANSWER>>

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''
        index = 0
        while len(all_cands) > index*step:
            cands = (all_cands + all_cands)[index*step:index*step+step]
            cands = '\n'.join(cands).strip() or "No candidates"
            prompts.append(prompting.format(cands=cands, context_left=context_left, mention=mention, context_right=context_right))
            index += 1
        
        index = 0
        while len(all_cands) > index*step*2:
            cands = (all_cands + all_cands)[index*step*2:index*step*2+step*2]
            cands = '\n'.join(cands).strip() or "No candidates"
            prompts.append(prompting.format(cands=cands, context_left=context_left, mention=mention, context_right=context_right))
            index += 1

        return prompts

def cbii(entity: Dict, candidates: List[Dict], high_conf_entities: List[Dict], database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_cbi_output(output: str) -> List[int]:
            for line in output.split('\n'):
                if line.startswith('ENTITY:'):
                    try:
                        qid, confidence = line.replace('ENTITY:', '').split(':')[:2]
                        return {'id': qid, 'confidence': float(re.search(r'[-+]?\d*\.\d+', confidence).group()) if re.search(r'[-+]?\d*\.\d+', confidence) else 0.0}
                    except Exception as e:
                        continue
            return {'id': 'Not found', 'confidence': 0.0}
        return parse_cbi_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        cands = [f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in candidates if 'id' in i]
        random.shuffle(cands)
        cands = '\n\n'.join(cands).strip() or "No candidates"
        high_conf_description = [f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in high_conf_entities]
        random.shuffle(high_conf_description)
        high_conf_description = '\n\n'.join(high_conf_description).strip() or "First loop, no high-confidence entities"
        return f'''## **Step 2: Confidence-based Improvement (CBI)**
- **What it does:** Repeat the linking process for low-confidence entities, while the information of linked entities (high-confidence) is provided in the context to enhance LLM in understanding overall context of all entities.
- **How it works:**
  - Repeat k times
    - Link all detected entities and return a confidence score for each entity
    - If the confidence score (a%) is above a threshold, return the entity information and have a% of puting it in the entity return list, 1-a% change the high-confidence entity is put to the re-processing list (double check with context from other high-confidence entities)
    - If the confidence score is below a threshold, add the entity to the re-processing list, enhance its context by adding the information of linked entities (high-confidence) to the context (what is that entity, description, aliases, ...)
    - After k times, only return entities with high confidence, entities with low confidence are ignored

### How to output the result?
- You should output the id of the entity that you think is the correct match (or you think is the most relevant in all candidates) with the confidence score (0.0-1.0).
- Example of perfect entity: ENTITY: Q023454: 1.0
- Example of high-confidence entity: ENTITY: Q023454: 0.8
- Example of relevant entity: ENTITY: Q023454: 0.5
- Example of low-confidence entity: ENTITY: Q023454: 0.1
- Example of no match: ENTITY: <NIL>: 0.0

### Example
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Candidates:
Q32454. Arctic Fox:
- Description: A fox that lives in the Arctic regions.
- Aliases: Arctic Fox, Arctic Foxes
Q1234. Red Fox:
- Description:
- Aliases:
Q654767. Rippell's Fox:
- Description: A fox that lives in the Rippell's regions.
- Aliases: Rippell's Fox, Rippell's Foxes
Q023454. Fennec Fox:
- Description: A fox that lives in the Fennec regions.
- Aliases: Fennec Fox, Fennec Foxes
Answer:
I believe the correct match is Q023454
ENTITY: Q023454: 0.9

### Input
High-confidence entities:
{high_conf_description}
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
'''


class GraphLinker(AbstractEntityLinker):
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        self.entity_db = entity_database
        model_path = config.get("llm_model_path", "Orion-zhen/Qwen3-8B-AWQ")
        self.llm_client = LLMClient(model_path)
        self.model = spacy.load(LARGE_MODEL_NAME, disable=["lemmatizer"])
        self.entity_db.load_entity_names()
        self.entity_db.load_alias_to_entities()
        self.entity_db.load_hyperlink_to_most_popular_candidates()
        self.entity_db.load_sitelink_counts()

        # Graph-specific parameters
        self.CBI_ITERATIONS = config.get("cbi_iterations", 4)
        self.HIGH_CONFIDENCE_THRESHOLD = config.get("high_confidence_threshold", 0.9)
        
    def has_entity(self, entity_id: str) -> bool:
        return self.entity_db.contains_entity(entity_id)
    
    def _detect_entities_with_spacy(self, text: str, doc: Optional[Doc] = None) -> List[Dict]:
        logger.info(f"[NER] Starting entity detection for text length: {len(text)}")
        if doc is None: doc = self.model(text)
        entity_spans = [{
            'text': ent.text,
            'start_pos': ent.start_char,
            'end_pos': ent.end_char,
        } for ent in doc.ents if ent.label_ not in NER_IGNORE_TAGS]

        logger.info(f"[NER] Detected {len(entity_spans)} entity spans: {[e['text'] for e in entity_spans[:5]]}")

        entities = []
        for span_info in entity_spans:
            entities.append({
                'text': span_info['text'],
                'start_pos': span_info['start_pos'],
                'end_pos': span_info['end_pos'],
                'context_left': text[:span_info['start_pos']],
                'context_right': text[span_info['end_pos']:],
                'aliases': [span_info['text']],
                'linked_entity': {},
                'candidates': []
            })

        logger.info(f"[NER] Built {len(entities)} entity dicts")
        return entities
    
    def _enp(self, entities: List[Dict]) -> List[str]:
        logger.info(f"[ENP] Starting ENP for {len(entities)} entities")
        prompts = [enp(entity, is_parser=False) for entity in entities]
        parser = enp(is_parser=True)
        parsed_entities = [parser(response) for response in self.llm_client.call_batch(prompts)]
        logger.info(f"[ENP] Completed: {len(parsed_entities)} entities with normalized mention")
        return [{**entity, 'text': normalized_mention} for entity, normalized_mention in zip(entities, parsed_entities)]

    def _mrel(self, entities: List[Dict]) -> List[List[Dict]]:
        logger.info(f"[EMRP] Starting EMRP for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        spans = []
        for idx, entity in enumerate(entities):
            prompt = mrel(entity, self.entity_db, is_parser=False)
            spans.append((len(prompts), len(prompts) + len(prompt)))
            prompts += prompt
        
        # Call LLM
        logger.info(f"[EMRP] Calling LLM with {len(prompts)} prompts")
        llm_responses = self.llm_client.call_batch(prompts)
        logger.info(f"[EMRP] Received {len(llm_responses)} responses")
        
        # Log sample responses
        for idx, (entity, response) in enumerate(zip(entities[:3], llm_responses[:3])):
            logger.debug(f"[EMRP] Entity '{entity['text']}' response ({len(response)} chars):\n{response[:300]}...")
        
        # Parse responses
        parser = mrel(is_parser=True)
        parsed_qids = []
        for idx, (entity, (start, end)) in enumerate(zip(entities, spans)):
            try:
                qids = ranking([parser(response) for response in llm_responses[start:end]])
                parsed_qids.append(qids[:24])
                logger.debug(f"[EMRP] Entity '{entity['text']}': Parsed {len(qids)} QIDs: {qids[:5]}")
            except Exception as e:
                logger.warning(f"[EMRP] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[EMRP] Raw response: {response[:200]}")
                parsed_qids.append([])
        
        # Get entity info from Wikidata
        logger.info(f"[EMRP] Fetching entity info from Wikidata for {sum(len(qids) for qids in parsed_qids)} QIDs")
        results = []
        for idx, (entity, qids) in enumerate(zip(entities, parsed_qids)):
            entity_infos = []
            for qid in qids:
                info = get_entity_info(qid)
                if 'error' not in info:
                    entity_infos.append(info)
                else:
                    logger.debug(f"[EMRP] Entity '{entity['text']}': QID {qid} error: {info.get('error')}")
            results.append(entity_infos)
            logger.debug(f"[EMRP] Entity '{entity['text']}': Got {len(entity_infos)} valid entity infos")
        
        logger.info(f"[EMRP] Completed: {[len(r) for r in results]} candidates per entity")
        return results
    
    def _cbii(self, entities, candidates, high_conf_entities):
        logger.info(f"[CBI] Starting CBI for {len(entities)} entities")
        prompts = [cbii(entity, candidates[i], [i for i in high_conf_entities if i != entity['linked_entity']], self.entity_db, is_parser=False) for i, entity in enumerate(entities)]
        parser = cbii(None, None, None, None, is_parser=True)
        parsed_qid_and_confidence = [parser(response) for response in self.llm_client.call_batch(prompts)]
        parsed_qid_and_confidence = [i if isinstance(i, dict) else {'id': '<NIL>', 'confidence': 0.0} for i in parsed_qid_and_confidence]
        entity_with_confidence = [{'id': ent['id'], 'confidence': ent['confidence'], 'entity': get_entity_info(ent['id'])} for ent in parsed_qid_and_confidence]
        logger.info(f"[CBI] Completed: {len(entity_with_confidence)} entities with confidence")
        return entity_with_confidence

    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """Predict entities using EMRP + CBI (Confidence-based Improvement)"""
        predictions = {}
        
        entities = self._detect_entities_with_spacy(text, doc)
        entities = self._enp(entities)
        confirmed_entities = []

        candidates = self._mrel(entities)
        redo_entities = []
        redo_candidates = []
        for _ in range(self.CBI_ITERATIONS):
            linked_entities = self._cbii(entities, candidates, [i['linked_entity'] for i in confirmed_entities])
            # if len([i for i in linked_entities if i['confidence'] > self.HIGH_CONFIDENCE_THRESHOLD]) == 0: break
            for i in range(len(entities)):
                confidence = linked_entities[i]['confidence']
                entity = linked_entities[i]['entity'].copy()
                entities[i]['linked_entity'] = entity
                if 'id' not in entity: confidence = 0.0 # empty entity or error entity -> redo
                entities[i]['confidence'] = confidence
                entities[i]['candidates'] = candidates[i].copy()
                if confidence > self.HIGH_CONFIDENCE_THRESHOLD and random.random() * 1.2 < confidence:
                    confirmed_entities.append(entities[i])
                else:
                    redo_entities.append(entities[i])
                    redo_candidates.append(candidates[i])
            entities = redo_entities.copy()
            candidates = redo_candidates.copy()
            redo_entities = []
            redo_candidates = []

        for entity in confirmed_entities + entities:
            if entity['confidence'] < self.HIGH_CONFIDENCE_THRESHOLD: continue
            entity_id = entity.get('linked_entity', {}).get('id', None)
            if entity_id == None: continue

            span = (entity['start_pos'], entity['end_pos'])
            candidates = {c['id'] for c in entity.get('candidates', [])}
            predictions[span] = EntityPrediction(span, entity_id, candidates)

        return predictions
