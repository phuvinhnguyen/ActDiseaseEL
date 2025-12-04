"""
### **Step 1: Entity Reduction Processor (ERP)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "fennec fox" and the surrounding text/context) and a long list of possible **candidate entities** that could match that mention.
- **How it works:** It filters out candidates that are obviously wrong based on the context.  
  *Example:* If the context says "living in desert regions of North Africa," it will remove candidates like *Arctic Fox* (lives in the cold) or fictional characters like *Nick Wilde*.  
- **Result:** A shorter, more relevant list of candidate entities (e.g., Red Fox, Rippell’s Fox).

---

### **Step 2: Dual-perspective Entity Linker (DEL)**
- **What it does:** This step takes the filtered candidates and tries to decide which one is the *correct* match using **two different perspectives**.

  1. **Prior Entity Linker:**  
     - Asks: *"Based on common knowledge, which entity do people usually mean when they say 'fennec fox'?"*  
     - Uses general world knowledge (like Wikipedia or common sense) to pick the most likely candidate.

  2. **Contextual Entity Linker:**  
     - Asks: *"Given the specific context provided, which entity fits best here?"*  
     - Carefully reads the surrounding text to make a decision.

- **How it works:** The system runs both linkers, compares their answers, and decides on the best candidate.

---

### **Step 3: Entity Consensus Judger (ECJ)**
- **What it does:** This is the final check to make sure the chosen entity is consistent and correct.
- **How it works:**
  - Takes the top candidate(s) from the previous step.
  - Uses a **Large Language Model (LLM)** to double-check if the choice makes sense with the context.
  - The **Consistency Algorithm** ensures the explanation is logical.
  - If everything agrees, the **Merge Module** produces the final answer.

  *Example:*  
  The LLM might explain: *"Rippell’s fox is correct because it lives in deserts in North Africa, which matches the context."*

---

### **Final Output:**
- The system outputs the **correctly linked entity** (e.g., *Rippell’s fox*) with a clear explanation.

---

### **In Simple Summary:**
1. **ERP** → Reduces a big list to a few possible options.  
2. **DEL** → Looks at the problem from two angles (common knowledge + specific context) to pick one.  
3. **ECJ** → Double-checks the choice using an LLM to ensure it’s correct and consistent.  

This helps computers understand *which real-world thing* a word refers to, especially when words can mean different things in different situations.
"""
import logging
import re
import random
import spacy
import json
from contextlib import contextmanager
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any, List

from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.utils.knowledge_base_mapper import UnknownEntity
from elevant.settings import LARGE_MODEL_NAME, NER_IGNORE_TAGS
from elevant.llm_client import LLMClient
import requests
import json
from functools import cache

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

def erp(entity: Dict, database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_erp_output(output: str) -> List[int]:
            return [i.strip() for i in output.split('<<ANSWER>>')[-1].split('\n')[0].strip().split(',')]
        return parse_erp_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        entity_ids = list(database.get_candidates(entity['text']))
        entities = [{ 'id': ids, 'label': database.get_entity_name(ids)} for ids in entity_ids]
        cands = '\n'.join([f"{i['id']}. {i['label']}" for i in entities[:20]])
        return f'''## **Step 1: Entity Reduction Processor (ERP)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "fennec fox" and the surrounding text/context) and a long list of possible **candidate entities** that could match that mention.
- **How it works:** It filters out candidates that are obviously wrong based on the context.  
  *Example:* If the context says "living in desert regions of North Africa," it will remove candidates like *Arctic Fox* (lives in the cold) or fictional characters like *Nick Wilde*.  
- **Result:** A shorter, more relevant list of candidate entities (e.g., Red Fox, Rippell’s Fox).

### How to output the result?
- You should output the id of entities that are relevant to the mention text given the context.
- Each id must be separated by a comma.
- Example: Q023454, Q320004, Q32454, Q1234

### Example
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Candidates:
Q32454. Arctic Fox: A fox that lives in the Arctic regions.
Q1234. Red Fox:
Q654767. Rippell's Fox: A fox that lives in the Rippell's regions.
Q123456. Nick Wilde: A fox that lives in the Nick Wilde regions.
Q023454. Fennec Fox: A fox that lives in the Fennec regions.
Q320004. Desert Fox:
Answer:
I believe the relevant entities are Q023454, Q320004, Q1234. Artic Fox does not live in the desert regions of North Africa.
<<ANSWER>>Q023454, Q320004, Q32454, Q1234

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

def del_prior_entity_linker(entity: Dict, candidates: List[Dict], database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_del_prior_entity_linker_output(output: str) -> List[int]:
            return output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
        return parse_del_prior_entity_linker_output
    else:
        mention = entity['text']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in candidates[:10]])
        return f'''### **Step 2: Dual-perspective Entity Linker (DEL)**
- **What it does:** This step takes the filtered candidates and tries to decide which one is the *correct* match using **two different perspectives**.

Your perspective is:
- **Prior Entity Linker:**  
  - Asks: *"Based on common knowledge, which entity do people usually mean when they say the mention text?"*  
  - Uses general world knowledge (like Wikipedia or common sense) to pick the most likely candidate.

### How to output the result?
- You should output the id of the entity that is the correct match.
- Example: Q023454

### Example
Mention: fennec fox
Candidates:
Q1234. Red Fox:
- Description: A fox that lives in the Red regions.
- Aliases: Red Fox, Red Foxes
Q654767. Rippell's Fox:
- Description: A fox that lives in the Rippell's regions.
- Aliases: Rippell's Fox, Rippell's Foxes
Q023454. Fennec Fox:
- Description: A fox that lives in the Fennec desert regions.
- Aliases: Fennec Fox, Fennec Foxes
Answer:
I believe the relevant entity is Q023454.
<<ANSWER>>Q023454

### Input
Mention: {mention}
Candidates:
{cands}
# '''

def del_contextual_entity_linker(entity: Dict, candidates: List[Dict], database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_del_contextual_entity_linker_output(output: str) -> List[int]:
            return output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
        return parse_del_contextual_entity_linker_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in candidates[:10]])
        return f'''### **Step 2: Dual-perspective Entity Linker (DEL)**
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Candidates:
Q1234. Red Fox:
- Description: A fox that lives in the Red regions.
- Aliases: Red Fox, Red Foxes
Q654767. Rippell's Fox:
- Description: A fox that lives in the Rippell's regions.
- Aliases: Rippell's Fox, Rippell's Foxes
Q023454. Fennec Fox:
- Description: A fox that lives in the Fennec desert regions.
- Aliases: Fennec Fox, Fennec Foxes
Answer:
I believe the relevant entity is Q023454.
<<ANSWER>>Q023454

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

def ecj(entity: Dict, candidates: List[Dict], database: EntityDatabase, is_parser: bool = False):
    if is_parser:
        def parse_ecj_output(output: str) -> List[int]:
            return output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
        return parse_ecj_output
    else:
        if candidates[0]['id'] == candidates[1]['id']:
            return f'''Return <<ANSWER>>{candidates[0]['id']} in your answer without anything else. Just copy paste the input.
## Example
Input: <<ANSWER>>Q1234
Output: <<ANSWER>>Q1234

Input: <<ANSWER>>Q4452365
Output: <<ANSWER>>Q4452365

Input: <<ANSWER>>Q0345646
Output: <<ANSWER>>Q0345646

Input: <<ANSWER>>ERT8030452
Output: <<ANSWER>>ERT8030452

Input: <<ANSWER>>Q023454
Output: <<ANSWER>>Q023454

## Your input:
Input: <<ANSWER>>{candidates[0]['id']}
Output:
'''
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i['description']}\n- Aliases: {', '.join(i['aliases'])}" for i in candidates[:20]])
        return f'''### **Step 3: Entity Consensus Judger (ECJ)**
- **What it does:** This is the final check to make sure the chosen entity is consistent and correct.
- **How it works:**
- Takes the top candidate(s) from the previous step.
- Uses a **Large Language Model (LLM)** to double-check if the choice makes sense with the context.
- The **Consistency Algorithm** ensures the explanation is logical.
- If everything agrees, the **Merge Module** produces the final answer.

- **Result:** The final answer is the entity that is the correct match.

### How to output the result?
- You should output the id of the entity that is the correct match.
- Example: Q023454

### Example
Context: The **fennec fox** is a small fox that lives in the desert regions of North Africa.
Mention: fennec fox
Candidates:
Q654767. Rippell's Fox:
- Description: A fox that lives in the Rippell's regions.
- Aliases: Rippell's Fox, Rippell's Foxes
Q023454. Fennec Fox:
- Description: A fox that lives in the Fennec desert regions.
- Aliases: Fennec Fox, Fennec Foxes
Answer:
I believe the relevant entity is Q023454.
<<ANSWER>>Q023454

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

class OneNetLinker(AbstractEntityLinker):    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any]):
        logger.info("[INIT] Initializing OneNetLinker")
        self.entity_db = entity_database        
        model_path = config.get("llm_model_path", "Orion-zhen/Qwen3-8B-AWQ")
        use_4bit = config.get("use_4bit", True)
        logger.info(f"[INIT] LLM model: {model_path}, use_4bit: {use_4bit}")
        self.llm_client = LLMClient(model_path, use_4bit=use_4bit)
        logger.info(f"[INIT] Loading spaCy model: {LARGE_MODEL_NAME}")
        self.model = spacy.load(LARGE_MODEL_NAME, disable=["lemmatizer"])
        logger.info("[INIT] Loading entity database...")
        self.entity_db.load_entity_names()
        self.entity_db.load_alias_to_entities()
        self.entity_db.load_hyperlink_to_most_popular_candidates()
        self.entity_db.load_sitelink_counts()
        logger.info("[INIT] Entity database loaded")

        self.top_k = config.get("top_k", 5)
        self.shuffle_candidates = config.get("shuffle_candidates", True)
        logger.info(f"[INIT] Configuration: top_k={self.top_k}, shuffle_candidates={self.shuffle_candidates}")
        logger.info("[INIT] OneNetLinker initialized successfully")
        
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
                'link_entities': {},
                'candidates': []
            })

        logger.info(f"[NER] Built {len(entities)} entity dicts")
        return entities
    
    def _erp(self, entities: List[Dict]) -> List[List[Dict]]:
        logger.info(f"[ERP] Starting ERP for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            prompt = erp(entity, self.entity_db, is_parser=False)
            prompts.append(prompt)
        
        # Call LLM
        logger.info(f"[ERP] Calling LLM with {len(prompts)} prompts")
        llm_responses = self.llm_client.call_batch(prompts)
        logger.info(f"[ERP] Received {len(llm_responses)} responses")
        
        # Log sample responses
        for idx, (entity, response) in enumerate(zip(entities[:3], llm_responses[:3])):
            logger.debug(f"[ERP] Entity '{entity['text']}' response ({len(response)} chars):\n{response[:300]}...")
        
        # Parse responses
        parser = erp(None, None, is_parser=True)
        parsed_qids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                qids = parser(response)
                parsed_qids.append(qids)
                logger.debug(f"[ERP] Entity '{entity['text']}': Parsed {len(qids)} QIDs: {qids[:5]}")
            except Exception as e:
                logger.warning(f"[ERP] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[ERP] Raw response: {response[:200]}")
                parsed_qids.append([])
        
        # Get entity info from Wikidata
        logger.info(f"[ERP] Fetching entity info from Wikidata for {sum(len(qids) for qids in parsed_qids)} QIDs")
        results = []
        for idx, (entity, qids) in enumerate(zip(entities, parsed_qids)):
            entity_infos = []
            for qid in qids:
                info = get_entity_info(qid)
                if 'error' not in info:
                    entity_infos.append(info)
                else:
                    logger.debug(f"[ERP] Entity '{entity['text']}': QID {qid} error: {info.get('error')}")
            results.append(entity_infos)
            logger.debug(f"[ERP] Entity '{entity['text']}': Got {len(entity_infos)} valid entity infos")
        
        logger.info(f"[ERP] Completed: {[len(r) for r in results]} candidates per entity")
        return results
    
    def _del_prior_entity_linker(self, entities: List[Dict], candidates: List[List[Dict]]) -> str:
        logger.info(f"[DEL-PRIOR] Starting Prior Entity Linker for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            cands = candidates[idx] if idx < len(candidates) else []
            prompt = del_prior_entity_linker(entity, cands, self.entity_db, is_parser=False)
            prompts.append(prompt)
            logger.debug(f"[DEL-PRIOR] Entity {idx+1}/{len(entities)} '{entity['text']}': {len(cands)} candidates, prompt ({len(prompt)} chars)")
            if idx == 0:  # Log first prompt as sample
                logger.debug(f"[DEL-PRIOR] Sample prompt (first entity):\n{prompt[:500]}...")
        
        # Call LLM
        logger.info(f"[DEL-PRIOR] Calling LLM with {len(prompts)} prompts")
        llm_responses = self.llm_client.call_batch(prompts)
        logger.info(f"[DEL-PRIOR] Received {len(llm_responses)} responses")
        
        # Log sample responses
        for idx, (entity, response) in enumerate(zip(entities[:3], llm_responses[:3])):
            logger.debug(f"[DEL-PRIOR] Entity '{entity['text']}' response ({len(response)} chars):\n{response[:300]}...")
        
        # Parse responses
        parser = del_prior_entity_linker(None, None, None, is_parser=True)
        parsed_qids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                qid = parser(response)
                parsed_qids.append(qid)
                logger.debug(f"[DEL-PRIOR] Entity '{entity['text']}': Parsed QID: {qid}")
            except Exception as e:
                logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[DEL-PRIOR] Raw response: {response[:200]}")
                parsed_qids.append(None)
        
        # Get entity info from Wikidata
        logger.info(f"[DEL-PRIOR] Fetching entity info from Wikidata for {len([q for q in parsed_qids if q])} QIDs")
        results = []
        for idx, (entity, qid) in enumerate(zip(entities, parsed_qids)):
            if qid:
                info = get_entity_info(qid)
                if 'error' not in info:
                    results.append(info)
                    logger.debug(f"[DEL-PRIOR] Entity '{entity['text']}': Got entity info for {qid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': QID {qid} error: {info.get('error')}")
                    results.append({'id': qid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': No QID parsed")
                results.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
        
        logger.info(f"[DEL-PRIOR] Completed: {len([r for r in results if r.get('id')])} entities linked")
        return results

    def _del_contextual_entity_linker(self, entities: List[Dict], candidates: List[List[Dict]]) -> str:
        logger.info(f"[DEL-CONTEXT] Starting Contextual Entity Linker for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            cands = candidates[idx] if idx < len(candidates) else []
            prompt = del_contextual_entity_linker(entity, cands, self.entity_db, is_parser=False)
            prompts.append(prompt)
            logger.debug(f"[DEL-CONTEXT] Entity {idx+1}/{len(entities)} '{entity['text']}': {len(cands)} candidates, prompt ({len(prompt)} chars)")
            if idx == 0:  # Log first prompt as sample
                logger.debug(f"[DEL-CONTEXT] Sample prompt (first entity):\n{prompt[:500]}...")
        
        # Call LLM
        logger.info(f"[DEL-CONTEXT] Calling LLM with {len(prompts)} prompts")
        llm_responses = self.llm_client.call_batch(prompts)
        logger.info(f"[DEL-CONTEXT] Received {len(llm_responses)} responses")
        
        # Log sample responses
        for idx, (entity, response) in enumerate(zip(entities[:3], llm_responses[:3])):
            logger.debug(f"[DEL-CONTEXT] Entity '{entity['text']}' response ({len(response)} chars):\n{response[:300]}...")
        
        # Parse responses
        parser = del_contextual_entity_linker(None, None, None, is_parser=True)
        parsed_qids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                qid = parser(response)
                parsed_qids.append(qid)
                logger.debug(f"[DEL-CONTEXT] Entity '{entity['text']}': Parsed QID: {qid}")
            except Exception as e:
                logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[DEL-CONTEXT] Raw response: {response[:200]}")
                parsed_qids.append(None)
        
        # Get entity info from Wikidata
        logger.info(f"[DEL-CONTEXT] Fetching entity info from Wikidata for {len([q for q in parsed_qids if q])} QIDs")
        results = []
        for idx, (entity, qid) in enumerate(zip(entities, parsed_qids)):
            if qid:
                info = get_entity_info(qid)
                if 'error' not in info:
                    results.append(info)
                    logger.debug(f"[DEL-CONTEXT] Entity '{entity['text']}': Got entity info for {qid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': QID {qid} error: {info.get('error')}")
                    results.append({'id': qid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': No QID parsed")
                results.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
        
        logger.info(f"[DEL-CONTEXT] Completed: {len([r for r in results if r.get('id')])} entities linked")
        return results
    
    def _ecj(self, entities: List[Dict], candidates: List[List[Dict]]) -> str:
        logger.info(f"[ECJ] Starting Entity Consensus Judger for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            cands = candidates[idx] if idx < len(candidates) else []
            prompt = ecj(entity, cands, self.entity_db, is_parser=False)
            prompts.append(prompt)
            logger.debug(f"[ECJ] Entity {idx+1}/{len(entities)} '{entity['text']}': {len(cands)} candidates, prompt ({len(prompt)} chars)")
            if idx == 0:  # Log first prompt as sample
                logger.debug(f"[ECJ] Sample prompt (first entity):\n{prompt[:500]}...")
        
        # Call LLM
        logger.info(f"[ECJ] Calling LLM with {len(prompts)} prompts")
        llm_responses = self.llm_client.call_batch(prompts)
        logger.info(f"[ECJ] Received {len(llm_responses)} responses")
        
        # Log sample responses
        for idx, (entity, response) in enumerate(zip(entities[:3], llm_responses[:3])):
            logger.debug(f"[ECJ] Entity '{entity['text']}' response ({len(response)} chars):\n{response[:300]}...")
        
        # Parse responses
        parser = ecj(None, None, None, is_parser=True)
        parsed_qids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                qid = parser(response)
                parsed_qids.append(qid)
                logger.debug(f"[ECJ] Entity '{entity['text']}': Parsed QID: {qid}")
            except Exception as e:
                logger.warning(f"[ECJ] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[ECJ] Raw response: {response[:200]}")
                parsed_qids.append(None)
        
        # Get entity info from Wikidata
        logger.info(f"[ECJ] Fetching entity info from Wikidata for {len([q for q in parsed_qids if q])} QIDs")
        results = []
        for idx, (entity, qid) in enumerate(zip(entities, parsed_qids)):
            if qid:
                info = get_entity_info(qid)
                if 'error' not in info:
                    results.append(info)
                    logger.debug(f"[ECJ] Entity '{entity['text']}': Got entity info for {qid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[ECJ] Entity '{entity['text']}': QID {qid} error: {info.get('error')}")
                    results.append({'id': qid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[ECJ] Entity '{entity['text']}': No QID parsed")
                results.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
        
        logger.info(f"[ECJ] Completed: {len([r for r in results if r.get('id')])} entities linked")
        return results

    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        logger.info(f"[PREDICT] Starting prediction for text length: {len(text)}")
        predictions = {}

        # Step 1: Entity Detection
        logger.info("[PREDICT] === Step 1: Entity Detection ===")
        detected_entities = self._detect_entities_with_spacy(text, doc)
        logger.info(f"[PREDICT] Detected {len(detected_entities)} entities: {[e['text'] for e in detected_entities]}")

        if not detected_entities:
            logger.info("[PREDICT] No entities detected, returning empty predictions")
            return predictions

        # Step 2: ERP - Entity Reduction Processor
        logger.info("[PREDICT] === Step 2: ERP (Entity Reduction Processor) ===")
        erp_results = self._erp(detected_entities)
        logger.info(f"[PREDICT] ERP results: {[len(r) for r in erp_results]} candidates per entity")
        for idx, (entity, cands) in enumerate(zip(detected_entities, erp_results)):
            logger.debug(f"[PREDICT] Entity '{entity['text']}': {len(cands)} candidates after ERP: {[c.get('label', c.get('id', 'N/A')) for c in cands[:3]]}")

        # Step 3: DEL - Dual-perspective Entity Linker
        logger.info("[PREDICT] === Step 3: DEL (Dual-perspective Entity Linker) ===")
        logger.info("[PREDICT] --- 3a: Prior Entity Linker ---")
        del_prior_entity_linker_results = self._del_prior_entity_linker(detected_entities, erp_results)
        logger.info("[PREDICT] --- 3b: Contextual Entity Linker ---")
        del_contextual_entity_linker_results = self._del_contextual_entity_linker(detected_entities, erp_results)
        
        # Combine prior and contextual results
        candidates = [[i, j] for i, j in zip(del_prior_entity_linker_results, del_contextual_entity_linker_results)]
        logger.info(f"[PREDICT] DEL results combined: {len(candidates)} entity pairs")
        for idx, (entity, prior, context) in enumerate(zip(detected_entities, del_prior_entity_linker_results, del_contextual_entity_linker_results)):
            prior_id = prior.get('id') if prior else None
            context_id = context.get('id') if context else None
            logger.debug(f"[PREDICT] Entity '{entity['text']}': Prior={prior_id}, Context={context_id}")

        # Step 4: ECJ - Entity Consensus Judger
        logger.info("[PREDICT] === Step 4: ECJ (Entity Consensus Judger) ===")
        linked_entities = self._ecj(detected_entities, candidates)
        logger.info(f"[PREDICT] ECJ results: {len(linked_entities)} final entities")

        # Build predictions
        logger.info("[PREDICT] === Building final predictions ===")
        for entity, linked_entity, cands in zip(detected_entities, linked_entities, erp_results):
            span = (entity['start_pos'], entity['end_pos'])
            entity_id = linked_entity.get('id') if linked_entity else None
            if entity_id:
                candidate_set = {c.get('id') for c in cands if c.get('id')}
                
                logger.debug(f"[PREDICT] Entity '{entity['text']}' ({span}): Linked to {entity_id} ({linked_entity.get('label', 'N/A')}), {len(candidate_set)} candidates")
                
                predictions[span] = EntityPrediction(span, entity_id, candidate_set)

        logger.info(f"[PREDICT] Completed: {len(predictions)} predictions made")
        return predictions
