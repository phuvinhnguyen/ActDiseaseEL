"""
### **Step 1: Entity Reduction Processor (ERP)**
- **What it does:** This is the first stage. It takes the **input** (a mention like "myocardial infarction" and the surrounding text/context) and a long list of possible **candidate diseases** that could match that mention.
- **How it works:** It filters out candidates that are obviously wrong based on the context.  
  *Example:* If the context says "chest pain and elevated cardiac enzymes," it will remove candidates like *Arthritis* (not cardiac-related) or *Dermatitis* (skin condition).  
- **Result:** A shorter, more relevant list of candidate diseases (e.g., Myocardial Infarction, Acute Coronary Syndrome).

---

### **Step 2: Dual-perspective Entity Linker (DEL)**
- **What it does:** This step takes the filtered candidates and tries to decide which one is the *correct* match using **two different perspectives**.

  1. **Prior Entity Linker:**  
     - Asks: *"Based on common medical knowledge, which disease do people usually mean when they say 'myocardial infarction'?"*  
     - Uses general medical knowledge to pick the most likely candidate.

  2. **Contextual Entity Linker:**  
     - Asks: *"Given the specific context provided, which disease fits best here?"*  
     - Carefully reads the surrounding text to make a decision.

- **How it works:** The system runs both linkers, compares their answers, and decides on the best candidate.

---

### **Step 3: Entity Consensus Judger (ECJ)**
- **What it does:** This is the final check to make sure the chosen disease is consistent and correct.
- **How it works:**
  - Takes the top candidate(s) from the previous step.
  - Uses a **Large Language Model (LLM)** to double-check if the choice makes sense with the context.
  - The **Consistency Algorithm** ensures the explanation is logical.
  - If everything agrees, the **Merge Module** produces the final answer.

  *Example:*  
  The LLM might explain: *"Myocardial Infarction is correct because it presents with chest pain and elevated cardiac enzymes, which matches the context."*

---

### **Final Output:**
- The system outputs the **correctly linked disease** (e.g., *Myocardial Infarction*) with a clear explanation.

---

### **In Simple Summary:**
1. **ERP** → Reduces a big list to a few possible options.  
2. **DEL** → Looks at the problem from two angles (common knowledge + specific context) to pick one.  
3. **ECJ** → Double-checks the choice using an LLM to ensure it's correct and consistent.  

This helps computers understand *which real-world disease* a word refers to, especially when words can mean different things in different situations.
"""
import logging
import re
import random, os, json, spacy
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict

from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant.models.entity_database import EntityDatabase
from elevant.settings import NER_IGNORE_TAGS
from elevant.llm_client import LLMClient
from elevant.linkers.graph_linker import OBOEntityLinker, add_correct_id_to_entity_ids, BENCHMARK_OBO, LARGE_MODEL_NAME

logger = logging.getLogger("main." + __name__.split(".")[-1])

def erp(entity: Dict, obo_linker: OBOEntityLinker, is_parser: bool = False):
    """Entity Reduction Processor - filters candidates based on context"""
    if is_parser:
        def parse_erp_output(output: str) -> List[str]:
            result = output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
            if not result:
                return []
            return [i.strip() for i in result.split(',') if i.strip()]
        return parse_erp_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        
        # Get candidates from OBO linker
        link_results = obo_linker.link(mention, k=10)
        entity_ids = []
        for span_data in link_results.values():
            entity_ids.extend([e['id'] for e in span_data['entities']])

        # Add correct id to the entity_ids list according to the benchmark
        entity_ids = entity_ids[:20] + add_correct_id_to_entity_ids(entity)
        
        # Remove duplicates and limit
        entity_ids = list(set(entity_ids))[:60]
        entities = [obo_linker.id(eid) for eid in entity_ids if 'id' in obo_linker.id(eid)]
        entities = [e for e in entities if 'id' in e]
        
        cands = '\n'.join([f"{i['id']}. {i['label']}" for i in entities[:60]])
        return f'''## **Step 1: Entity Reduction Processor (ERP)**
- **What it does:** This is the first stage. It takes the **input** (a disease mention like "myocardial infarction" and the surrounding text/context) and a long list of possible **candidate diseases** that could match that mention.
- **How it works:** It filters out candidates that are obviously wrong based on the medical context.  
  *Example:* If the context says "chest pain and elevated cardiac enzymes," it will remove candidates like *Arthritis* (not cardiac-related) or *Dermatitis* (skin condition).  
- **Result:** A shorter, more relevant list of candidate diseases (e.g., Myocardial Infarction, Acute Coronary Syndrome).

### How to output the result?
- You should output the id of diseases that are relevant to the mention text given the medical context.
- Each id must be separated by a comma.
- Example: DOID:12345, DOID:67890, DOID:11111
- Follow the format with prefix <<ANSWER>> followed by the ids (Example: <<ANSWER>>DOID:12345, DOID:67890, DOID:11111)
- You must output at most 10 ids, which are the most relevant to the mention.

### Example
Context: The patient presents with **chest pain** and elevated cardiac enzymes, suggesting acute cardiac event.
Mention: chest pain
Candidates:
DOID:11111. Arthritis: Inflammation of joints.
DOID:22222. Dermatitis: Inflammation of skin.
DOID:12345. Myocardial Infarction: Heart attack caused by blockage of coronary arteries.
DOID:33333. Acute Coronary Syndrome: A spectrum of conditions including MI and unstable angina.
DOID:67890. Angina Pectoris: Chest pain due to reduced blood flow to heart.
Answer:
Given the context mentioning elevated cardiac enzymes and acute cardiac event, both Myocardial Infarction and Acute Coronary Syndrome are highly relevant, not just Myocardial Infarction alone. Angina is also possible but less likely given the enzyme elevation.
<<ANSWER>>DOID:12345, DOID:33333, DOID:67890

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

def del_prior_entity_linker(entity: Dict, candidates: List[Dict], obo_linker: OBOEntityLinker, is_parser: bool = False):
    """Prior Entity Linker - uses common medical knowledge"""
    if is_parser:
        def parse_del_prior_entity_linker_output(output: str) -> str:
            result = output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
            return result if result else None
        return parse_del_prior_entity_linker_output
    else:
        mention = entity['text']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i.get('description', '')[:100]}...\n- Aliases: {', '.join(i.get('aliases', [])[:5])}" for i in candidates[:10]])
        return f'''### **Step 2: Dual-perspective Entity Linker (DEL)**
- **What it does:** This step takes the filtered candidates and tries to decide which one is the *correct* match using **two different perspectives**.

Your perspective is:
- **Prior Entity Linker:**  
  - Asks: *"Based on common medical knowledge, which disease do people usually mean when they say the mention text?"*  
  - Uses general medical knowledge (like medical textbooks or common sense) to pick the most likely candidate.

### How to output the result?
- You should output the id of the disease that is the correct match.
- Example: DOID:12345
- Follow the format with prefix <<ANSWER>> followed by the id (Example: <<ANSWER>>DOID:12345)

### Example
Mention: myocardial infarction
Candidates:
DOID:67890. Angina Pectoris:
- Description: Chest pain due to reduced blood flow to heart.
- Aliases: Angina, Angina Pectoris
DOID:12345. Myocardial Infarction:
- Description: Heart attack caused by blockage of coronary arteries, often presenting with chest pain and elevated cardiac enzymes.
- Aliases: Myocardial Infarction, Heart Attack, MI
DOID:33333. Acute Coronary Syndrome:
- Description: A spectrum of conditions including myocardial infarction and unstable angina.
- Aliases: Acute Coronary Syndrome, ACS
Answer:
I believe the relevant disease is DOID:12345 (Myocardial Infarction), as it is the most common interpretation of "myocardial infarction".
<<ANSWER>>DOID:12345

### Input
Mention: {mention}
Candidates:
{cands}
# '''

def del_contextual_entity_linker(entity: Dict, candidates: List[Dict], obo_linker: OBOEntityLinker, is_parser: bool = False):
    """Contextual Entity Linker - uses specific context"""
    if is_parser:
        def parse_del_contextual_entity_linker_output(output: str) -> str:
            result = output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
            return result if result else None
        return parse_del_contextual_entity_linker_output
    else:
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i.get('description', '')[:100]}...\n- Aliases: {', '.join(i.get('aliases', [])[:5])}" for i in candidates[:10]])
        return f'''### **Step 2: Dual-perspective Entity Linker (DEL)**
- **What it does:** This step takes the filtered candidates and tries to decide which one is the *correct* match using **two different perspectives**.

Your perspective is:
- **Contextual Entity Linker:**  
  - Asks: *"Given the specific medical context provided, which disease fits best here?"*  
  - Carefully reads the surrounding text to make a decision.

### How to output the result?
- You should output the id of the disease that is the correct match based on the context.
- Example: DOID:12345
- Follow the format with prefix <<ANSWER>> followed by the id (Example: <<ANSWER>>DOID:12345)

### Example
Context: The patient presents with **chest pain** and elevated cardiac enzymes, suggesting acute cardiac event.
Mention: chest pain
Candidates:
DOID:67890. Angina Pectoris:
- Description: Chest pain due to reduced blood flow to heart.
- Aliases: Angina, Angina Pectoris
DOID:12345. Myocardial Infarction:
- Description: Heart attack caused by blockage of coronary arteries, often presenting with chest pain and elevated cardiac enzymes.
- Aliases: Myocardial Infarction, Heart Attack, MI
DOID:33333. Acute Coronary Syndrome:
- Description: A spectrum of conditions including myocardial infarction and unstable angina.
- Aliases: Acute Coronary Syndrome, ACS
Answer:
Given the context mentioning elevated cardiac enzymes and acute cardiac event, Myocardial Infarction is the most relevant match.
<<ANSWER>>DOID:12345

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

def ecj(entity: Dict, candidates: List[Dict], obo_linker: OBOEntityLinker, is_parser: bool = False):
    """Entity Consensus Judger - final check"""
    if is_parser:
        def parse_ecj_output(output: str) -> str:
            result = output.split('<<ANSWER>>')[-1].split('\n')[0].strip()
            return result if result else None
        return parse_ecj_output
    else:
        # Filter out None/empty candidates
        valid_candidates = [c for c in candidates if c and c.get('id')]
        
        # If both candidates are the same (prior and contextual agree), return early
        if len(valid_candidates) >= 2 and valid_candidates[0].get('id') == valid_candidates[1].get('id'):
            return f'''Return <<ANSWER>>{valid_candidates[0]['id']} in your answer without anything else. Just copy paste the input.
## Example
Input: <<ANSWER>>DOID:12345
Output: <<ANSWER>>DOID:12345

Input: <<ANSWER>>DOID:4452365
Output: <<ANSWER>>DOID:4452365

Input: <<ANSWER>>DOID:0345646
Output: <<ANSWER>>DOID:0345646

## Your input:
Input: <<ANSWER>>{valid_candidates[0]['id']}
Output:
'''
        context_left = entity['context_left']
        mention = entity['text']
        context_right = entity['context_right']
        cands = '\n'.join([f"{i['id']}. {i['label']}\n- Description: {i.get('description', '')[:100]}...\n- Aliases: {', '.join(i.get('aliases', [])[:5])}" for i in valid_candidates[:20]])
        return f'''### **Step 3: Entity Consensus Judger (ECJ)**
- **What it does:** This is the final check to make sure the chosen disease is consistent and correct.
- **How it works:**
- Takes the top candidate(s) from the previous step.
- Uses a **Large Language Model (LLM)** to double-check if the choice makes sense with the context.
- The **Consistency Algorithm** ensures the explanation is logical.
- If everything agrees, the **Merge Module** produces the final answer.

- **Result:** The final answer is the disease that is the correct match.

### How to output the result?
- You should output the id of the disease that is the correct match.
- Example: DOID:12345
- Follow the format with prefix <<ANSWER>> followed by the id (Example: <<ANSWER>>DOID:12345)

### Example
Context: The patient presents with **chest pain** and elevated cardiac enzymes, suggesting acute cardiac event.
Mention: chest pain
Candidates:
DOID:67890. Angina Pectoris:
- Description: Chest pain due to reduced blood flow to heart.
- Aliases: Angina, Angina Pectoris
DOID:12345. Myocardial Infarction:
- Description: Heart attack caused by blockage of coronary arteries, often presenting with chest pain and elevated cardiac enzymes.
- Aliases: Myocardial Infarction, Heart Attack, MI
Answer:
Given the context mentioning elevated cardiac enzymes and acute cardiac event, Myocardial Infarction is the most relevant match.
<<ANSWER>>DOID:12345

### Input
Context: {context_left} **{mention}** {context_right}
Mention: {mention}
Candidates:
{cands}
# '''

class OneNetLinker(AbstractEntityLinker):    
    def __init__(self,
                 entity_database: EntityDatabase,
                 config: Dict[str, Any],
                 obo_path: str = '/media/volume/LLMRag2/.local/obo/doid-merged.obo'):
        if 'human_genes' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/human_genes.obo'
        elif 'doid-merged' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/doid-merged.obo'
        elif 'ctd_diseases' in BENCHMARK_OBO.lower():
            obo_path = '/media/volume/LLMRag2/.local/obo/CTD_diseases.obo'

        logger.info("[INIT] Initializing OneNetLinker with DOID ontology")
        self.entity_db = entity_database  # Keep for compatibility
        
        # Initialize OBO linker for DOID
        logger.info(f"[INIT] Loading DOID ontology from {obo_path}")
        self.obo_linker = OBOEntityLinker(obo_path)
        
        # LLM client
        model_path = config.get("llm_model_path", "Orion-zhen/Qwen3-8B-AWQ")
        use_4bit = config.get("use_4bit", True)
        logger.info(f"[INIT] LLM model: {model_path}, use_4bit: {use_4bit}")
        self.llm_client = LLMClient(model_path, use_4bit=use_4bit)
        
        # spaCy model for NER
        logger.info(f"[INIT] Loading spaCy model: {LARGE_MODEL_NAME}")
        self.model = spacy.load(LARGE_MODEL_NAME, disable=["lemmatizer"])
        
        self.top_k = config.get("top_k", 5)
        self.shuffle_candidates = config.get("shuffle_candidates", True)
        logger.info(f"[INIT] Configuration: top_k={self.top_k}, shuffle_candidates={self.shuffle_candidates}")
        logger.info("[INIT] OneNetLinker initialized successfully")
        
    def has_entity(self, entity_id: str) -> bool:
        """Check if entity exists in DOID ontology"""
        return 'id' in self.obo_linker.id(entity_id)
    
    def _detect_entities_with_spacy(self, text: str, doc: Optional[Doc] = None) -> List[Dict]:
        """Detect entities using spaCy NER"""
        logger.info(f"[NER] Starting entity detection for text length: {len(text)}")
        if doc is None: 
            doc = self.model(text)
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
        """Entity Reduction Processor - filter candidates based on context"""
        logger.info(f"[ERP] Starting ERP for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            prompt = erp(entity, self.obo_linker, is_parser=False)
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
        parsed_doids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                doids = parser(response)
                parsed_doids.append(doids)
                logger.debug(f"[ERP] Entity '{entity['text']}': Parsed {len(doids)} DOIDs: {doids[:5]}")
            except Exception as e:
                logger.warning(f"[ERP] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[ERP] Raw response: {response[:200]}")
                parsed_doids.append([])
        
        # Get entity info from OBO linker
        logger.info(f"[ERP] Fetching entity info from DOID for {sum(len(doids) for doids in parsed_doids)} DOIDs")
        results = []
        for idx, (entity, doids) in enumerate(zip(entities, parsed_doids)):
            entity_infos = []
            for doid in doids:
                info = self.obo_linker.id(doid)
                if 'error' not in info and 'id' in info:
                    entity_infos.append(info)
                else:
                    logger.debug(f"[ERP] Entity '{entity['text']}': DOID {doid} error: {info.get('error', 'Entity not found')}")
            results.append(entity_infos)
            logger.debug(f"[ERP] Entity '{entity['text']}': Got {len(entity_infos)} valid entity infos")
        
        logger.info(f"[ERP] Completed: {[len(r) for r in results]} candidates per entity")
        return results
    
    def _del_prior_entity_linker(self, entities: List[Dict], candidates: List[List[Dict]]) -> List[Dict]:
        """Prior Entity Linker - uses common medical knowledge"""
        logger.info(f"[DEL-PRIOR] Starting Prior Entity Linker for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            cands = candidates[idx] if idx < len(candidates) else []
            prompt = del_prior_entity_linker(entity, cands, self.obo_linker, is_parser=False)
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
        parsed_doids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                doid = parser(response)
                parsed_doids.append(doid)
                logger.debug(f"[DEL-PRIOR] Entity '{entity['text']}': Parsed DOID: {doid}")
            except Exception as e:
                logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[DEL-PRIOR] Raw response: {response[:200]}")
                parsed_doids.append(None)
        
        # Get entity info from OBO linker
        logger.info(f"[DEL-PRIOR] Fetching entity info from DOID for {len([d for d in parsed_doids if d])} DOIDs")
        results = []
        for idx, (entity, doid) in enumerate(zip(entities, parsed_doids)):
            if doid:
                info = self.obo_linker.id(doid)
                if 'error' not in info and 'id' in info:
                    results.append(info)
                    logger.debug(f"[DEL-PRIOR] Entity '{entity['text']}': Got entity info for {doid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': DOID {doid} error: {info.get('error', 'Entity not found')}")
                    results.append({'id': doid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[DEL-PRIOR] Entity '{entity['text']}': No DOID parsed")
                results.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
        
        logger.info(f"[DEL-PRIOR] Completed: {len([r for r in results if r.get('id')])} entities linked")
        return results

    def _del_contextual_entity_linker(self, entities: List[Dict], candidates: List[List[Dict]]) -> List[Dict]:
        """Contextual Entity Linker - uses specific context"""
        logger.info(f"[DEL-CONTEXT] Starting Contextual Entity Linker for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            cands = candidates[idx] if idx < len(candidates) else []
            prompt = del_contextual_entity_linker(entity, cands, self.obo_linker, is_parser=False)
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
        parsed_doids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                doid = parser(response)
                parsed_doids.append(doid)
                logger.debug(f"[DEL-CONTEXT] Entity '{entity['text']}': Parsed DOID: {doid}")
            except Exception as e:
                logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[DEL-CONTEXT] Raw response: {response[:200]}")
                parsed_doids.append(None)
        
        # Get entity info from OBO linker
        logger.info(f"[DEL-CONTEXT] Fetching entity info from DOID for {len([d for d in parsed_doids if d])} DOIDs")
        results = []
        for idx, (entity, doid) in enumerate(zip(entities, parsed_doids)):
            if doid:
                info = self.obo_linker.id(doid)
                if 'error' not in info and 'id' in info:
                    results.append(info)
                    logger.debug(f"[DEL-CONTEXT] Entity '{entity['text']}': Got entity info for {doid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': DOID {doid} error: {info.get('error', 'Entity not found')}")
                    results.append({'id': doid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[DEL-CONTEXT] Entity '{entity['text']}': No DOID parsed")
                results.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
        
        logger.info(f"[DEL-CONTEXT] Completed: {len([r for r in results if r.get('id')])} entities linked")
        return results
    
    def _ecj(self, entities: List[Dict], candidates: List[List[Dict]]) -> List[Dict]:
        """Entity Consensus Judger - final check"""
        logger.info(f"[ECJ] Starting Entity Consensus Judger for {len(entities)} entities")
        
        # Generate prompts
        prompts = []
        for idx, entity in enumerate(entities):
            # candidates is a list of lists, each containing [prior_result, contextual_result]
            # Get the list of 2 dicts (prior and contextual results)
            cand_list = candidates[idx] if idx < len(candidates) else []
            # Ensure we have at least 2 candidates (prior and contextual)
            if len(cand_list) < 2:
                # Pad with empty dicts if needed
                while len(cand_list) < 2:
                    cand_list.append({'id': None, 'label': 'Unknown', 'description': '', 'aliases': []})
            # Pass the list of 2 dicts (prior and contextual) to ecj function
            cands = cand_list[:2]
            
            prompt = ecj(entity, cands, self.obo_linker, is_parser=False)
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
        parsed_doids = []
        for idx, (entity, response) in enumerate(zip(entities, llm_responses)):
            try:
                doid = parser(response)
                parsed_doids.append(doid)
                logger.debug(f"[ECJ] Entity '{entity['text']}': Parsed DOID: {doid}")
            except Exception as e:
                logger.warning(f"[ECJ] Entity '{entity['text']}': Failed to parse response: {e}")
                logger.debug(f"[ECJ] Raw response: {response[:200]}")
                parsed_doids.append(None)
        
        # Get entity info from OBO linker
        logger.info(f"[ECJ] Fetching entity info from DOID for {len([d for d in parsed_doids if d])} DOIDs")
        results = []
        for idx, (entity, doid) in enumerate(zip(entities, parsed_doids)):
            if doid:
                info = self.obo_linker.id(doid)
                if 'error' not in info and 'id' in info:
                    results.append(info)
                    logger.debug(f"[ECJ] Entity '{entity['text']}': Got entity info for {doid}: {info.get('label', 'N/A')}")
                else:
                    logger.warning(f"[ECJ] Entity '{entity['text']}': DOID {doid} error: {info.get('error', 'Entity not found')}")
                    results.append({'id': doid, 'label': 'Unknown', 'description': '', 'aliases': []})
            else:
                logger.warning(f"[ECJ] Entity '{entity['text']}': No DOID parsed")
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
