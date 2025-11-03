import random
from typing import Dict, Tuple, Optional, Any

import spacy
from spacy.tokens import Doc

from elevant.linkers.abstract_entity_linker import AbstractEntityLinker
from elevant.models.entity_prediction import EntityPrediction
from elevant import settings


class RandomLinker(AbstractEntityLinker):
    """
    A truly simple random linker for testing and baseline evaluation.
    Uses SpaCy NER to detect entities and randomly assigns entity IDs.
    Does not depend on any external database files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.linker_identifier = config.get("linker_name", "SimpleRandom")
        self.ner_identifier = "SpaCy"
        
        # Load SpaCy model for NER without any postprocessing
        self.model = spacy.load(settings.LARGE_MODEL_NAME, disable=["lemmatizer"])
        
        # A list of common Wikidata entity IDs for random selection
        # These are popular entities (countries, people, organizations, etc.)
        self.random_entity_pool = [
            "Q30",      # United States
            "Q142",     # France
            "Q183",     # Germany
            "Q145",     # United Kingdom
            "Q159",     # Russia
            "Q148",     # China
            "Q17",      # Japan
            "Q668",     # India
            "Q155",     # Brazil
            "Q16",      # Canada
            "Q96",      # Mexico
            "Q38",      # Italy
            "Q29",      # Spain
            "Q408",     # Australia
            "Q664",     # New Zealand
            "Q5043",    # Christianity
            "Q9268",    # Judaism
            "Q432",     # Islam
            "Q7318",    # Nazi Germany
            "Q15180",   # Soviet Union
            "Q193563",  # European Union
            "Q1065",    # United Nations
            "Q8733",    # Barack Obama
            "Q22686",   # Donald Trump
            "Q76",      # Barack Obama
            "Q6279",    # Joe Biden
            "Q23505",   # George W. Bush
            "Q9696",    # John F. Kennedy
            "Q255"      # Albert Einstein
        ]
        
        # Set random seed for reproducibility if specified in config
        if "random_seed" in config:
            random.seed(config["random_seed"])
    
    def has_entity(self, entity_id: str) -> bool:
        """Always return True for any entity ID."""
        return True
    
    def predict(self,
                text: str,
                doc: Optional[Doc] = None,
                uppercase: Optional[bool] = False) -> Dict[Tuple[int, int], EntityPrediction]:
        """
        Predict entities by detecting spans with SpaCy NER and randomly assigning entity IDs.
        """
        if doc is None:
            doc = self.model(text)
        
        predictions = {}
        
        # NER tags to ignore (dates, quantities, etc.)
        ignore_tags = {"CARDINAL", "MONEY", "ORDINAL", "QUANTITY", "TIME", "DATE"}
        
        # Extract entity spans using SpaCy NER
        for ent in doc.ents:
            if ent.label_ in ignore_tags:
                continue
            
            span = (ent.start_char, ent.end_char)
            snippet = text[span[0]:span[1]]
            
            # Skip lowercase entities if uppercase flag is set
            if uppercase and snippet.islower():
                continue
            
            # Randomly select an entity ID from the pool
            predicted_entity_id = random.choice(self.random_entity_pool)
            
            # Create prediction with the random entity ID
            # The candidates set is the entire pool (for consistency)
            candidates = set(self.random_entity_pool)
            predictions[span] = EntityPrediction(span, predicted_entity_id, candidates)
        
        return predictions

