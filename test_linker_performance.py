#!/usr/bin/env python3
"""
Test script to benchmark entity linker performance
"""
import time
import json
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test samples for disease entity linking (DOID)
DISEASE_TEST_SAMPLES = [
    {
        "text": "Patient diagnosed with type 2 diabetes mellitus and hypertension. History of coronary artery disease.",
        "expected_entities": ["type 2 diabetes mellitus", "hypertension", "coronary artery disease"]
    },
    {
        "text": "The patient presents with symptoms of rheumatoid arthritis including joint pain and inflammation.",
        "expected_entities": ["rheumatoid arthritis"]
    },
    {
        "text": "Patient has a history of asthma and seasonal allergies. Recently diagnosed with chronic bronchitis.",
        "expected_entities": ["asthma", "seasonal allergies", "chronic bronchitis"]
    }
]

# Test samples for general entity linking (Wikipedia)
GENERAL_TEST_SAMPLES = [
    {
        "text": "Apple Inc. announced new iPhone models at their event in Cupertino. Tim Cook presented the keynote.",
        "expected_entities": ["Apple Inc.", "iPhone", "Cupertino", "Tim Cook"]
    },
    {
        "text": "The Eiffel Tower in Paris is one of the most visited landmarks in France.",
        "expected_entities": ["Eiffel Tower", "Paris", "France"]
    }
]


def test_linker(linker, samples: List[Dict], linker_name: str):
    """Test a linker with given samples"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {linker_name}")
    logger.info(f"{'='*60}")
    
    total_time = 0
    results = []
    
    for i, sample in enumerate(samples):
        text = sample["text"]
        logger.info(f"\nSample {i+1}: {text[:80]}...")
        
        start_time = time.time()
        try:
            predictions = linker.predict(text)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Extract predicted entities
            predicted_entities = []
            for span, prediction in predictions.items():
                entity_text = text[span[0]:span[1]]
                entity_id = prediction.entity_id
                predicted_entities.append({
                    "text": entity_text,
                    "entity_id": entity_id,
                    "span": span
                })
            
            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  Found {len(predicted_entities)} entities:")
            for ent in predicted_entities:
                logger.info(f"    - {ent['text']} -> {ent['entity_id']}")
            
            results.append({
                "sample_id": i,
                "text": text,
                "time": elapsed,
                "num_entities": len(predicted_entities),
                "entities": predicted_entities
            })
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "sample_id": i,
                "text": text,
                "time": None,
                "error": str(e)
            })
    
    avg_time = total_time / len(samples) if samples else 0
    logger.info(f"\n{linker_name} Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average time per sample: {avg_time:.2f}s")
    logger.info(f"  Target: < 30s per sample")
    logger.info(f"  Status: {'✓ PASS' if avg_time < 30 else '✗ FAIL'}")
    
    return results, avg_time


def main():
    logger.info("Starting Entity Linker Performance Tests")
    
    # Test configurations
    configs = []
    
    # Check if we can test disease linkers
    try:
        from elevant.models.entity_database import EntityDatabase
        from elevant import settings
        
        # Test OneNet Linker (Disease)
        try:
            logger.info("\n" + "="*80)
            logger.info("TESTING DISEASE ENTITY LINKERS (DOID)")
            logger.info("="*80)
            
            # Load config
            import os
            config_path = "/media/volume/LLMRag2/.local/ActDiseaseEL/configs/general-onenet-llm.config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Initialize entity database
                entity_db = EntityDatabase()
                
                # Test OneNet Linker
                from elevant.linkers.onenet_linker import OneNetLinker
                logger.info("\nInitializing OneNet Linker...")
                onenet = OneNetLinker(entity_db, config)
                onenet_results, onenet_time = test_linker(onenet, DISEASE_TEST_SAMPLES, "OneNet Linker (Disease)")
                
                # Test Graph Linker
                from elevant.linkers.graph_linker import GraphLinker
                logger.info("\nInitializing Graph Linker...")
                graph = GraphLinker(entity_db, config)
                graph_results, graph_time = test_linker(graph, DISEASE_TEST_SAMPLES, "Graph Linker (Disease)")
                
                # Compare
                logger.info("\n" + "="*80)
                logger.info("COMPARISON (Disease Linkers)")
                logger.info("="*80)
                logger.info(f"OneNet: {onenet_time:.2f}s avg")
                logger.info(f"Graph:  {graph_time:.2f}s avg")
                logger.info(f"Speedup: {onenet_time/graph_time:.2f}x" if graph_time > 0 else "N/A")
                
        except Exception as e:
            logger.error(f"Error testing disease linkers: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        logger.error(f"Error importing modules: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*80)
    logger.info("Performance Tests Complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()

