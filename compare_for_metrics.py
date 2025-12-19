#!/usr/bin/env python3
"""
Evaluate GraphLLM and OneNetLLM predictions against ground truth benchmarks.
Automatically finds and compares all matching files from doid-results and meddata.
Generates comprehensive markdown tables comparing OneNet vs Graph across all scenarios.
"""

import json
import argparse
import re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from pathlib import Path


def expand_entity_ids(entity_id: str) -> List[str]:
    """
    Expand entity ID that may contain multiple IDs into a list.
    Handles malformed formats:
    - Comma-separated: "12458,12772,12775" -> ["12458", "12772", "12775"]
    - Pipe-separated: "D019572|D010871" -> ["D019572", "D010871"]
    - With taxon info: "172659(Tax:6239)" -> ["172659"]
    - Multiple patterns: "8482,20361(Tax:10090)" -> ["8482", "20361"]
    """
    if not entity_id:
        return []
    
    # Remove taxon information in parentheses: "172659(Tax:6239)" -> "172659"
    entity_id = re.sub(r'\([^)]*\)', '', entity_id)
    
    # Split by comma or pipe to handle multiple IDs
    parts = re.split(r'[,|]', entity_id)
    
    expanded_ids = []
    for part in parts:
        part = part.strip()
        if part:
            expanded_ids.append(part)
    
    return expanded_ids if expanded_ids else [entity_id]

def normalize_entity_id(entity_id: str) -> str:
    """
    Normalize entity ID for comparison (returns first ID from expanded list).
    Used for backward compatibility and simple comparisons.
    
    For NCBI benchmark: removes MESH: prefix so D001260 and MESH:D001260 both become D001260
    For gene benchmarks: removes NCBIGene: prefix
    """
    expanded = expand_entity_ids(entity_id)
    if not expanded:
        return ""
    
    normalized = expanded[0]
    
    # Remove MESH: prefix for NCBI benchmark (ground truth uses D001260, predictions use MESH:D001260)
    if normalized.startswith("MESH:"):
        normalized = normalized.replace("MESH:", "")
    
    # Remove NCBIGene: prefix for gene benchmarks (for backward compatibility)
    if normalized.startswith("NCBIGene:"):
        normalized = normalized.replace("NCBIGene:", "")
    
    return normalized


def spans_overlap(span1: List[int], span2: List[int]) -> bool:
    """Check if two spans overlap."""
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 <= start2 or end2 <= start1)


def spans_match(span1: List[int], span2: List[int], exact: bool = False) -> bool:
    """Check if two spans match (exact or overlap)."""
    if exact:
        return span1 == span2
    return spans_overlap(span1, span2)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_mentions(entry: Dict, key: str = "labels") -> List[Dict]:
    """Extract mentions from entry (either labels or entity_mentions)."""
    mentions = []
    if key in entry:
        for mention in entry[key]:
            # Handle different formats: 'entity_id' (ground truth) vs 'id' (predictions)
            entity_id = mention.get('entity_id') or mention.get('id', '')
            if entity_id:
                # For predictions, entity might be nested
                if 'entity' in mention and isinstance(mention['entity'], dict):
                    entity_id = mention['entity'].get('id', entity_id)
                
                # Expand entity IDs to handle multiple IDs (comma/pipe separated)
                expanded_ids = expand_entity_ids(str(entity_id))
                # Normalize each expanded ID
                normalized_expanded_ids = [normalize_entity_id(eid) for eid in expanded_ids]
                
                mentions.append({
                    'span': tuple(mention['span']),
                    'entity_id': normalize_entity_id(str(entity_id)),  # First ID for backward compatibility
                    'entity_ids': normalized_expanded_ids,  # All expanded IDs for matching
                    'name': mention.get('name', '') or mention.get('entity_name', ''),
                    'type': mention.get('type', '')
                })
    return mentions


def compute_ner_metrics(ground_truth: List[Dict], predictions: List[Dict]) -> Dict:
    """Compute NER metrics (span matching only, ignoring entity IDs)."""
    
    # Initialize counters
    tp_exact = 0
    tp_overlap = 0
    fp_exact = 0
    fp_overlap = 0
    fn_exact = 0
    fn_overlap = 0
    
    # Create mapping from doc_id to entries
    gt_dict = {entry['id']: entry for entry in ground_truth}
    pred_dict = {entry['id']: entry for entry in predictions}
    
    # Process each document
    for doc_id in sorted(set(gt_dict.keys()) | set(pred_dict.keys())):
        gt_entry = gt_dict.get(doc_id)
        pred_entry = pred_dict.get(doc_id)
        
        if not gt_entry:
            continue
        
        gt_mentions = extract_mentions(gt_entry, 'labels')
        pred_mentions = extract_mentions(pred_entry, 'entity_mentions') if pred_entry else []
        
        # Track which mentions have been matched (NER: only span matching)
        gt_matched_exact = set()
        gt_matched_overlap = set()
        pred_matched_exact = set()
        pred_matched_overlap = set()
        
        # Match exact spans first (NER: any span match counts)
        for i, gt_mention in enumerate(gt_mentions):
            for j, pred_mention in enumerate(pred_mentions):
                if j in pred_matched_exact:
                    continue
                if spans_match(gt_mention['span'], pred_mention['span'], exact=True):
                    tp_exact += 1
                    gt_matched_exact.add(i)
                    pred_matched_exact.add(j)
                    break
        
        # Match overlapping spans (NER: any overlap counts)
        for i, gt_mention in enumerate(gt_mentions):
            if i in gt_matched_exact or i in gt_matched_overlap:
                continue
            for j, pred_mention in enumerate(pred_mentions):
                if j in pred_matched_exact or j in pred_matched_overlap:
                    continue
                if spans_overlap(gt_mention['span'], pred_mention['span']):
                    tp_overlap += 1
                    gt_matched_overlap.add(i)
                    pred_matched_overlap.add(j)
                    break
        
        # Count false negatives
        fn_exact += len(gt_mentions) - len(gt_matched_exact)
        fn_overlap += len(gt_mentions) - len(gt_matched_overlap) - len(gt_matched_exact)
        
        # Count false positives
        fp_exact += len(pred_mentions) - len(pred_matched_exact)
        fp_overlap += len(pred_mentions) - len(pred_matched_exact) - len(pred_matched_overlap)
    
    # Compute metrics
    precision_exact = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) > 0 else 0
    recall_exact = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) > 0 else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0
    
    precision_overlap = tp_overlap / (tp_overlap + fp_overlap) if (tp_overlap + fp_overlap) > 0 else 0
    recall_overlap = tp_overlap / (tp_overlap + fn_overlap) if (tp_overlap + fn_overlap) > 0 else 0
    f1_overlap = 2 * precision_overlap * recall_overlap / (precision_overlap + recall_overlap) if (precision_overlap + recall_overlap) > 0 else 0
    
    return {
        'exact_matching': {
            'precision': precision_exact,
            'recall': recall_exact,
            'f1': f1_exact,
            'tp': tp_exact,
            'fp': fp_exact,
            'fn': fn_exact
        },
        'overlap_matching': {
            'precision': precision_overlap,
            'recall': recall_overlap,
            'f1': f1_overlap,
            'tp': tp_overlap,
            'fp': fp_overlap,
            'fn': fn_overlap
        }
    }


def compute_linking_metrics(ground_truth: List[Dict], predictions: List[Dict]) -> Dict:
    """Compute linking metrics (span + entity ID matching)."""
    
    # Initialize counters
    tp_exact = 0
    tp_overlap = 0
    fp_exact = 0
    fp_overlap = 0
    fn_exact = 0
    fn_overlap = 0
    
    # Matched-only metrics
    tp_matched_exact = 0
    fp_matched_exact = 0
    tp_matched_overlap = 0
    fp_matched_overlap = 0
    
    # Create mapping from doc_id to entries
    gt_dict = {entry['id']: entry for entry in ground_truth}
    pred_dict = {entry['id']: entry for entry in predictions}
    
    # Process each document
    for doc_id in sorted(set(gt_dict.keys()) | set(pred_dict.keys())):
        gt_entry = gt_dict.get(doc_id)
        pred_entry = pred_dict.get(doc_id)
        
        if not gt_entry:
            continue
        
        gt_mentions = extract_mentions(gt_entry, 'labels')
        pred_mentions = extract_mentions(pred_entry, 'entity_mentions') if pred_entry else []
        
        # Track which mentions have been matched
        gt_matched_exact = set()
        gt_matched_overlap = set()
        pred_matched_exact = set()
        pred_matched_overlap = set()
        
        # Match exact spans first (linking: span + entity ID must match)
        for i, gt_mention in enumerate(gt_mentions):
            for j, pred_mention in enumerate(pred_mentions):
                if j in pred_matched_exact:
                    continue
                if spans_match(gt_mention['span'], pred_mention['span'], exact=True):
                    # Check if any of the expanded entity IDs match
                    gt_ids = set(gt_mention.get('entity_ids', [gt_mention['entity_id']]))
                    pred_ids = set(pred_mention.get('entity_ids', [pred_mention['entity_id']]))
                    if gt_ids & pred_ids:  # Any overlap means match
                        tp_exact += 1
                        tp_matched_exact += 1
                        gt_matched_exact.add(i)
                        pred_matched_exact.add(j)
                    else:
                        fp_exact += 1
                        fp_matched_exact += 1
                        pred_matched_exact.add(j)
                    break
        
        # Match overlapping spans (linking: span overlap + entity ID must match)
        for i, gt_mention in enumerate(gt_mentions):
            if i in gt_matched_exact or i in gt_matched_overlap:
                continue
            for j, pred_mention in enumerate(pred_mentions):
                if j in pred_matched_exact or j in pred_matched_overlap:
                    continue
                if spans_overlap(gt_mention['span'], pred_mention['span']):
                    # Check if any of the expanded entity IDs match
                    gt_ids = set(gt_mention.get('entity_ids', [gt_mention['entity_id']]))
                    pred_ids = set(pred_mention.get('entity_ids', [pred_mention['entity_id']]))
                    if gt_ids & pred_ids:  # Any overlap means match
                        tp_overlap += 1
                        tp_matched_overlap += 1
                        gt_matched_overlap.add(i)
                        pred_matched_overlap.add(j)
                    else:
                        fp_overlap += 1
                        fp_matched_overlap += 1
                        pred_matched_overlap.add(j)
                    break
        
        # Count false negatives
        fn_exact += len(gt_mentions) - len(gt_matched_exact)
        fn_overlap += len(gt_mentions) - len(gt_matched_overlap) - len(gt_matched_exact)
        
        # Count false positives
        fp_exact += len(pred_mentions) - len(pred_matched_exact)
        fp_overlap += len(pred_mentions) - len(pred_matched_exact) - len(pred_matched_overlap)
    
    # Compute metrics
    precision_exact = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) > 0 else 0
    recall_exact = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) > 0 else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0
    
    precision_overlap = tp_overlap / (tp_overlap + fp_overlap) if (tp_overlap + fp_overlap) > 0 else 0
    recall_overlap = tp_overlap / (tp_overlap + fn_overlap) if (tp_overlap + fn_overlap) > 0 else 0
    f1_overlap = 2 * precision_overlap * recall_overlap / (precision_overlap + recall_overlap) if (precision_overlap + recall_overlap) > 0 else 0
    
    precision_matched_exact = tp_matched_exact / (tp_matched_exact + fp_matched_exact) if (tp_matched_exact + fp_matched_exact) > 0 else 0
    precision_matched_overlap = tp_matched_overlap / (tp_matched_overlap + fp_matched_overlap) if (tp_matched_overlap + fp_matched_overlap) > 0 else 0
    
    return {
        'exact_matching': {
            'precision': precision_exact,
            'recall': recall_exact,
            'f1': f1_exact,
            'tp': tp_exact,
            'fp': fp_exact,
            'fn': fn_exact
        },
        'overlap_matching': {
            'precision': precision_overlap,
            'recall': recall_overlap,
            'f1': f1_overlap,
            'tp': tp_overlap,
            'fp': fp_overlap,
            'fn': fn_overlap
        },
        'matched_only': {
            'exact': {
                'precision': precision_matched_exact,
                'tp': tp_matched_exact,
                'fp': fp_matched_exact
            },
            'overlap': {
                'precision': precision_matched_overlap,
                'tp': tp_matched_overlap,
                'fp': fp_matched_overlap
            }
        }
    }


def compute_metrics(ground_truth: List[Dict], predictions: List[Dict]) -> Dict:
    """Compute comprehensive evaluation metrics (NER, linking, and full)."""
    ner_metrics = compute_ner_metrics(ground_truth, predictions)
    linking_metrics = compute_linking_metrics(ground_truth, predictions)
    
    # Full results combine NER and linking
    return {
        'ner': ner_metrics,
        'linking': linking_metrics,
        'full': linking_metrics  # Full is same as linking (end-to-end)
    }


def find_matching_files(meddata_dir: Path, doid_results_dir: Path) -> List[Tuple[str, Path, Optional[Path], Optional[Path]]]:
    """
    Find matching benchmark and prediction files.
    Returns list of (scenario_name, benchmark_path, onenet_path, graph_path) tuples.
    """
    matches = []
    
    # Find all benchmark files
    benchmark_files = list(meddata_dir.glob("*.benchmark.jsonl"))
    benchmark_files.extend(meddata_dir.glob("*healthcare_history.jsonl"))
    
    for bench_file in benchmark_files:
        bench_name = bench_file.stem.replace('.benchmark', '').replace('_healthcare_history', '')
        
        # Try different naming patterns for predictions
        onenet_path = None
        graph_path = None
        
        # Names to try when looking for prediction files.
        # For NCBI we support both \"NCBItestset\" and legacy \"NCBI\" prefixes.
        search_names = [bench_name]
        if bench_name == "NCBItestset":
            search_names.append("NCBI")
        
        # Pattern 1: {name}.OneNetLLM.jsonl / {name}.GraphLLM.jsonl
        onenet_candidate1 = None
        graph_candidate1 = None
        for name in search_names:
            cand_onenet = doid_results_dir / f"{name}.OneNetLLM.jsonl"
            cand_graph = doid_results_dir / f"{name}.GraphLLM.jsonl"
            if cand_onenet.exists() and onenet_candidate1 is None:
                onenet_candidate1 = cand_onenet
            if cand_graph.exists() and graph_candidate1 is None:
                graph_candidate1 = cand_graph
        
        # Pattern 2: {bench_name}_onenet_llm_doid.jsonl / {bench_name}_graph_llm_doid.jsonl
        onenet_candidate2 = doid_results_dir / f"{bench_name}_onenet_llm_doid.jsonl"
        graph_candidate2 = doid_results_dir / f"{bench_name}_graph_llm_doid.jsonl"
        
        # Pattern 3: BC2GNtest.PubTator.* (special case)
        if 'BC2GN' in bench_name:
            onenet_candidate3 = doid_results_dir / "BC2GNtest.PubTator.OneNetLLM.jsonl"
            graph_candidate3 = doid_results_dir / "BC2GNtest.PubTator.GraphLLM.jsonl"
            if onenet_candidate3.exists():
                onenet_path = onenet_candidate3
            if graph_candidate3.exists():
                graph_path = graph_candidate3
        
        if not onenet_path:
            if onenet_candidate1 is not None:
                onenet_path = onenet_candidate1
            elif onenet_candidate2.exists():
                onenet_path = onenet_candidate2
        
        if not graph_path:
            if graph_candidate1 is not None:
                graph_path = graph_candidate1
            elif graph_candidate2.exists():
                graph_path = graph_candidate2
        
        # Also check for versioned files (v2, v3)
        if not graph_path:
            for v in ['v2', 'v3']:
                graph_candidate_v = doid_results_dir / f"{bench_name}_graph_llm_doid_{v}.jsonl"
                if graph_candidate_v.exists():
                    graph_path = graph_candidate_v
                    break
        
        # Check for english/multilingual versioned files
        if not graph_path and bench_name in ['english', 'multilingual']:
            for v in ['v2', 'v3']:
                graph_candidate_v = doid_results_dir / f"{bench_name}_graph_llm_doid_{v}.jsonl"
                if graph_candidate_v.exists():
                    graph_path = graph_candidate_v
                    break
        
        matches.append((bench_name, bench_file, onenet_path, graph_path))
    
    return matches


def _add_metrics_table(md_lines: List[str], title: str, scenarios: List[str], results: Dict, 
                       metric_type: str, exact_key: str = 'exact_matching', overlap_key: str = 'overlap_matching'):
    """Helper function to add a metrics table."""
    md_lines.append(f"## {title}\n")
    md_lines.append("| Scenario | Method | Precision | Recall | F1 | TP | FP | FN |")
    md_lines.append("|----------|--------|-----------|--------|----|----|----|----|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        
        # OneNet row
        if 'onenet' in data:
            onenet_data = data['onenet'].get(metric_type, {})
            onenet_exact = onenet_data.get(exact_key, {})
            md_lines.append(
                f"| {scenario} | OneNet | "
                f"{onenet_exact.get('precision', 0):.4f} | "
                f"{onenet_exact.get('recall', 0):.4f} | "
                f"{onenet_exact.get('f1', 0):.4f} | "
                f"{onenet_exact.get('tp', 0)} | "
                f"{onenet_exact.get('fp', 0)} | "
                f"{onenet_exact.get('fn', 0)} |"
            )
        
        # Graph row
        if 'graph' in data:
            graph_data = data['graph'].get(metric_type, {})
            graph_exact = graph_data.get(exact_key, {})
            md_lines.append(
                f"| {scenario} | Graph | "
                f"{graph_exact.get('precision', 0):.4f} | "
                f"{graph_exact.get('recall', 0):.4f} | "
                f"{graph_exact.get('f1', 0):.4f} | "
                f"{graph_exact.get('tp', 0)} | "
                f"{graph_exact.get('fp', 0)} | "
                f"{graph_exact.get('fn', 0)} |"
            )
    
    md_lines.append("")
    
    # Overlap matching table
    md_lines.append(f"### {title} - Overlap Matching\n")
    md_lines.append("| Scenario | Method | Precision | Recall | F1 | TP | FP | FN |")
    md_lines.append("|----------|--------|-----------|--------|----|----|----|----|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        
        # OneNet row
        if 'onenet' in data:
            onenet_data = data['onenet'].get(metric_type, {})
            onenet_overlap = onenet_data.get(overlap_key, {})
            md_lines.append(
                f"| {scenario} | OneNet | "
                f"{onenet_overlap.get('precision', 0):.4f} | "
                f"{onenet_overlap.get('recall', 0):.4f} | "
                f"{onenet_overlap.get('f1', 0):.4f} | "
                f"{onenet_overlap.get('tp', 0)} | "
                f"{onenet_overlap.get('fp', 0)} | "
                f"{onenet_overlap.get('fn', 0)} |"
            )
        
        # Graph row
        if 'graph' in data:
            graph_data = data['graph'].get(metric_type, {})
            graph_overlap = graph_data.get(overlap_key, {})
            md_lines.append(
                f"| {scenario} | Graph | "
                f"{graph_overlap.get('precision', 0):.4f} | "
                f"{graph_overlap.get('recall', 0):.4f} | "
                f"{graph_overlap.get('f1', 0):.4f} | "
                f"{graph_overlap.get('tp', 0)} | "
                f"{graph_overlap.get('fp', 0)} | "
                f"{graph_overlap.get('fn', 0)} |"
            )
    
    md_lines.append("")


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table comparing OneNet vs Graph across all scenarios."""
    
    md_lines = []
    md_lines.append("# Evaluation Results: OneNet vs Graph\n")
    md_lines.append("Comparison of OneNetLLM and GraphLLM across different scenarios.\n")
    
    scenarios = sorted(results.keys())
    
    # NER Results (span matching only)
    _add_metrics_table(md_lines, "NER Results (Span Detection Only)", scenarios, results, 'ner')
    
    # Linking Results (span + entity ID matching)
    _add_metrics_table(md_lines, "Linking Results (Span + Entity ID Matching)", scenarios, results, 'linking')
    
    # Full Results (end-to-end, same as linking)
    _add_metrics_table(md_lines, "Full Results (End-to-End)", scenarios, results, 'full')
    
    # Summary comparison table (F1 scores) - NER
    md_lines.append("## Summary: NER F1 Scores Comparison\n")
    md_lines.append("| Scenario | OneNet (Exact) | Graph (Exact) | OneNet (Overlap) | Graph (Overlap) |")
    md_lines.append("|----------|----------------|--------------|------------------|-----------------|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        onenet_ner = data.get('onenet', {}).get('ner', {})
        graph_ner = data.get('graph', {}).get('ner', {})
        onenet_exact_f1 = onenet_ner.get('exact_matching', {}).get('f1', 0)
        graph_exact_f1 = graph_ner.get('exact_matching', {}).get('f1', 0)
        onenet_overlap_f1 = onenet_ner.get('overlap_matching', {}).get('f1', 0)
        graph_overlap_f1 = graph_ner.get('overlap_matching', {}).get('f1', 0)
        
        md_lines.append(
            f"| {scenario} | "
            f"{onenet_exact_f1:.4f} | "
            f"{graph_exact_f1:.4f} | "
            f"{onenet_overlap_f1:.4f} | "
            f"{graph_overlap_f1:.4f} |"
        )
    
    md_lines.append("")
    
    # Summary comparison table (F1 scores) - Linking
    md_lines.append("## Summary: Linking F1 Scores Comparison\n")
    md_lines.append("| Scenario | OneNet (Exact) | Graph (Exact) | OneNet (Overlap) | Graph (Overlap) |")
    md_lines.append("|----------|----------------|--------------|------------------|-----------------|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        onenet_linking = data.get('onenet', {}).get('linking', {})
        graph_linking = data.get('graph', {}).get('linking', {})
        onenet_exact_f1 = onenet_linking.get('exact_matching', {}).get('f1', 0)
        graph_exact_f1 = graph_linking.get('exact_matching', {}).get('f1', 0)
        onenet_overlap_f1 = onenet_linking.get('overlap_matching', {}).get('f1', 0)
        graph_overlap_f1 = graph_linking.get('overlap_matching', {}).get('f1', 0)
        
        md_lines.append(
            f"| {scenario} | "
            f"{onenet_exact_f1:.4f} | "
            f"{graph_exact_f1:.4f} | "
            f"{onenet_overlap_f1:.4f} | "
            f"{graph_overlap_f1:.4f} |"
        )
    
    md_lines.append("")
    
    # Summary comparison table (F1 scores) - Full
    md_lines.append("## Summary: Full Results F1 Scores Comparison\n")
    md_lines.append("| Scenario | OneNet (Exact) | Graph (Exact) | OneNet (Overlap) | Graph (Overlap) |")
    md_lines.append("|----------|----------------|--------------|------------------|-----------------|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        onenet_full = data.get('onenet', {}).get('full', {})
        graph_full = data.get('graph', {}).get('full', {})
        onenet_exact_f1 = onenet_full.get('exact_matching', {}).get('f1', 0)
        graph_exact_f1 = graph_full.get('exact_matching', {}).get('f1', 0)
        onenet_overlap_f1 = onenet_full.get('overlap_matching', {}).get('f1', 0)
        graph_overlap_f1 = graph_full.get('overlap_matching', {}).get('f1', 0)
        
        md_lines.append(
            f"| {scenario} | "
            f"{onenet_exact_f1:.4f} | "
            f"{graph_exact_f1:.4f} | "
            f"{onenet_overlap_f1:.4f} | "
            f"{graph_overlap_f1:.4f} |"
        )
    
    md_lines.append("")
    
    # Matched-only metrics table (for linking)
    md_lines.append("## Matched-Only Evaluation (Disambiguation Accuracy)\n")
    md_lines.append("| Scenario | Method | Exact Precision | Overlap Precision |")
    md_lines.append("|----------|--------|-----------------|-------------------|")
    
    for scenario in scenarios:
        if scenario not in results:
            continue
        
        data = results[scenario]
        
        # OneNet row
        if 'onenet' in data:
            onenet_linking = data['onenet'].get('linking', {})
            onenet_matched = onenet_linking.get('matched_only', {})
            exact_prec = onenet_matched.get('exact', {}).get('precision', 0)
            overlap_prec = onenet_matched.get('overlap', {}).get('precision', 0)
            md_lines.append(
                f"| {scenario} | OneNet | "
                f"{exact_prec:.4f} | "
                f"{overlap_prec:.4f} |"
            )
        
        # Graph row
        if 'graph' in data:
            graph_linking = data['graph'].get('linking', {})
            graph_matched = graph_linking.get('matched_only', {})
            exact_prec = graph_matched.get('exact', {}).get('precision', 0)
            overlap_prec = graph_matched.get('overlap', {}).get('precision', 0)
            md_lines.append(
                f"| {scenario} | Graph | "
                f"{exact_prec:.4f} | "
                f"{overlap_prec:.4f} |"
            )
    
    return "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically compare OneNet and Graph predictions against benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--meddata-dir',
        type=str,
        default='ActDiseaseEL/meddata',
        help='Directory containing benchmark files (default: ActDiseaseEL/meddata)'
    )
    parser.add_argument(
        '--doid-results-dir',
        type=str,
        default='doid-results',
        help='Directory containing prediction files (default: doid-results)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='evaluation_results.md',
        help='Output markdown file (default: evaluation_results.md)'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Optional JSON output file for detailed metrics'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    args = parser.parse_args()
    
    # Convert to Path objects
    base_dir = Path(__file__).parent.parent
    meddata_dir = base_dir / args.meddata_dir
    doid_results_dir = base_dir / args.doid_results_dir
    
    if not meddata_dir.exists():
        print(f"Error: Meddata directory not found: {meddata_dir}")
        return
    
    if not doid_results_dir.exists():
        print(f"Error: Doid-results directory not found: {doid_results_dir}")
        return
    
    # Find matching files
    print("Finding matching benchmark and prediction files...")
    matches = find_matching_files(meddata_dir, doid_results_dir)
    
    if not matches:
        print("No matching files found!")
        return
    
    print(f"Found {len(matches)} benchmark files")
    
    # Process all matches
    all_results = {}
    
    for scenario_name, bench_path, onenet_path, graph_path in matches:
        if args.verbose:
            print(f"\nProcessing scenario: {scenario_name}")
            print(f"  Benchmark: {bench_path.name}")
            print(f"  OneNet: {onenet_path.name if onenet_path else 'NOT FOUND'}")
            print(f"  Graph: {graph_path.name if graph_path else 'NOT FOUND'}")
        
        all_results[scenario_name] = {}
        
        # Load benchmark
        try:
            ground_truth = load_jsonl(str(bench_path))
        except Exception as e:
            print(f"Error loading benchmark {bench_path}: {e}")
            continue
        
        # Check if benchmark has any labels
        has_labels = False
        for entry in ground_truth:
            if 'labels' in entry and entry['labels']:
                has_labels = True
                break
        
        if not has_labels:
            if args.verbose:
                print(f"  WARNING: {scenario_name} has no ground truth labels, skipping evaluation")
            continue
        
        # Process OneNet
        if onenet_path and onenet_path.exists():
            try:
                predictions = load_jsonl(str(onenet_path))
                metrics = compute_metrics(ground_truth, predictions)
                all_results[scenario_name]['onenet'] = metrics
                if args.verbose:
                    ner_f1 = metrics['ner']['exact_matching']['f1']
                    linking_f1 = metrics['linking']['exact_matching']['f1']
                    full_f1 = metrics['full']['exact_matching']['f1']
                    print(f"  OneNet - NER F1: {ner_f1:.4f}, Linking F1: {linking_f1:.4f}, Full F1: {full_f1:.4f}")
            except Exception as e:
                print(f"Error processing OneNet {onenet_path}: {e}")
        
        # Process Graph
        if graph_path and graph_path.exists():
            try:
                predictions = load_jsonl(str(graph_path))
                metrics = compute_metrics(ground_truth, predictions)
                all_results[scenario_name]['graph'] = metrics
                if args.verbose:
                    ner_f1 = metrics['ner']['exact_matching']['f1']
                    linking_f1 = metrics['linking']['exact_matching']['f1']
                    full_f1 = metrics['full']['exact_matching']['f1']
                    print(f"  Graph - NER F1: {ner_f1:.4f}, Linking F1: {linking_f1:.4f}, Full F1: {full_f1:.4f}")
            except Exception as e:
                print(f"Error processing Graph {graph_path}: {e}")
    
    # Generate markdown table
    print(f"\nGenerating markdown table...")
    markdown_table = generate_markdown_table(all_results)
    
    # Save markdown
    output_path = base_dir / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    print(f"Markdown table saved to: {output_path}")
    
    # Save JSON if requested
    if args.json_output:
        json_path = base_dir / args.json_output
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed metrics saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(markdown_table)


if __name__ == "__main__":
    main()
