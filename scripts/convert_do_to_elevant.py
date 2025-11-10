"""
Convert Human Disease Ontology (DO) to elevant format
This script parses DO OBO files and creates mappings compatible with elevant's entity linking system.

Example usage:
```bash
python scripts/convert_do_to_elevant.py /path/to/HumanDiseaseOntology/src/ontology/<ontology_name>.obo
```

```bash
# Specify custom data directory
python scripts/convert_do_to_elevant.py /path/to/doid.obo --data_directory /path/to/custom/data
```

## Using DO Entities with Elevant

When running `link_benchmark.py`, use the `-c` or `--custom_kb` flag:

```bash
python3 link_benchmark.py experiment_name -l bm25-db -b kore50 -c
```

This will use the custom mappings from `{data_directory}/custom-mappings/`.
"""
import argparse
import sys
import os
import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.append(".")

from elevant import settings
from elevant.utils import log


def parse_obo_file(obo_file: str) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    """
    Parse OBO file and extract:
    - entity_to_name: DOID -> name
    - entity_to_aliases: DOID -> list of synonyms
    - entity_to_types: DOID -> list of parent types (is_a relationships)
    - entity_to_def: DOID -> definition
    """
    logger = logging.getLogger("main." + __name__.split(".")[-1])
    
    entity_to_name = {}
    entity_to_aliases = defaultdict(list)
    entity_to_types = defaultdict(list)
    entity_to_def = {}
    
    current_term = None
    current_id = None
    current_name = None
    
    logger.info(f"Parsing OBO file: {obo_file}")
    
    with open(obo_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Start of new term
            if line == "[Term]":
                current_term = {}
                current_id = None
                current_name = None
                continue
            
            # End of term (empty line or next [Term])
            if not line and current_id:
                # Save the term
                if current_id and current_name:
                    entity_to_name[current_id] = current_name
                    # Add name as primary alias
                    entity_to_aliases[current_id].append(current_name)
                current_term = None
                current_id = None
                current_name = None
                continue
            
            # Parse term fields
            if line.startswith("id: "):
                current_id = line[4:].strip()
            elif line.startswith("name: "):
                current_name = line[6:].strip()
            elif line.startswith("synonym: "):
                # Extract synonym text (format: "text" TYPE [xrefs]
                match = re.match(r'synonym:\s+"([^"]+)"', line)
                if match:
                    synonym = match.group(1)
                    if current_id and synonym not in entity_to_aliases[current_id]:
                        entity_to_aliases[current_id].append(synonym)
            elif line.startswith("alt_id: "):
                # Alternative IDs are also aliases
                alt_id = line[8:].strip()
                if current_id and alt_id:
                    # Map alt_id to same name/aliases as main ID
                    if current_name:
                        entity_to_name[alt_id] = current_name
                        entity_to_aliases[alt_id].append(current_name)
            elif line.startswith("is_a: "):
                # Extract parent type (format: DOID:xxxxx ! type_name)
                parent = line[6:].strip()
                # Extract DOID from parent
                match = re.match(r'(DOID:\d+)', parent)
                if match:
                    parent_id = match.group(1)
                    if current_id:
                        entity_to_types[current_id].append(parent_id)
            elif line.startswith("def: "):
                # Extract definition text
                match = re.match(r'def:\s+"([^"]+)"', line)
                if match:
                    definition = match.group(1)
                    if current_id:
                        entity_to_def[current_id] = definition
    
    # Handle last term if file doesn't end with newline
    if current_id and current_name:
        entity_to_name[current_id] = current_name
        entity_to_aliases[current_id].append(current_name)
    
    logger.info(f"Parsed {len(entity_to_name)} entities from OBO file")
    return entity_to_name, dict(entity_to_aliases), dict(entity_to_types), entity_to_def


def create_custom_mappings(entity_to_name: Dict[str, str],
                          entity_to_aliases: Dict[str, List[str]],
                          entity_to_types: Dict[str, List[str]],
                          output_dir: str):
    """Create custom mapping files for elevant"""
    logger = logging.getLogger("main." + __name__.split(".")[-1])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Entity to name mapping
    entity_to_name_file = os.path.join(output_dir, "entity_to_name.tsv")
    logger.info(f"Writing entity to name mappings to {entity_to_name_file}")
    with open(entity_to_name_file, 'w', encoding='utf-8') as f:
        for entity_id, name in sorted(entity_to_name.items()):
            f.write(f"{entity_id}\t{name}\n")
    logger.info(f"Wrote {len(entity_to_name)} entity to name mappings")
    
    # 2. Entity to types mapping
    # For DO, we'll use "Disease" as the main type, plus parent types
    entity_to_types_file = os.path.join(output_dir, "entity_to_types.tsv")
    logger.info(f"Writing entity to types mappings to {entity_to_types_file}")
    with open(entity_to_types_file, 'w', encoding='utf-8') as f:
        for entity_id in sorted(entity_to_name.keys()):
            types = ["Disease"]  # All DO entities are diseases
            # Add parent types if available
            if entity_id in entity_to_types:
                types.extend(entity_to_types[entity_id])
            f.write(f"{entity_id}\t" + "\t".join(types) + "\n")
    logger.info(f"Wrote {len(entity_to_name)} entity to types mappings")
    
    # 3. Whitelist types (just "Disease" for now, but could include parent types)
    whitelist_types_file = os.path.join(output_dir, "whitelist_types.tsv")
    logger.info(f"Writing whitelist types to {whitelist_types_file}")
    all_types = {"Disease": "Disease"}
    # Collect all parent types
    for types_list in entity_to_types.values():
        for parent_type in types_list:
            if parent_type in entity_to_name:
                all_types[parent_type] = entity_to_name[parent_type]
    
    with open(whitelist_types_file, 'w', encoding='utf-8') as f:
        for type_id, type_name in sorted(all_types.items()):
            f.write(f"{type_id}\t{type_name}\n")
    logger.info(f"Wrote {len(all_types)} whitelist types")
    
    return entity_to_name_file, entity_to_types_file, whitelist_types_file


def create_database_files(entity_to_name: Dict[str, str],
                         entity_to_aliases: Dict[str, List[str]],
                         output_dir: str,
                         data_dir: str):
    """Create LMDB database files similar to Wikipedia format"""
    logger = logging.getLogger("main." + __name__.split(".")[-1])
    
    # Create wikidata-mappings directory in data directory
    mappings_dir = os.path.join(data_dir, "wikidata-mappings")
    os.makedirs(mappings_dir, exist_ok=True)
    
    # 1. DOID to label (name) - similar to qid_to_label.db
    doid_to_label_file = os.path.join(output_dir, "doid_to_label.tsv")
    logger.info(f"Creating DOID to label mapping: {doid_to_label_file}")
    with open(doid_to_label_file, 'w', encoding='utf-8') as f:
        for entity_id, name in sorted(entity_to_name.items()):
            f.write(f"{entity_id}\t{name}\n")
    
    # Create database (use qid_to_label.db name that system expects)
    output_db = os.path.join(mappings_dir, "qid_to_label.db")
    logger.info(f"Creating database: {output_db}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_db_script = os.path.join(script_dir, "create_databases.py")
    os.system(f"cd {project_root} && PYTHONPATH=src python3 {create_db_script} {doid_to_label_file} -o {output_db}")
    
    # 2. Label to DOIDs (reverse) - similar to label_to_qids.db
    label_to_doids_file = os.path.join(output_dir, "label_to_doids.tsv")
    logger.info(f"Creating label to DOIDs mapping: {label_to_doids_file}")
    label_to_doids = defaultdict(set)
    for entity_id, name in entity_to_name.items():
        label_to_doids[name].add(entity_id)
    
    with open(label_to_doids_file, 'w', encoding='utf-8') as f:
        for label, doids in sorted(label_to_doids.items()):
            f.write(f"{label}\t{','.join(sorted(doids))}\n")
    
    output_db = os.path.join(mappings_dir, "label_to_qids.db")
    logger.info(f"Creating database: {output_db}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_db_script = os.path.join(script_dir, "create_databases.py")
    os.system(f"cd {project_root} && PYTHONPATH=src python3 {create_db_script} {label_to_doids_file} -f multiple_values -o {output_db}")
    
    # 3. DOID to aliases - similar to qid_to_aliases.db
    doid_to_aliases_file = os.path.join(output_dir, "doid_to_aliases.tsv")
    logger.info(f"Creating DOID to aliases mapping: {doid_to_aliases_file}")
    with open(doid_to_aliases_file, 'w', encoding='utf-8') as f:
        for entity_id in sorted(entity_to_name.keys()):
            aliases = entity_to_aliases.get(entity_id, [])
            # Remove the primary name from aliases (it's already in name mapping)
            aliases = [a for a in aliases if a != entity_to_name.get(entity_id, "")]
            if aliases:
                f.write(f"{entity_id}\t{';'.join(aliases)}\n")
    
    output_db = os.path.join(mappings_dir, "qid_to_aliases.db")
    logger.info(f"Creating database: {output_db}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_db_script = os.path.join(script_dir, "create_databases.py")
    os.system(f"cd {project_root} && PYTHONPATH=src python3 {create_db_script} {doid_to_aliases_file} -f multiple_values_semicolon_separated -o {output_db}")
    
    # 4. Alias to DOIDs (reverse) - similar to alias_to_qids.db
    alias_to_doids_file = os.path.join(output_dir, "alias_to_doids.tsv")
    logger.info(f"Creating alias to DOIDs mapping: {alias_to_doids_file}")
    alias_to_doids = defaultdict(set)
    for entity_id, aliases in entity_to_aliases.items():
        for alias in aliases:
            alias_to_doids[alias].add(entity_id)
    
    with open(alias_to_doids_file, 'w', encoding='utf-8') as f:
        for alias, doids in sorted(alias_to_doids.items()):
            f.write(f"{alias}\t{','.join(sorted(doids))}\n")
    
    output_db = os.path.join(mappings_dir, "alias_to_qids.db")
    logger.info(f"Creating database: {output_db}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_db_script = os.path.join(script_dir, "create_databases.py")
    os.system(f"cd {project_root} && PYTHONPATH=src python3 {create_db_script} {alias_to_doids_file} -f multiple_values -o {output_db}")
    
    # 5. DOID to sitelinks (popularity) - use a simple metric (e.g., 1 for all, or based on hierarchy depth)
    doid_to_sitelinks_file = os.path.join(output_dir, "doid_to_sitelinks.tsv")
    logger.info(f"Creating DOID to sitelinks mapping: {doid_to_sitelinks_file}")
    with open(doid_to_sitelinks_file, 'w', encoding='utf-8') as f:
        for entity_id in sorted(entity_to_name.keys()):
            # Use number of aliases as a simple popularity metric
            popularity = len(entity_to_aliases.get(entity_id, []))
            f.write(f"{entity_id}\t{popularity}\n")
    
    output_db = os.path.join(mappings_dir, "qid_to_sitelinks.db")
    logger.info(f"Creating database: {output_db}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    create_db_script = os.path.join(script_dir, "create_databases.py")
    os.system(f"cd {project_root} && PYTHONPATH=src python3 {create_db_script} {doid_to_sitelinks_file} -o {output_db}")
    
    logger.info("Database files created successfully")


def main(args):
    logger = log.setup_logger(sys.argv[0])
    
    # Parse OBO file
    entity_to_name, entity_to_aliases, entity_to_types, entity_to_def = parse_obo_file(args.obo_file)
    
    # Create custom mappings
    custom_mappings_dir = os.path.join(args.data_directory, "custom-mappings")
    entity_to_name_file, entity_to_types_file, whitelist_types_file = create_custom_mappings(
        entity_to_name, entity_to_aliases, entity_to_types, custom_mappings_dir
    )
    
    # Create database files
    temp_dir = os.path.join(args.data_directory, "temp-do-mappings")
    os.makedirs(temp_dir, exist_ok=True)
    create_database_files(entity_to_name, entity_to_aliases, temp_dir, args.data_directory)
    
    logger.info(f"\nConversion complete!")
    logger.info(f"Custom mappings created in: {custom_mappings_dir}")
    logger.info(f"Database files created in: {os.path.join(args.data_directory, 'wikidata-mappings')}")
    logger.info(f"\nTo use with elevant, set custom_kb=True when running link_benchmark.py")
    logger.info(f"Or update settings to point to the custom mappings directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    
    parser.add_argument("obo_file", type=str,
                        help="Path to DO OBO file (e.g., doid.obo or HumanDO.obo)")
    parser.add_argument("--data_directory", type=str, default=None,
                        help="Data directory where custom mappings and databases will be created. "
                             "Defaults to DATA_DIRECTORY from elevant.config.json")
    
    logger = log.setup_logger(sys.argv[0])
    logger.debug(' '.join(sys.argv))
    
    args = parser.parse_args()
    
    # Get data directory from settings if not provided
    if args.data_directory is None:
        args.data_directory = settings.DATA_DIRECTORY.rstrip("/")
    
    main(args)

