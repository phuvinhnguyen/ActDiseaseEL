"""
Text processor script that combines .txt files by journal / language.

usage:
    python text_process.py -j <input_folder>  # by journal
    python text_process.py -l <input_folder>  # by language
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

JOURNAL_MAPPER = {
    "BMJ": "bmj",
    "The_Diabetic_Journal": "tdj",
    "allergia": "allergia",
    "Diabetes": "diabetes",
    "Der_Allergiker": "da",
    "DAAB_Bericht": "daab",
    "journal_des_diabetiques": "jdd",
    "courrier_sclerose_en_plaques": "csep",
    "faire_face_berry": "ffb",
    "faire_face": "ff",
}

LANGUAGE_MAPPER = {
    "BMJ": "en",
    "The_Diabetic_Journal": "en",
    "allergia": "sv",
    "Diabetes": "sv",
    "Der_Allergiker": "de",
    "DAAB_Bericht": "de",
    "journal_des_diabetiques": "fr",
    "courrier_sclerose_en_plaques": "fr",
    "faire_face_berry": "fr",
    "faire_face": "fr",
}

LANGUAGE_NAMES = {
    "en": "english",
    "sv": "swedish",
    "de": "german",
    "fr": "french",
}


def strip_markdown(text: str) -> str:
    """
    remove markdown and escape sequences from text.
    """
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"^\[[^\]]+\]:\s*\S+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text


def identify_journal_prefix(filename: str) -> str | None:
    """
    Identify which journal prefix a filename starts with.
    Returns the prefix key or None if not found.
    """
    filename_lower = filename.lower()
    
    sorted_prefixes = sorted(JOURNAL_MAPPER.keys(), key=len, reverse=True)
    
    for prefix in sorted_prefixes:
        prefix_lower = prefix.lower()
        if filename_lower.startswith(prefix_lower + "_") or filename_lower == prefix_lower + ".txt":
            return prefix
    
    return None


def process_files(input_folder: str, mode: str) -> None:
    """
    Process all .txt files in the input folder.
    mode: 'journal' or 'language'
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: {input_folder} does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: {input_folder} is not a folder")
        return
    
    # create output folder with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "journal":
        output_folder = input_path / f"per_journal_{timestamp}"
    else:
        output_folder = input_path / f"per_language_{timestamp}"
    
    output_folder.mkdir(exist_ok=True)
    
    collected_texts = defaultdict(list)
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print("no .txt files found in '{input_folder}'.")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for txt_file in txt_files:
        filename = txt_file.name
        
        prefix = identify_journal_prefix(filename)
        
        if prefix is None:
            print(f"- skipping {filename} (prefix error).")
            skipped_count += 1
            continue
        
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"- skipping '{filename}. Error: {e}.")
            skipped_count += 1
            continue
        
        # remove markdown and make it a single line
        processed_text = strip_markdown(content)
        
        if not processed_text:
            print(f"- skipping {filename}.")
            skipped_count += 1
            continue
        
        if mode == "journal":
            output_key = JOURNAL_MAPPER[prefix]
        else:  # language mode
            lang_code = LANGUAGE_MAPPER[prefix]
            output_key = LANGUAGE_NAMES.get(lang_code, lang_code)
        
        collected_texts[output_key].append(processed_text)
        processed_count += 1
        print(f"- processed: {filename} -> {output_key}")
    
    # create output files
    for output_key, texts in collected_texts.items():
        output_file = output_folder / f"{output_key}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        
        print(f"- created: {output_file.name} ({len(texts)} entries)")
    
    print(f"\n{'='*50}")
    print(f"processing complete!")
    print(f"Saved in: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="process and combine .txt files by journal / language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            examples:
                python text_process.py -j ./<articles_as_txt>    # by journal
                python text_process.py -l ./<articles_as_txt>    # by language
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-j",
        action="store_true",
        help="By journal abbreviation"
    )
    group.add_argument(
        "-l",
        action="store_true",
        help="By language"
    )
    
    parser.add_argument(
        "input_folder",
        help="path to folder where .txt files to process exists"
    )
    
    args = parser.parse_args()
    
    if args.journal:
        mode = "journal"
    else:
        mode = "language"
    
    process_files(args.input_folder, mode)


if __name__ == "__main__":
    main()
