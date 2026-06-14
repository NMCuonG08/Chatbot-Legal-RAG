#!/usr/bin/env python3
"""
Script to process, clean, and deduplicate merged dataset for Llama fine-tuning.
"""

import json
from pathlib import Path
import os
import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_llama_data():
    script_dir = Path(__file__).parent.resolve()
    
    # Check paths in order of preference
    possible_paths = [
        script_dir / "raw_data" / "finetune_llm_data.jsonl",
        script_dir.parent.parent / "data_pipeline" / "data" / "finetune_llm_data.jsonl"
    ]
    
    input_file = None
    for path in possible_paths:
        if path.exists():
            input_file = path
            break
            
    if not input_file:
        print("❌ Merged dataset not found! Checked locations:")
        for p in possible_paths:
            print(f"  - {p}")
        print("Please run data gộp first: python data_pipeline/utils/merge_instruction_data.py")
        return

    output_file = script_dir / "processed_llama_data.jsonl"
    print(f"🧹 Processing data from {input_file} to {output_file}...")

    raw_count = 0
    unique_records = {}
    invalid_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw_count += 1
            try:
                item = json.loads(line)
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output", "")
                
                # Check for empty fields
                if not instruction or (not inp and not output):
                    invalid_count += 1
                    continue
                
                # Clean text fields
                cleaned_instruction = clean_text(str(instruction))
                cleaned_input = clean_text(str(inp))
                cleaned_output = clean_text(str(output))
                
                # Create deduplication key based on instruction and input
                dedup_key = (cleaned_instruction, cleaned_input)
                
                # Save first occurrence or keep the longest output
                if dedup_key not in unique_records:
                    unique_records[dedup_key] = {
                        "instruction": cleaned_instruction,
                        "input": cleaned_input,
                        "output": cleaned_output
                    }
                else:
                    if len(cleaned_output) > len(unique_records[dedup_key]["output"]):
                        unique_records[dedup_key]["output"] = cleaned_output
            except json.JSONDecodeError:
                invalid_count += 1
                continue

    # Write processed records
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for record in unique_records.values():
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Data processing complete:")
    print(f"  - Raw records: {raw_count:,}")
    print(f"  - Invalid/empty records filtered: {invalid_count:,}")
    print(f"  - Duplicate records removed: {raw_count - len(unique_records) - invalid_count:,}")
    print(f"  - Final processed records saved: {len(unique_records):,}")

    # Generate a small sample file for debugging
    sample_file = script_dir / "sample_processed_data.json"
    sample_size = min(5, len(unique_records))
    sample_data = list(unique_records.values())[:sample_size]
    with open(sample_file, 'w', encoding='utf-8') as fsample:
        json.dump(sample_data, fsample, indent=2, ensure_ascii=False)
    print(f"  - Sample output saved to: {sample_file}")

if __name__ == "__main__":
    process_llama_data()
