#!/usr/bin/env python3
"""
Script to analyze the structure and quality of the fine-tuning dataset.
"""

import json
import os
from pathlib import Path
import numpy as np

def analyze_dataset():
    # Paths relative to script directory
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
        print("❌ Dataset not found! Checked locations:")
        for p in possible_paths:
            print(f"  - {p}")
        print("Please run data gộp first: python data_pipeline/utils/merge_instruction_data.py")
        return
        
    print(f"🔍 Analyzing dataset: {input_file}")
    
    records = []
    invalid_count = 0
    missing_fields = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'instruction' in data and 'input' in data and 'output' in data:
                    records.append(data)
                else:
                    invalid_count += 1
                    missing = [field for field in ['instruction', 'input', 'output'] if field not in data]
                    missing_fields.append((line_idx, missing))
            except json.JSONDecodeError:
                invalid_count += 1
                missing_fields.append((line_idx, "Invalid JSON"))
                
    total_records = len(records)
    print(f"📊 Total valid records found: {total_records:,}")
    if invalid_count > 0:
        print(f"⚠️  Total invalid records/lines: {invalid_count:,}")
        for idx, err in missing_fields[:5]:
            print(f"   Line {idx}: {err}")
        if len(missing_fields) > 5:
            print(f"   ... and {len(missing_fields) - 5} more errors")
            
    if total_records == 0:
        print("❌ No valid records to analyze.")
        return
        
    # Stats lists
    inst_lens = []
    input_lens = []
    output_lens = []
    total_lens = []
    
    for r in records:
        inst_len = len(r['instruction'])
        inp_len = len(r['input'])
        out_len = len(r['output'])
        
        inst_lens.append(inst_len)
        input_lens.append(inp_len)
        output_lens.append(out_len)
        total_lens.append(inst_len + inp_len + out_len)
        
    print("\n" + "="*50)
    print("📏 CHARACTER LENGTH STATISTICS (min / max / mean / median)")
    print("="*50)
    print(f"  Instruction: {min(inst_lens)} / {max(inst_lens)} / {np.mean(inst_lens):.1f} / {np.median(inst_lens):.1f}")
    print(f"  Input:       {min(input_lens)} / {max(input_lens)} / {np.mean(input_lens):.1f} / {np.median(input_lens):.1f}")
    print(f"  Output:      {min(output_lens)} / {max(output_lens)} / {np.mean(output_lens):.1f} / {np.median(output_lens):.1f}")
    print(f"  Total:       {min(total_lens)} / {max(total_lens)} / {np.mean(total_lens):.1f} / {np.median(total_lens):.1f}")
    
    # Word count estimation (approximate tokens)
    inst_words = [len(r['instruction'].split()) for r in records]
    input_words = [len(r['input'].split()) for r in records]
    output_words = [len(r['output'].split()) for r in records]
    total_words = [inst_words[i] + input_words[i] + output_words[i] for i in range(total_records)]
    
    print("\n" + "="*50)
    print("📝 ESTIMATED WORD COUNT STATISTICS")
    print("="*50)
    print(f"  Instruction: {min(inst_words)} / {max(inst_words)} / {np.mean(inst_words):.1f} / {np.median(inst_words):.1f}")
    print(f"  Input:       {min(input_words)} / {max(input_words)} / {np.mean(input_words):.1f} / {np.median(input_words):.1f}")
    print(f"  Output:      {min(output_words)} / {max(output_words)} / {np.mean(output_words):.1f} / {np.median(output_words):.1f}")
    print(f"  Total:       {min(total_words)} / {max(total_words)} / {np.mean(total_words):.1f} / {np.median(total_words):.1f}")
    print("="*50)

if __name__ == "__main__":
    analyze_dataset()
