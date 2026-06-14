#!/usr/bin/env python3
"""
Script để gộp chỉ dữ liệu instruction format thành file finetune_llm_data.jsonl
"""

import os
import json
from pathlib import Path

def merge_instruction_format_data():
    """Gộp tất cả dữ liệu instruction format thành một file"""
    
    # Đường dẫn thư mục gốc
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # File output
    output_file = data_dir / "finetune_llm_data.jsonl"
    
    # Tìm tất cả file instruction format
    input_files = []
    for data_folder in data_dir.glob("finetune_data*"):
        if data_folder.is_dir():
            for jsonl_file in data_folder.glob("*instruction_format.jsonl"):
                input_files.append(jsonl_file)
    
    print(f"🎯 Tìm thấy {len(input_files)} files instruction format:")
    for file_path in input_files:
        print(f"  - {file_path.relative_to(project_root)}")
    
    if not input_files:
        print("❌ Không tìm thấy file instruction format nào!")
        return
    
    # Gộp tất cả file
    total_lines = 0
    file_stats = {}
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            lines_in_file = 0
            print(f"\n📄 Đang xử lý: {input_file.name}")
            
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        try:
                            # Kiểm tra JSON hợp lệ và có đúng format
                            data = json.loads(line)
                            if 'instruction' in data and 'input' in data and 'output' in data:
                                outfile.write(line + '\n')
                                lines_in_file += 1
                                total_lines += 1
                            else:
                                print(f"⚠️  Bỏ qua dòng không đúng instruction format")
                        except json.JSONDecodeError:
                            print(f"⚠️  Bỏ qua dòng JSON không hợp lệ")
            
            file_stats[input_file.name] = lines_in_file
            print(f"✅ Đã gộp {lines_in_file:,} dòng")
    
    # Hiển thị kết quả
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH GỘP DỮ LIỆU INSTRUCTION FORMAT!")
    print("="*60)
    print(f"📊 Tổng số dòng: {total_lines:,}")
    print(f"📁 File output: {output_file.relative_to(project_root)}")
    print(f"💾 Kích thước: {file_size_mb:.2f} MB")
    
    print("\n📈 Chi tiết theo file:")
    for file_name, lines in file_stats.items():
        percentage = (lines / total_lines * 100) if total_lines > 0 else 0
        print(f"  • {file_name}: {lines:,} dòng ({percentage:.1f}%)")
    
    print("="*60)
    
    # Kiểm tra mẫu dữ liệu
    print("\n📋 Mẫu dữ liệu (3 dòng đầu):")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"\nDòng {i+1}:")
            print(f"  Instruction: {data['instruction'][:100]}...")
            print(f"  Input: {data['input'][:100]}...")
            print(f"  Output: {data['output'][:150]}...")

if __name__ == "__main__":
    merge_instruction_format_data()