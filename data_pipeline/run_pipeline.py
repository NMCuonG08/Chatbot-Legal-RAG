#!/usr/bin/env python3
"""
Script orchestration: Chạy toàn bộ pipeline xử lý dữ liệu cho Legal RAG
"""
import subprocess
import sys
from pathlib import Path

def run_ingest():
    print("\n=== Bước 1: Ingest dữ liệu gốc ===")
    script_path = Path(__file__).parent / "ingest_data.py"
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print("❌ Lỗi ingest dữ liệu!")
        sys.exit(1)
    print("✅ Đã ingest dữ liệu!")

def run_transform():
    print("\n=== Bước 2: Transform dữ liệu sang QA format ===")
    script_path = Path(__file__).parent / "transform_data.py"
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print("❌ Lỗi transform dữ liệu!")
        sys.exit(1)
    print("✅ Đã transform dữ liệu!")

def run_merge_instruction():
    print("\n=== Bước 3: Gộp dữ liệu instruction format ===")
    script_path = Path(__file__).parent / "utils" / "merge_instruction_data.py"
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        print("❌ Lỗi khi gộp dữ liệu instruction format!")
        sys.exit(1)
    print("✅ Đã gộp dữ liệu instruction format!")

def main():
    print("BẮT ĐẦU CHẠY DATA PIPELINE CHO LEGAL RAG\n" + "="*60)
    run_ingest()
    run_transform()
    run_merge_instruction()
    print("\n🎉 HOÀN THÀNH TOÀN BỘ DATA PIPELINE!\n" + "="*60)

if __name__ == "__main__":
    main()
