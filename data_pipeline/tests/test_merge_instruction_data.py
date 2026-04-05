import unittest
import json
from pathlib import Path
from utils import merge_instruction_data

class TestMergeInstructionData(unittest.TestCase):
    def test_output_file_exists(self):
        data_dir = Path(__file__).parent.parent / "data"  # data_pipeline/data
        output_file = data_dir / "finetune_llm_data.jsonl"
        # Chạy hàm merge
        merge_instruction_data.merge_instruction_format_data()
        self.assertTrue(output_file.exists(), "Output file không tồn tại sau khi merge!")

    def test_output_file_format(self):
        data_dir = Path(__file__).parent.parent / "data"
        output_file = data_dir / "finetune_llm_data.jsonl"
        merge_instruction_data.merge_instruction_format_data()
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 10:
                    break
                data = json.loads(line)
                self.assertIn('instruction', data)
                self.assertIn('input', data)
                self.assertIn('output', data)

if __name__ == "__main__":
    unittest.main()
