#traverses the samples directory and combines all the samples into a single file

import os
import json
import argparse
from util import get_data_directory
from pathlib import Path 





# Example usage
if __name__ == "__main__":
    argparse = __import__('argparse')
    parser = argparse.ArgumentParser(description="Combine OpenPipe fine-tuning samples.")
    parser.add_argument("--input_path", default="batched-samples", type=str, help="Path to the directory that holds samples in jsonl format.")
    args = parser.parse_args()

    input_path = Path(get_data_directory("datasets", args.input_path))
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")
    
    sample_types = ["commands", "properties", "routines"]

    for type in sample_types:
        output_file = input_path / f"{type.upper()}_combined.jsonl"
        #now traverse the input directory and combine all jsonl files into a single file
        jsonl_data = []
        for jsonl_file in input_path.glob(f"sample_batch*_{type}.jsonl"):
            print(f"Processing file: {jsonl_file.name}")
            with jsonl_file.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        jsonl_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {jsonl_file.name}: {e}")   

        with output_file.open("w", encoding='utf-8') as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + "\n")

        print(f"✅ {len(jsonl_data)} entries saved to {output_file}")  
