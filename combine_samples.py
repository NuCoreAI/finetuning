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
    parser.add_argument("--input_path", type=str, help="Path to the directory that holds samples in jsonl format.")
    parser.add_argument("--output_file", type=str, help="Path to the output file. If not specified, defaults to input directory name with '_combined.jsonl' suffix.")
    args = parser.parse_args()

    INPUT_DIR = Path(get_data_directory("datasets", "devices"))

    input_path = Path(args.input_path) if args.input_path else INPUT_DIR
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")

    output_file = Path(args.output_file) if args.output_file else (input_path / f"{input_path.name}_combined.jsonl")
    if not output_file.parent.exists():
        raise ValueError(f"Output directory {output_file.parent} does not exist. Please create it first.")

    #now traverse the input directory and combine all jsonl files into a single file
    jsonl_data = []
    for jsonl_file in input_path.glob("*.jsonl"):
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
