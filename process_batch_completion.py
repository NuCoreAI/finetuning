#This file waits for a batch of requests to be completed and then processes them using the OpenAI batch endpoint. 


from openai import OpenAI
import json, os, tempfile
from pathlib import Path
from nucore import NuCore
from util import get_data_directory
from typing import Literal, List


# === CONFIGURATION ===
SECRETS_DIR = Path(get_data_directory("secrets", None))
if not SECRETS_DIR.exists():
    raise FileNotFoundError(f"Secrets directory {SECRETS_DIR} does not exist. Please create it and add your OpenAI API key.")
# Load the OpenAI API key from the secrets file
if not (SECRETS_DIR / "keys.py").exists():
    raise FileNotFoundError(f"Secrets file {SECRETS_DIR / 'keys.py'} does not exist. Please create it and add your OpenAI API key.")

exec(open(SECRETS_DIR / "keys.py").read())  # This will set OPENAI_API_KEY  

PROMPTS_DIR = Path(get_data_directory("prompts", None))
if not PROMPTS_DIR.exists():
    raise FileNotFoundError(f"Prompt directory {PROMPTS_DIR} does not exist. Please use git clone to get everything." ) 


#MODEL = "gpt-4o"  # Use the latest model available
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0
ENDPOINT = "/v1/chat/completions" # you can also use /v1/embeddings, /v1/responses, etc.
COMPLETION_WINDOW = "24h"         # 24h or 4h depending on availability in your account
BATCH_MAX_LINES_PER_REQUEST = 1800  # max lines per batch request for OpenAI 


def list_batches(client:OpenAI):
    """
    List all batches and their statuses.
    """
    batches = client.batches.list()
    for batch in batches.data:
        print(f"Batch ID: {batch.id}, Status: {batch.status}, Created: {batch.created_at}, Completed: {batch.completed_at}")
    return batches.data 

def make_and_save_batch(client:OpenAI, batch_num:int, lines: List[dict], batched_requests_dir: Path)->str:
    """
    Save a batch of lines to a temporary jsonl file and return the path.
    """
    print(f"Saving batch {batch_num} of {len(lines)} lines to temporary file...")
    jsonl_path = batched_requests_dir / f"batch_{batch_num}.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error saving batch {batch_num}: {e}")
        return None
    print(f"Batch saved to {jsonl_path}")

    up = client.files.create(file=jsonl_path.open("rb"), purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id = up.id,
        endpoint      = ENDPOINT,          # must match the "url" used in each line
        completion_window = COMPLETION_WINDOW
    )
    batch_id = batch.id
    #now rename the request file to include the batch id
    new_jsonl_path = batched_requests_dir / f"batch_{batch_num}_{batch_id}.jsonl"
    os.rename(jsonl_path, new_jsonl_path)
    jsonl_path = new_jsonl_path
    print(f"Created batch: {jsonl_path} with id: {batch_id}")
    return jsonl_path, batch_id


# Example usage
if __name__ == "__main__":
    argparse = __import__('argparse')
    parser = argparse.ArgumentParser(description="Process batch completions (cancel, wait, list).")
    parser.add_argument("--output_path", type=str, help="Path to the output directory where the samples are stored. If none given, it will be printed to stdout.")
    parser.add_argument("--types", type=str, help="Type of training: properties, commands, routines, general.")
    parser.add_argument("--operation", type=str, help="Operation to perform on the batches: cancel, wait, list.")

    args = parser.parse_args()

    types = args.types.split(",") if args.types else ["properties", "commands"]
    operation = args.operation if args.operation else "list"


    REFERENCE_DIR = Path(get_data_directory("customer_data", None))
    BATCHED_REQUESTS_DIR = Path(get_data_directory("datasets", "batched-requests"))
    OUTPUT_DIR = Path(get_data_directory("datasets", "batched-samples"))

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"Output path {output_path} does not exist or is not a directory.")

    for type in types:
        type=type.strip()
        client = OpenAI(api_key=globals()[f"OPENAI_API_KEY_BATCH"])  # or use environment variable
        try:
            if operation == "cancel":
                print(f"Cancelling all batches for {type}...")
                batches = list_batches(client)
                for batch in batches:
                    try:
                        client.batches.cancel(batch.id)
                        print(f"Cancelled batch {batch.id}")
                    except Exception as e:
                        print(f"Error cancelling batch {batch.id}: {e}")
            elif operation == "wait":
                print(f"Waiting for all batches to complete for {type}...")
                batches = list_batches(client)
                for batch in batches:
                    if batch.status != "completed":
                        print(f"Waiting for batch {batch.id} to complete...")
                        completed_batch = client.batches.wait(batch.id, max_wait=3600)  # wait up to 1 hour
                        print(f"Batch {batch.id} completed with status: {completed_batch.status}")
                    else:
                        print(f"Batch {batch.id} is already completed.")
            elif operation == "list":
                print(f"Listing all batches for {type}...")
                list_batches(client)
        except Exception as e:
            print(f"Error processing batch for {type}. Skipping: {e}")
            continue

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
