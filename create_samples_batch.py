#This file creates samples that bake in the system prompt into the model.
#This way, we do not use unnecessary tokens during inference.


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
COMPLETION_WINDOW = "24h"          # 24h or 4h depending on availability in your account
BATCH_MAX_LINES_PER_REQUEST = 900  # max lines per batch request for OpenAI


TRAIN_PROMPT = ""
RUN_PROMPT = ""

def setup_prompts(type: Literal["properties", "commands", "routines", "general"]):
    global TRAIN_PROMPT, RUN_PROMPT
    TRAIN_PROMPT = ""
    RUN_PROMPT = ""
    try:
        with open(os.path.join(PROMPTS_DIR, f"{type}.prompt.train"), "r") as f:
            TRAIN_PROMPT = f.read()
    except:
        raise ValueError(f"Failed to load training prompt for type {type}.")

    try:
        with open(os.path.join(PROMPTS_DIR, f"{type}.prompt.run"), "r") as f:
            RUN_PROMPT = f.read().replace("\n", "\\n")
    except:
        raise ValueError(f"Failed to load run prompt for type {type}.")

    ##Now, replace {{NUCORE_BASICS}} in SYSTEM_PROMPT with RUNTIME_SYSTEM_PROMPT
    TRAIN_PROMPT = TRAIN_PROMPT.replace("{{TEMPLATE_PROMPTS_RUNTIME}}", f"{RUN_PROMPT}")

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

def generate_request(full_text, request_id, type, dump=True):

    if full_text: 
        # replace <device_info> in the system prompt with the actual device info
        system_prompt = TRAIN_PROMPT.replace("{{DEVICE_STRUCTURE}}", full_text)
        return {
            "custom_id": request_id,             # must be unique (string)
            "method": "POST",
            "url": ENDPOINT,
            "body": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "messages": [
                    {"role": "system", "content": system_prompt},
                ],
            # add any other Chat Completions params you need:
            # "response_format": {"type": "json_object"}
            }
        }
    return None

# Example usage
if __name__ == "__main__":
    argparse = __import__('argparse')
    parser = argparse.ArgumentParser(description="Generate OpenPipe fine-tuning entries from device descriptions.")
    parser.add_argument("--input_path", type=str, help="Path to the directory that holds profiles and nodes directories within. If none given, it will use the default references directory.")
    parser.add_argument("--output_path", type=str, help="Path to the output directory where flattened structures are stored. If none given, it will be printed to stdout.")
    parser.add_argument("--types", type=str, help="Type of training: properties, commands, routines, general.")
    args = parser.parse_args()

    types = args.types.split(",") if args.types else ["properties", "commands"]


    REFERENCE_DIR = Path(get_data_directory("customer_data", None))
    BATCHED_REQUESTS_DIR = Path(get_data_directory("datasets", "batched-requests"))
    OUTPUT_DIR = Path(get_data_directory("datasets", "batched-samples"))

    input_path = Path(args.input_path) if args.input_path else REFERENCE_DIR
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"Output path {output_path} does not exist or is not a directory.")

    # now traverse the input directory where you will find profiles and nodes directories
    # start with files in the nodes directory and then use the name of the file (without extension) to find the corresponding profile in the profiles directory
    nodes_dir = input_path / "nodes"
    profiles_dir = input_path / "profiles"
    if not nodes_dir.exists() or not nodes_dir.is_dir():
        raise ValueError(f"Nodes directory {nodes_dir} does not exist or is not a directory.")
    if not profiles_dir.exists() or not profiles_dir.is_dir():
        raise ValueError(f"Profiles directory {profiles_dir} does not exist or is not a directory.")

    batch_lines = []
    batch_num  = 1

    client = OpenAI(api_key=globals()[f"OPENAI_API_KEY"])  # or use environment variable
    for type in types:
        type=type.strip()
        setup_prompts(type)
        for node_file in nodes_dir.glob("*.xml"):
            profile_file = profiles_dir / (f"{node_file.stem}.json").replace("nodes-", "profile-")
            out_file = f"{node_file.stem}_finetune"
            
            print(f"Processing node: {node_file.name} with profile: {profile_file.name}")
            if not profile_file.exists():
                print(f"Warning: Profile file {profile_file} does not exist for node {node_file}. Skipping.")
                continue
            nuCore = NuCore(collection_path="/tmp/nucore.finetuner", collection_name="finetuner", backend_url="http://localhost:8000", backend_username="admin", backend_password="admin"
            )
            try:
                nuCore.load(include_rag_docs=False, profile_path=profile_file, nodes_path=node_file)
            except Exception as e:
                print(f"Error loading NuCore with profile {profile_file} and node {node_file}. Skipping: {e}")
                continue
            try:
                rag = nuCore.format_nodes()
            except Exception as e:
                print(f"Error formatting nodes for profile {profile_file} and node {node_file}. Skipping: {e}")
                continue
            
            try:
                if not rag:
                    print(f"Warning: No RAG documents found for node {node_file}. Skipping.")
                    continue
                rag_docs = rag["documents"]
                if not rag_docs:
                    print(f"Warning: No documents found in RAG for node {node_file}. Skipping.")
                    continue
                for i in range(0, len(rag_docs), 3):
                    batch = rag_docs[i:i+3]
                    full_text = "".join(batch)
                    if out_file:
                        request_id = f"{out_file}_{i//3 + 1}_{type}"
                        print(f"Writing to {request_id}")
                        request = generate_request(full_text, request_id, type, dump=True)
                        if request:
                            batch_lines.append(request)
                            if len(batch_lines) >= BATCH_MAX_LINES_PER_REQUEST:
                                make_and_save_batch(client, batch_num, batch_lines, BATCHED_REQUESTS_DIR)
                                batch_lines = []
                                batch_num += 1
                #save any remaining lines
                if batch_lines:
                    make_and_save_batch(client, batch_num, batch_lines, BATCHED_REQUESTS_DIR)
                    batch_lines = []
                    batch_num += 1

            except Exception as e:
                print(f"Error processing RAG documents for node {node_file}. Skipping: {e}")
                continue    
                        
                    
                    
     #EXAMPLES.append((node_data, profile_data))

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
