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
    after = None
    while True:
        page = client.batches.list(limit=100, after=after)
        data = page.data or []
        if not data:
            break
        for batch in data:
            yield batch
        # The SDK returns items sorted newest→oldest; use last id as cursor
        after = data[-1].id
        if not page.has_more:
            break

def download_result(client:OpenAI, batch, path):
    """Download a Files API asset to disk."""
    try:
#        row = [
#            batch.id,
#            batch.status,
#            batch.endpoint,
#            batch.input_file_id,
#            batch.output_file_id or "-",
#            batch.error_file_id or "-",
#            batch.completion_window
#        ]
        out_path=""
        output_file_id=""
        error=False
        if getattr(batch, "output_file_id", None):
            out_path = path / f"{batch.id}_output.jsonl"
            output_file_id=batch.output_file_id
        elif getattr(batch, "error_file_id", None):
            out_path = path / f"{batch.id}_error.jsonl"
            output_file_id=batch.error_file_id
            error=True
        else:
            raise ValueError("Neitehr output file id nor error_file id returned any content")

        contents=[]    
        #download if and only if necessary
        if not out_path.exists():  # avoid re-downloading
            try:
                print (f"downloading {out_path} ...")
                full_contents = client.files.content(output_file_id).text
                out_path.write_text(full_contents, encoding="utf-8")
                contents =  full_contents.strip().splitlines()
                for content in contents:
                    content = json.loads(content)
                    samples_out_path = path / f"sample_{out_path.stem}_{content['custom_id']}.jsonl"
                    print (f"saving {samples_out_path} ...")
                    samples_out_path.write_text((content['response']['body']['choices'][0]['message']['content']).strip(), encoding="utf-8")
            except Exception as e:
                print(f"[warn] failed to download output for {b.id}: {e}")
                contents = [] 
                error = True
        else:
            print (f"{out_path} already downloaded; reading the file and returning contents ...")
            contents = out_path.read_text(encoding="utf-8").strip().splitlines()
        return error, contents
    except Exception as ex:
        print(f"Error downloading {batch.id}. Skipping: {e}")

def download_results(client:OpenAI, path:Path):
    outputs_all: List[str] = []
    errors_all: List[str]  = []
    try:
        batches = list_batches(client)
        for batch in batches:
            if batch.status == "cancelled":
                print(f"{batch.id} is cancelled; ignoring ...")
                continue
            if batch.status == "completed":
                error, contents = download_result(client, batch, path)
                if contents:
                    if error:
                        errors_all.extend(contents)
                    else:
                        outputs_all.extend(contents)
            else:
                print(f"{batch.id} is not complete (status == {batch.status})... ignoring")
    except Exception as ex:
        print(f"failed downloading results {str(ex)}")
        return None, None
    return outputs_all, errors_all



# Example usage
if __name__ == "__main__":
    argparse = __import__('argparse')
    parser = argparse.ArgumentParser(description="Process batch completions (cancel, wait, list).")
    parser.add_argument("--output_path", type=str, help="Path to the output directory where the samples are stored. If none given, it will be printed to stdout.")
    parser.add_argument("--types", type=str, help="Type of training: properties, commands, routines, general.")
    parser.add_argument("--operation", type=str, help="Operation to perform on the batches: cancel, list, process")

    args = parser.parse_args()

    types = args.types.split(",") if args.types else ["properties", "commands"]
    operation = args.operation if args.operation else "list"
    operation = operation.strip()


    REFERENCE_DIR = Path(get_data_directory("customer_data", None))
    BATCHED_REQUESTS_DIR = Path(get_data_directory("datasets", "batched-requests"))
    OUTPUT_DIR = Path(get_data_directory("datasets", "batched-samples"))

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"Output path {output_path} does not exist or is not a directory.")

    client = OpenAI(api_key=globals()[f"OPENAI_API_KEY_BATCH"])  # or use environment variable
    try:
        if operation == "cancel":
            print(f"Cancelling all batches for {type}...")
            for batch in list_batches(client):
                try:
                    client.batches.cancel(batch.id)
                    print(f"Cancelled batch {batch.id}")
                except Exception as e:
                    print(f"Error cancelling batch {batch.id}: {e}")
        elif operation == "process":
            print(f"Waiting for all batches to complete for {type}...")
            download_results(client, OUTPUT_DIR)

        elif operation == "list":
            print(f"Listing all batches ...")
            for batch in list_batches(client):
                print(f"Batch ID: {batch.id}, Status: {batch.status}, Created: {batch.created_at}, Completed: {batch.completed_at}")
    except Exception as e:
        print(f"Error processing batch for {type}. Skipping: {e}")

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
