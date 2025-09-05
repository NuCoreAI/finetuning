#This file creates samples that bake in the system prompt into the model.
#This way, we do not use unnecessary tokens during inference.


from openai import OpenAI
import json
import os
from pathlib import Path
from nucore import NuCore
from util import get_data_directory
from typing import Literal


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


def generate_openpipe_entries(full_text, output_path, dump=True):

    client = OpenAI(api_key=OPENAI_API_KEY)  # or use environment variable
    jsonl_data = [] 

    if full_text: 
        # replace <device_info> in the system prompt with the actual device info
        system_prompt = TRAIN_PROMPT.replace("{{DEVICE_STRUCTURE}}", full_text)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
            ]

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE
            )

            assistant_reply = response.choices[0].message.content.strip()
            if not assistant_reply:
                ("Assistant reply is empty. Please check the input text.")
            # Split the assistant reply into individual JSON objects
            entries = assistant_reply.split("\n")
            for entry in entries:
                entry = entry.strip()
                if entry:
                    try:
                        json_data = json.loads(entry)
                        jsonl_data.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e} | Entry: {entry}")
                        with open(output_path.with_suffix(".error"), "w") as f:
                            f.write(str(e)+"\n*****\n")
                            f.write(str(entry))

        except Exception as e:
            print(f"Error: {e} | Input: {full_text[:60]}")
            with open(output_path.with_suffix(".error"), "w") as f:
                f.write(str(e)+"\n*****\n")
                f.write(str(assistant_reply))

        with open(output_path, "w") as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + "\n")

    print(f"✅ {len(jsonl_data)} entries saved to {output_path}")

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
    OUTPUT_DIR = Path(get_data_directory("datasets", "samples"))

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

    for type in types:
        type=type.strip()
        setup_prompts(type)
        for node_file in nodes_dir.glob("*.xml"):
            profile_file = profiles_dir / (f"{node_file.stem}.json").replace("nodes-", "profile-")
            out_file = None if not output_path else output_path / f"{node_file.stem}_finetune.jsonl"
            
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
                        batch_file = out_file.with_stem(f"{out_file.stem}_{i//3 + 1}_{type}")
                        print(f"Writing to {batch_file}")
                        batch_file.parent.mkdir(parents=True, exist_ok=True)
                        generate_openpipe_entries(full_text, batch_file, dump=True) 

    #            full_text = ""
    #            for rag_doc in rag_docs:
    #                full_text += rag_doc
    #            if out_file:
    #                print(f"Writing to {out_file}")
    #                if not out_file.parent.exists():
    #                    out_file.parent.mkdir(parents=True, exist_ok=True)
    #                generate_openpipe_entries(full_text, out_file, dump=True)

            except Exception as e:
                print(f"Error processing RAG documents for node {node_file}. Skipping: {e}")
                continue    
                        
                    
                    
     #EXAMPLES.append((node_data, profile_data))

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
