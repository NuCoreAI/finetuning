
from openai import OpenAI
import json
import os
from pathlib import Path
from nucore import NuCore
from util import get_data_directory


# === CONFIGURATION ===
SECRETS_DIR = Path(get_data_directory("secrets", None))
if not SECRETS_DIR.exists():
    raise FileNotFoundError(f"Secrets directory {SECRETS_DIR} does not exist. Please create it and add your OpenAI API key.")
# Load the OpenAI API key from the secrets file
if not (SECRETS_DIR / "keys.py").exists():
    raise FileNotFoundError(f"Secrets file {SECRETS_DIR / 'keys.py'} does not exist. Please create it and add your OpenAI API key.")
exec(open(SECRETS_DIR / "keys.py").read())  # This will set OPENAI_API_KEY  

#MODEL = "gpt-4o"  # Use the latest model available
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0

SYSTEM_PROMPT = """
You are a smart home assistant trained to understand devices and their properties, commands, and parameters from structured context. Your job is to help users query, control, automate, and optimize devices using this context.

You will be given structured data about a device (its properties, accepted commands, send commands, parameter ranges, and units). Based on this, generate **5 fine-tuning examples** in the OpenPipe JSONL format. Each example should include:

1. A system message like: "You are a smart home assistant."
2. A user message with:
   - The device structure (context) clearly marked (e.g., starting with "DEVICE STRUCTURE:")
   - A natural-language user query or command about the device.
3. An assistant message that gives the correct, context-aware response based on the structure.
   - If a command is requested, include the command name, parameters, their values and units of measure, and list the permissible ranges for each parameter.
   - If a property is requested, include the property name and its value and unit of measure.
   - If a parameter range is requested, include the parameter name and its minimum and maximum values, or a subset if applicable.
   - If status is requested, look for the Status property and include its value and unit of measure and list the permissible ranges if applicable.
4. Ensure the response is helpful, accurate, and uses the device context effectively.

**Key instructions:**  
Place reasoning for the answer within the assistant's reply, before clearly stating the result, especially if the user-provided examples initially give results before rationale (reverse reasoning/conclusion order if needed).

Use this output format (each entry as a single line JSON object as shown below):

```json
{
  "messages": [
    {"role":"system","content":"You are a smart home assistant and NuCore expert in automation and optimization."},
    {"role": "user", "content": "DEVICE STRUCTURE:\n<device_info>\n\nUSER QUERY:\n<query here>"},
    {"role": "assistant", "content": "<correct, context-aware, reasoned response>"}
  ]
}

Only output the 5 examples. Each example must be in a single JSON object using double quotes and valid JSONL.

"""

def generate_openpipe_entries(full_text, output_path, dump=True):


    client = OpenAI(api_key=OPENAI_API_KEY)  # or use environment variable
    jsonl_data = [] 

    # replace <device_info> in the system prompt with the actual device info
    system_prompt = SYSTEM_PROMPT.replace("<device_info>", full_text)

    if full_text: 
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

        except Exception as e:
            print(f"Error: {e} | Input: {full_text[:60]}")

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
    args = parser.parse_args()

    REFERENCE_DIR = Path(get_data_directory("customer_data", None))
    OUTPUT_DIR = Path(get_data_directory("datasets", "devices"))

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
            full_text = ""
            for rag_doc in rag_docs:
                full_text += rag_doc
            if out_file:
                print(f"Writing to {out_file}")
                if not out_file.parent.exists():
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                generate_openpipe_entries(full_text, out_file, dump=True)

        except Exception as e:
            print(f"Error processing RAG documents for node {node_file}. Skipping: {e}")
            continue    
                    
                    
                    
     #EXAMPLES.append((node_data, profile_data))

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
