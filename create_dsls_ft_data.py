## Generate OpenPipe fine-tuning data for NuCore smart home devices using dsl (domain specific language)
from openai import OpenAI
import json
import os
from pathlib import Path
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
#MODEL = "gpt-4.1"
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.3
MAX_TOKENS = 32_000 # Adjust based on your needs, but ensure it fits within the model's limits

SYSTEM_PROMPT = """
You are an expert in NuCore DSL and smart-home automation.
Generate valid OpenPipe fine-tuning samples in JSON for NuCore DSL.
Each sample must be a separate OpenPipe object. Optimize outputs for code models.

---

You will receive a flattened smart device structure, labeled `DEVICE STRUCTURE:`
Each device shows:
- Properties (e.g. `ST`, `CSP`) with `uom` and `precision`
- Commands (`accepts`, `sends`)
- Parameters (name, value, uom, precision)

---

### OUTPUT FORMAT

**OpenPipe object per sample**
{
  "messages": [
    { "role": "system", "content": "You are a smart home assistant and NuCore expert in automation and optimization." },
    { "role": "user", "content": "DEVICE STRUCTURE:\n<device_info>\n\nUSER QUERY:\n<free form natural language request>" },
    { "role": "assistant", "content": "Reasoning: <clear concise explanation optimized for finetuning code models>\n\n{\"routine\": { ... }}" }
  ]
}

---

<dsls_prompt_template>

---

### COMPLEXITY REQUIREMENTS
- Use **mixed triggers**: COS, COC, and schedules 
- Always mix at least two logical operators (and + or).
- Prefer **grouped logic** with `(` and `)`
- Alternate `and`/`or` combinations
- Vary devices and properties across samples
- Output valid JSON only, no trailing commas
- Return OpenPipe JSON objects only, no extra commentary


"""

system_prompt_training = {
    "role": "system",
    "content": "You are a smart home assistant and NuCore expert in automation and optimization."
}

jsonl_data = []

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
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            assistant_reply = response.choices[0].message.content.strip()
            if not assistant_reply:
                ("Assistant reply is empty. Please check the input text.")
                if dump:
                    print(f"Assistant reply: {assistant_reply}")
            
            encoded = assistant_reply.encode("utf-8")
            try:
                encoded = encoded.replace('{"\\(": 1}', '{"(": 1}')
                encoded = encoded.replace('{"\\)": 1}', '{")": 1}')
                # (optionally handle cases with spaces:)
                encoded = encoded.replace('"\\(": 1', '"(": 1').replace('"\\)": 1', '")": 1')
                unescaped = json.loads(encoded)
    
                i = 0
                for i in range(len(unescaped['messages'])-1):
                    role = unescaped['messages'][i]['role']
                    if role == "system":
                        i+=1
                        continue
                    
                    user=unescaped['messages'][i],
                    assistant=unescaped['messages'][i+1],
                    jsonl= {"messages": [
                        system_prompt_training,
                        user[0],
                        assistant[0],
                    ]}
                    i+=2

                    jsonl_data.append(jsonl)

            except Exception as e:
                print(f"Error processing JSON: {e}")
                with open(output_path.with_suffix(".error"), "w") as f:
                    f.write(str(e)+"\n*****\n")
                    f.write(str(encoded))
                return

        except Exception as e:
                print(f"Error processing : {e}")
                with open(output_path.with_suffix(".error"), "w") as f:
                    f.write(str(e)+"\n*****\n")
                    f.write(str(encoded))

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

    INPUT_DIR = Path(get_data_directory("datasets", "devices"))
    OUTPUT_DIR = Path(get_data_directory("datasets", "dsls"))

    input_path = Path(args.input_path) if args.input_path else INPUT_DIR
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR

    if not output_path.exists() or not output_path.is_dir():
        raise ValueError(f"Output path {output_path} does not exist or is not a directory.")

    # now traverse the input directory where you will find profiles and nodes directories
    # start with files in the nodes directory and then use the name of the file (without extension) to find the corresponding profile in the profiles directory
    for node_file in INPUT_DIR.glob("*.jsonl"):
        
        try:
            print(f"Processing node: {node_file.name} ")
            with open(node_file, 'r') as f:
                full_text = f.read()
                #extract the device structure from the file
                if not full_text:
                    print(f"Warning: File {node_file} is empty. Skipping.")
                    continue
                #read each line as a JSON object
                jsonl_data = []
                for line in full_text.splitlines():
                    if line.strip():
                        jsonl_data.append(json.loads(line))
                if not jsonl_data:
                    print(f"Warning: No valid JSON objects found in {node_file}. Skipping.")
                    continue
                sample_count = 0
                out_file = output_path / f"{node_file.stem}_{sample_count}.jsonl"
                full_text = ""
                for sample in jsonl_data:
                    if "messages" not in sample or len(sample["messages"]) < 3:
                        print(f"Warning: Invalid sample format in {node_file}. Skipping.")
                        continue
                    # now get the first element which is role = system
                    user_data = sample["messages"][1] if len(sample["messages"]) > 1 else None
                    if not user_data or "content" not in user_data:
                        print(f"Warning: No user content found in {node_file}. Skipping.")
                        continue
                    content = user_data["content"]
                    if not content:
                        print(f"Warning: User content in {node_file} is empty. Skipping.")
                        continue
                    # now remove the DEVICE STRUCTURE: part
                    device_structure_start = content.find("DEVICE STRUCTURE:")
                    if device_structure_start != -1:
                        content = content[device_structure_start + len("DEVICE STRUCTURE:"):].strip()
                    if not content:
                        print(f"Warning: No device structure found in {node_file}. Skipping.")
                        continue
                    # now remove all the extra text starting with \nUSER QUERY:\n and all the way to the end of the string
                    user_query_start = content.find("\nUSER QUERY:\n")
                    if user_query_start != -1:
                        content = content[:user_query_start]
                    if not content:
                        print(f"Warning: No device structure found in {node_file}. Skipping.")
                        continue
                    content = content.strip()
                    full_text += content + "\n"
            if out_file:
                print(f"Processing  {out_file}")
                generate_openpipe_entries(full_text, out_file, dump=True)

        except Exception as e:
            print(f"Error processing device documents for node {node_file}. Skipping: {e}")
            continue    
                    
                    
                    
     #EXAMPLES.append((node_data, profile_data))

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
