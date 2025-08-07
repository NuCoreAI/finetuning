## Generate OpenPipe fine-tuning data for NuCore smart home devices Colored Petri Nets(CPN) format
from openai import OpenAI
import json
import os
from pathlib import Path
from ai_iox_workflow.config import DEFAULT_AI_INSTALL_DIR
from ai_iox_workflow.nucore import NuCore


# === CONFIGURATION ===
SECRETS_DIR = Path(os.path.join(os.path.expanduser("~"), DEFAULT_AI_INSTALL_DIR, "finetuning_data", "secrets"))
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
You are an expert in smart home automation and optimization for NuCore-based systems.

You will receive a flattened structure description of smart devices, including 
- Properties (e.g. `ST`, `CSP`)
- Commands (`accepts`, `sends`)
- Parameters (with name, value, unit of measure, precision)
The full structure is labled as `DEVICE STRUCTURE:` and a device is delineated by ***Device***.

Your task is to generate OpenPipe fine-tuning samples that teach a model to construct automation and optimization routines using an extended version of JSONLogic.

---

**Important Distinction Between COS and COC:**

1. **COS (Change of State)**:  
   A device property changes as a result of a command. This is recorded if and only if the value truly changes.  
   - Example: `ST` becomes 100% when the device is turned on via API.

2. **COC (Change of Control)**:  
   A device initiates a command (e.g. physical double-click). This is always reported, even if the property does not change.  
   - Example: `DFON` sent when user physically turns on device—even if it was already on.

---

### Condition Examples:

**COS**: Check property state.
{
  "==": {
    "device": "thermostat_1",
    "status": "ST",
    "value": 100,
    "uom": "percent",
    "precision": 1
  }
}

**COC**: Check command from **sends**:
{
  "==": {
    "device": "switch",
    "coc": "DFON",
    "parameters": [
      { "name": "power", "value": 100, "uom": "percent", "precision": 1 }
    ]
  }
}

Action Format:
{
  "device": "thermostat_1",
  "command": "SetCoolSetPoint",
  "parameters": [
    { "name": "CSP", "value": 7800, "uom": "fahrenheit", "precision": 2 }
  ]
}

---

Output Requirements:
- Use double quotes for all strings and object keys (valid JSON).
- Device names and commands must come from the provided DEVICE STRUCTURE.
- Use natural language in user queries (e.g. “make it cooler,” “turn off lights,” “optimize for low price”).
- Conditions may include COS, COC, time, price, or status logic.
- Actions must include full command syntax with parameters and metadata.
- Use realistic device names, values, units, and patterns.
- Use  JSONLogic:
    - Give a meaningful name to each automation routine using the `name` key.
    - No nested if blocks
    - One top-level object: "automation" or "routine"
    - Conditions and actions expressed as plain JSON key/values
    - Return only the **raw JSON object**, not a string. Do not wrap it in quotes. Do not escape it.

Output must follow OpenPipe format::
{
  "messages": [
    {
      "role": "system",
      "content": "You are a smart home assistant and NuCore expert in automation and optimization."
    },
    {
      "role": "user",
      "content": "DEVICE STRUCTURE:\n<device_info>\n\nUSER QUERY:\n<query here>"
    },
    {
      "role": "assistant",
      "content": "<step-by-step reasoning>\n\n<JSONLogic routine as json object>"
    }
  ]
}

Guidelines:

    Always start with detailed step-by-step reasoning, followed by JSON output.

    Use double quotes for all strings and object keys (valid JSON).

    Device names and commands must come from the provided DEVICE STRUCTURE.

    Use natural language in user queries (e.g. “make it cooler,” “turn off lights,” “optimize for low price”).

    Conditions may include COS, COC, time, price, or status logic.

    Actions must include full command syntax with parameters and metadata.

    Use realistic device names, values, units, and patterns.
    

Reminder:

COS = changes in property values.
COC = commands sent by devices due to physical or external control.
They are not interchangeable and must be evaluated separately.
Return only the **raw JSON object**, not a string. Do not wrap it in quotes. Do not escape it.

Finally:
Output as many samples as possible. Each example must be in a single JSON object.

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

    INPUT_DIR = Path(os.path.join(os.path.expanduser("~"), DEFAULT_AI_INSTALL_DIR, "finetuning_data", "datasets", "devices"))
    OUTPUT_DIR = Path(os.path.join(os.path.expanduser("~"), DEFAULT_AI_INSTALL_DIR, "finetuning_data", "datasets", "dsls"))

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
