
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
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0

SYSTEM_PROMPT = """
Create fine-tuning samples in OpenPipe JSONL format for a smart home assistant specializing in NuCore technology. You will be given structured information about a smart device, including its properties, accepts and sends commands, parameter ranges, and units of measure (UOMs). For each sample:

- Use a different device, and for each device, ensure your samples together cover at least 75% of its commands and 75% of its properties, with variety in UOMs, ranges, and permissible values. Do not be repetitious.
- Include realistic and educational user queries (in natural language) referencing the device structure.
- Device context must be clearly marked in the user message with "DEVICE STRUCTURE:".
- Avoid raw device IDs or excessively specific device naming. Use clear, natural terms.

Output for each sample must include:
    1. System prompt: "You are a smart home assistant and NuCore expert."
    2. User prompt: includes device structure explicitly (labelled "DEVICE STRUCTURE:") followed by a realistic user query or command.
    3. Assistant reply: include clear, step-by-step reasoning based on device context and user query, followed by the result or direct answer last (reasoning should come before any conclusions or lists; reverse this order if the user's examples initially show results before rationale).


Output format for each sample: a single JSON object per line as per OpenPipe standards as follows:
```json
{
  "messages": [
    {"role": "system", "content": "You are a smart home assistant."},
    {"role": "user", "content": "DEVICE STRUCTURE:\n<device_info>\n\nUSER QUERY:\n<query here>"},
    {"role": "assistant", "content": "<correct, context-aware, reasoned response>"}
  ]
}

Continue generating samples until sufficient variety (across commands, parameters, properties, units, ranges) is demonstrated per device and the coverage targets are met. Persist until objectives are complete. Think step-by-step before providing the assistant's conclusion.

Explicit output requirements:
- Format: Each example is a single line JSON object (do not use markdown or code blocks).
- Structure: Strictly follow the output format above, always using double quotes for all JSON.
- Length: Assistant responses should be thorough, including both detailed reasoning steps and a clear, concise conclusion/answer.
- For property/command details in user queries, always base your response precisely on the supplied context.

Example samples:

Example 1:
{"messages":[
  {"role":"system","content":"You are a smart home assistant and NuCore expert in automation and optimization."},
  {"role":"user","content":"DEVICE STRUCTURE:\nDevice: Smart Thermostat\nProperties: Temperature (float, range: 10.0-30.0 °C), Humidity (int, range: 10-90 %), Mode (string, values: [off, heat, cool, auto])\nAccepted Commands: setTemperature(value: float, °C), setMode(mode: string), setHumidityTarget(value: int, %)\nSend Commands: reportTemperature(), reportHumidity()\n\nUSER QUERY:\nHow would I change the thermostat to heat mode?"},
  {"role":"assistant","content":"First, I check the available modes in the device structure: modes include off, heat, cool, and auto. The accepted command for changing modes is setMode, which requires a string parameter.\n\nTo change the thermostat to heat mode, use the setMode command with the parameter 'heat'.\n\nResult: Execute setMode with mode='heat' to switch to heat mode."}
]}

Example 2:
{"messages":[
  {"role":"system","content":"You are a smart home assistant and NuCore expert."},
  {"role":"user","content":"DEVICE STRUCTURE:\nDevice: Smart Light Bulb\nProperties: Brightness (int, range: 0-100 %), Color (string, values: [red, green, blue, white, warm_white]), Power (string, values: [on, off])\nAccepted Commands: setBrightness(value: int, %), setColor(color: string), setPower(state: string)\nSend Commands: reportStatus()\n\nUSER QUERY:\nIf I want the bedroom light to be 50% brightness, what command do I use?"},
  {"role":"assistant","content":"I examine the device's properties and note that Brightness is an integer from 0 to 100 percent. The accepted command to change brightness is setBrightness, which takes a value parameter.\n\nTo set the light to 50% brightness, use setBrightness with value=50.\n\nResult: Send setBrightness(value=50) to set the bedroom light to 50% brightness."}
]}

(Real examples should use realistic device structures and queries, ensuring broad property/command/unit coverage with appropriately complex content.)

Edge cases & reminders:
- Ensure is sample is unique and covers all aspects of the device.
- Avoid repetition across samples.
- Ensure reasoning always precedes conclusions in the assistant's reply.
- Avoid device IDs or overly technical device names.
- Cover various units (e.g., %, °C, kWh, etc.) and property types (string, int, float).
- Make queries reflect plausible user intentions.
- Maintain strict compliance with single-line, double-quoted JSON per sample.


Output as many samples as needed. Each example must be in a single JSON object using double quotes and valid JSONL.

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

    INPUT_DIR = Path(get_data_directory("datasets", "devices"))
    OUTPUT_DIR = Path(get_data_directory("datasets", "concepts"))

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
                for sample in jsonl_data:
                    out_file = output_path / f"{node_file.stem}_{sample_count}.jsonl"
                    sample_count += 1
                    if "messages" not in sample or len(sample["messages"]) < 3:
                        print(f"Warning: Invalid sample format in {node_file}. Skipping.")
                        continue
                    # now get the first element which is role = system
                    user_data = sample["messages"][1] if len(sample["messages"]) > 1 else None
                    if not user_data or "content" not in user_data:
                        print(f"Warning: No user content found in {node_file}. Skipping.")
                        continue
                    full_text = user_data["content"]
                    if not full_text:
                        print(f"Warning: User content in {node_file} is empty. Skipping.")
                        continue
                    # now remove the DEVICE STRUCTURE: part
                    device_structure_start = full_text.find("DEVICE STRUCTURE:")
                    if device_structure_start != -1:
                        full_text = full_text[device_structure_start + len("DEVICE STRUCTURE:"):].strip()
                    if not full_text:
                        print(f"Warning: No device structure found in {node_file}. Skipping.")
                        continue
                    # now remove all the extra text starting with \nUSER QUERY:\n and all the way to the end of the string
                    user_query_start = full_text.find("\nUSER QUERY:\n")
                    if user_query_start != -1:
                        full_text = full_text[:user_query_start]
                    if not full_text:
                        print(f"Warning: No device structure found in {node_file}. Skipping.")
                        continue
                    full_text = full_text.strip()
                    if out_file:
                        print(f"Writing to {out_file}")
                    generate_openpipe_entries(full_text, out_file, dump=True)

        except Exception as e:
            print(f"Error processing device documents for node {node_file}. Skipping: {e}")
            continue    
                    
                    
                    
     #EXAMPLES.append((node_data, profile_data))

#    generate_openpipe_entries(EXAMPLES, "openpipe_finetune.jsonl")
