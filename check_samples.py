#this file checks samples that are in jsonl format for structural validity.
#if not valid, they are moved to errors directory for manual checking


import json, os
from pathlib import Path
from typing import List
from util import get_data_directory


def check_sample_structure(sample: dict) -> bool:
    """
    We have chat completion messages of the form:
    {"messages":[
        {"role":"system","content":"{{TEMPLATE_PROMPTS_RUNTIME}}"},
        {"role":"user","content":"DEVICE STRUCTURE:\n{{DEVICE_STRUCTURE}}\n\nUSER QUERY: <natural language command>"},
        {"role":"assistant","content":"<ASSISTANT RESPONSE>"}
    ]}  
    We want to ensure that each sample has the correct structure.   
    Returns True if the sample is valid, False otherwise.
    Invalid samples are logged to the error_pa

    """
    try:
        if not isinstance(sample, dict):
            print(f"Sample is not a dict: {sample}")
            return False
        if 'messages' not in sample:
            print(f"Sample does not have 'messages' key: {sample}")
            return False
        messages = sample['messages']
        if not isinstance(messages, list):
            print(f"'messages' is not a list: {messages}")
            return False
        if len(messages) != 3:
            print(f"'messages' does not have 3 elements: {messages}")
            return False
        system = messages[0]
        user = messages[1]
        assistant = messages[2]

        if not isinstance(system, dict) or system.get('role') != 'system':
            print(f"First element is not a system message: {system}")
            return False
        if not isinstance(user, dict) or user.get('role') != 'user':
            print(f"Second element is not an user message: {user}")
            return False
        if not isinstance(assistant, dict) or assistant.get('role') != 'assistant':
            print(f"Third element is not a assistant message: {assistant}")
            return False

        try:
             
            #now let's check the content for each message
            system_content = system.get('content', None).strip()
            user_content = user.get('content', None).strip()
            assistant_content = assistant.get('content', None).strip()

            if not isinstance(system_content, str):
                print(f"System message content is invalid: {system}")
                return False
            if not isinstance(user_content, str):
                print(f"User message content is invalid: {user}")
                return False
            if not isinstance(assistant_content, str):
                print(f"Assistant message content is invalid: {assistant}")
                return False
            
            # now let's check the actual content
            # Make sure system content has
            if "DEVICE STRUCTURE:" not in user_content:
                print(f"User message does not have DEVICE STRUCTURE: {user_content}")
                return False
            if "USER QUERY:" not in user_content:
                print(f"User message does not have USER QUERY: {user_content}")
                return False

            return True

        except Exception as e:
            print(f"Error checking message contents: {e}")
            return False
        
    except Exception as e:
        print(f"Error checking sample structure: {e}")
        return False

def check_samples_in_file(file_path: Path)-> bool:
    """
    Check all samples in a given JSONL file for structural validity.
    Invalid samples are logged to the error_path.
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    if line.strip():
                        sample = json.loads(line)
                        if check_sample_structure(sample):
                            continue
                        return False
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file {file_path}: {e}")
                    return False
            print(f"All samples in {file_path} are valid.")
            return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    
# Example usage
if __name__ == "__main__":
    argparse = __import__('argparse')
    parser = argparse.ArgumentParser(description="check samples for structural validity")
    parser.add_argument("--input-path", default="batched-samples", type=str, help="Path to the input directory where the samples are stored.")
    parser.add_argument("--errors-path", default="errors", type=str, help="Path to the directory where errors will be logged.")

    args = parser.parse_args()

    input_path = Path(get_data_directory("datasets", args.input_path))
    errors_path = Path(get_data_directory("datasets", args.errors_path))    
    if not errors_path.exists():
        os.makedirs(errors_path)
    
    """
    Check all JSONL files in the input_path for structural validity.
    Invalid samples are logged to the error_path.
    """
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")
    
    for file in input_path.glob("*.jsonl"):
        print(f"Checking samples in file: {file}")
        if not check_samples_in_file(file):
            print(f"Invalid samples found in {file}. Moving to errors directory.")
            dest_file = errors_path / file.name
            file.rename(dest_file)
            print(f"Moved {file} to {dest_file}")

    print ("Sample check completed.")
