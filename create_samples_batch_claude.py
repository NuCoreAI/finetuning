#!/usr/bin/env python3
"""
This file creates OpenPipe samples for NuCore using Claude Haiku with Anthropic's Batch API.
Generates training samples for commands, properties, and routines that bake in the system prompt.
"""

from random import random
import json
import os
from pathlib import Path
from nucore import NuCore
from util import get_data_directory
from typing import Literal, List
import anthropic


# === CONFIGURATION ===
SECRETS_DIR = Path(get_data_directory("secrets", None))
if not SECRETS_DIR.exists():
    raise FileNotFoundError(f"Secrets directory {SECRETS_DIR} does not exist. Please create it and add your Anthropic API key.")

# Load the Anthropic API key from the secrets file
if not (SECRETS_DIR / "keys.py").exists():
    raise FileNotFoundError(f"Secrets file {SECRETS_DIR / 'keys.py'} does not exist. Please create it and add your Anthropic API key.")

exec(open(SECRETS_DIR / "keys.py").read())  # This will set ANTHROPIC_API_KEY_*

PROMPTS_DIR = Path(get_data_directory("prompts", None))
if not PROMPTS_DIR.exists():
    raise FileNotFoundError(f"Prompt directory {PROMPTS_DIR} does not exist. Please use git clone to get everything.")


# Claude Haiku model
MODEL = "claude-3-5-haiku-20241022"  # Latest Haiku model
TEMPERATURE = 1.0  # Use higher temperature for diverse training samples
BATCH_MAX_REQUESTS = 10000  # Anthropic batch API limit

TRAIN_PROMPT = ""
RUN_PROMPT = ""


def setup_prompts(type: Literal["properties", "commands", "routines", "nucore"]):
    """Load and setup prompts for the given type."""
    global TRAIN_PROMPT, RUN_PROMPT
    TRAIN_PROMPT = ""
    RUN_PROMPT = ""

    try:
        with open(os.path.join(PROMPTS_DIR, f"{type}.prompt.train.claude"), "r") as f:
            TRAIN_PROMPT = f.read()
    except FileNotFoundError:
        # Fall back to regular prompt if Claude-specific not found
        try:
            with open(os.path.join(PROMPTS_DIR, f"{type}.prompt.train"), "r") as f:
                TRAIN_PROMPT = f.read()
        except:
            raise ValueError(f"Failed to load training prompt for type {type}.")

    try:
        with open(os.path.join(PROMPTS_DIR, f"system.prompt.preamble"), "r") as f:
            RUN_PROMPT = f.read().replace("\n", "\\n")
    except:
        raise ValueError(f"Failed to load system prompt preamble for type {type}.")

    # Replace template placeholder with runtime prompt
    TRAIN_PROMPT = TRAIN_PROMPT.replace("{{TEMPLATE_PROMPTS_RUNTIME}}", f"{RUN_PROMPT}")


def make_and_save_batch(client: anthropic.Anthropic, batch_num: int, requests: List[dict], batched_requests_dir: Path) -> tuple:
    """
    Save a batch of requests to a jsonl file and create an Anthropic batch.
    Returns (jsonl_path, batch_id).
    """
    print(f"Saving batch {batch_num} of {len(requests)} requests to temporary file...")
    jsonl_path = batched_requests_dir / f"batch_claude_{batch_num}.jsonl"

    try:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for obj in requests:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error saving batch {batch_num}: {e}")
        return None, None

    print(f"Batch saved to {jsonl_path}")

    try:
        # Create message batch using Anthropic API
        # Note: Anthropic's batch API requires requests in their specific format
        batch = client.messages.batches.create(
            requests=requests
        )
        batch_id = batch.id

        # Rename the request file to include the batch id
        new_jsonl_path = batched_requests_dir / f"batch_claude_{batch_num}_{batch_id}.jsonl"
        os.rename(jsonl_path, new_jsonl_path)
        jsonl_path = new_jsonl_path

        print(f"Created Anthropic batch: {jsonl_path} with id: {batch_id}")
        return jsonl_path, batch_id
    except Exception as e:
        print(f"Error creating batch for {jsonl_path}: {e}")
        return None, None


def generate_request(full_text: str, request_id: str, type: str) -> dict:
    """
    Generate a single batch request in Anthropic's format.
    """
    if full_text:
        # Replace device structure placeholder in the system prompt
        system_prompt = TRAIN_PROMPT.replace("{{DEVICE_STRUCTURE}}", full_text)

        # Anthropic batch request format
        return {
            "custom_id": request_id,
            "params": {
                "model": MODEL,
                "max_tokens": 4096,  # Enough for multiple training samples
                "temperature": TEMPERATURE,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate the training samples as specified in the system prompt."
                    }
                ]
            }
        }
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate OpenPipe fine-tuning samples using Claude Haiku batch API."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the directory with profiles and nodes directories. Defaults to customer_data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output directory for batch requests. Defaults to datasets/batched-requests."
    )
    parser.add_argument(
        "--types",
        type=str,
        help="Comma-separated types: properties, commands, routines, nucore. Default: properties,commands"
    )

    args = parser.parse_args()

    types = args.types.split(",") if args.types else ["properties", "commands", "routines"]

    REFERENCE_DIR = Path(get_data_directory("customer_data", None))
    BATCHED_REQUESTS_DIR = Path(get_data_directory("datasets", "batched-requests-claude"))

    input_path = Path(args.input_path) if args.input_path else REFERENCE_DIR
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input path {input_path} does not exist or is not a directory.")

    output_path = Path(args.output_path) if args.output_path else BATCHED_REQUESTS_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Traverse input directory for profiles and nodes
    nodes_dir = input_path / "nodes"
    profiles_dir = input_path / "profiles"

    if not nodes_dir.exists() or not nodes_dir.is_dir():
        raise ValueError(f"Nodes directory {nodes_dir} does not exist or is not a directory.")
    if not profiles_dir.exists() or not profiles_dir.is_dir():
        raise ValueError(f"Profiles directory {profiles_dir} does not exist or is not a directory.")

    for type in types:
        type = type.strip()

        # Initialize Anthropic client
        client = anthropic.Anthropic(
            api_key=globals()[f"ANTHROPIC_API_KEY_{type}"]
        )

        setup_prompts(type)

        batch_requests = []
        batch_num = 1

        # Handle nucore generic samples
        if type == "nucore":
            request_id = f"nucore_generic_{int(random() * 10000)}"
            request = generate_request(" ", request_id, type)
            if request:
                batch_requests.append(request)

                if len(batch_requests) >= BATCH_MAX_REQUESTS:
                    path, id = make_and_save_batch(client, batch_num, batch_requests, output_path)
                    if path is None or id is None:
                        print(f"Error creating batch for {request_id}. Stopping.")
                        break
                    batch_requests = []
                    batch_num += 1

                # Save any remaining requests
                if batch_requests:
                    path, id = make_and_save_batch(client, batch_num, batch_requests, output_path)
                    if path is None or id is None:
                        print(f"Error creating final batch for {request_id}.")
                    batch_num += 1
            continue

        # Process node files
        for node_file in nodes_dir.glob("*.xml"):
            profile_file = profiles_dir / (f"{node_file.stem}.json").replace("nodes-", "profile-")

            print(f"Processing node: {node_file.name} with profile: {profile_file.name}")

            if not profile_file.exists():
                print(f"Warning: Profile file {profile_file} does not exist for node {node_file}. Skipping.")
                continue

            # Initialize NuCore
            nuCore = NuCore(
                collection_path="/tmp/nucore.finetuner",
                collection_name="finetuner",
                backend_url="http://localhost:8000",
                backend_username="admin",
                backend_password="admin"
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

                # Process in batches of 3 documents
                for i in range(0, len(rag_docs), 3):
                    batch = rag_docs[i:i+3]
                    full_text = "".join(batch)

                    request_id = f"{node_file.stem}_{i//3 + 1}_{type}"
                    print(f"Creating request: {request_id}")

                    request = generate_request(full_text, request_id, type)
                    if request:
                        batch_requests.append(request)

                        # Create batch if we've hit the limit
                        if len(batch_requests) >= BATCH_MAX_REQUESTS:
                            path, id = make_and_save_batch(client, batch_num, batch_requests, output_path)
                            if path is None or id is None:
                                print(f"Error creating batch for {request_id}. Stopping.")
                                batch_requests = []
                                break
                            batch_requests = []
                            batch_num += 1

            except Exception as e:
                print(f"Error processing RAG documents for node {node_file}. Skipping: {e}")
                continue

        # Save any remaining requests
        if batch_requests:
            path, id = make_and_save_batch(client, batch_num, batch_requests, output_path)
            if path is None or id is None:
                print(f"Error creating final batch for type {type}.")
            else:
                print(f"Successfully created all batches for type {type}")


if __name__ == "__main__":
    main()
