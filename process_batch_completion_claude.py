#!/usr/bin/env python3
"""
This file processes completed Anthropic Claude batch requests and extracts the training samples.
"""

import anthropic
import json
from datetime import datetime
from pathlib import Path
from util import get_data_directory
from typing import List


# === CONFIGURATION ===
SECRETS_DIR = Path(get_data_directory("secrets", None))
if not SECRETS_DIR.exists():
    raise FileNotFoundError(f"Secrets directory {SECRETS_DIR} does not exist.")

if not (SECRETS_DIR / "keys.py").exists():
    raise FileNotFoundError(f"Secrets file {SECRETS_DIR / 'keys.py'} does not exist.")

exec(open(SECRETS_DIR / "keys.py").read())  # This will set ANTHROPIC_API_KEY_*


# A dictionary of completion status/archived
archives = {}


def is_archived(batch) -> bool:
    """Check if a batch is archived."""
    try:
        b = archives[batch.id]
        return b['status'] == 'archived'
    except Exception:
        return False


def list_batches(client: anthropic.Anthropic, include_archives: bool, include_cancels: bool = False, include_fails: bool = False):
    """
    List all Anthropic batches and their statuses.
    """
    # Get all message batches
    batches_response = client.messages.batches.list()

    for batch in batches_response.data:
        if not include_archives:
            if is_archived(batch):
                continue

        # Map Anthropic status to our filtering logic
        # Anthropic statuses: in_progress, canceling, ended
        # When ended, check processing_status: succeeded, errored, canceled, expired
        if batch.processing_status == "errored" and not include_fails:
            continue
        if batch.processing_status == "canceled" and not include_cancels:
            continue

        yield batch


def cancel_batches(client: anthropic.Anthropic):
    """Cancel all active batches."""
    print(f"Cancelling all active Claude batches...")
    for batch in list_batches(client, False):
        try:
            # Only cancel if not already ended
            if batch.processing_status in ["in_progress", "canceling"]:
                client.messages.batches.cancel(batch.id)
                print(f"Cancelled batch {batch.id}")
            else:
                print(f"Batch {batch.id} is already {batch.processing_status}, skipping cancel")
        except Exception as e:
            print(f"Error cancelling batch {batch.id}: {e}")
            continue


def archive_batches(client: anthropic.Anthropic, path: Path):
    """Archive all batches to a JSON file."""
    print(f"Archiving all batches...")
    for batch in list_batches(client, True):
        try:
            archives[batch.id] = {
                "status": "archived",
                "batch_status": batch.processing_status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Error archiving batch {batch.id}: {e}")
            continue

    with open(path, 'w') as fp:
        json.dump(archives, fp, indent=2)


def download_result(client: anthropic.Anthropic, batch, path: Path):
    """
    Download and process results from a completed Claude batch.
    Returns the extracted samples.
    """
    try:
        # Get batch results
        results_response = client.messages.batches.results(batch.id)

        out_path = path / f"{batch.id}_output.jsonl"
        samples_count = 0

        # Save raw output
        with out_path.open("w", encoding="utf-8") as f:
            for result in results_response:
                f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")

        print(f"Downloaded raw results to {out_path}")

        # Process each result and extract samples
        for result in results_response:
            try:
                custom_id = result.custom_id

                # Check if the result succeeded
                if result.result.type != "succeeded":
                    print(f"Skipping failed result for {custom_id}: {result.result.type}")
                    continue

                # Extract the assistant's message content
                message = result.result.message
                content = message.content[0].text if message.content else ""

                if not content:
                    print(f"Empty content for {custom_id}")
                    continue

                # Parse the JSONL samples from the response
                # Claude should return multiple JSONL lines
                samples_out_path = path / f"sample_{batch.id}_{custom_id}.jsonl"

                with open(samples_out_path, 'w', encoding="utf-8") as samples_file:
                    # Split by newlines and process each line as a potential sample
                    lines = content.strip().split('\n')

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Try to parse as JSON
                            sample = json.loads(line)
                            # Write the valid sample
                            samples_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            samples_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON line in {custom_id}: {line[:100]}...")
                            continue

                print(f"Extracted {samples_count} samples from {custom_id}")

            except Exception as ex:
                print(f"Error processing result {custom_id}: {ex}")
                continue

        return samples_count

    except Exception as ex:
        print(f"Error downloading batch {batch.id}: {ex}")
        return 0


def download_results(client: anthropic.Anthropic, path: Path):
    """Download all completed batch results."""
    total_samples = 0

    try:
        for batch in list_batches(client, False):
            if batch.processing_status == "canceled":
                print(f"{batch.id} is cancelled; ignoring...")
                continue

            if batch.processing_status == "succeeded":
                print(f"Processing completed batch {batch.id}...")
                count = download_result(client, batch, path)
                total_samples += count
            elif batch.processing_status == "in_progress":
                print(f"{batch.id} is still in progress (status == {batch.processing_status})... ignoring")
            else:
                print(f"{batch.id} status: {batch.processing_status}")

    except Exception as ex:
        print(f"Failed downloading results: {str(ex)}")
        return 0

    print(f"Total samples extracted: {total_samples}")
    return total_samples


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Anthropic Claude batch completions (cancel, wait, list, process, archive)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output directory for samples. Defaults to datasets/batched-samples-claude."
    )
    parser.add_argument(
        "--types",
        type=str,
        help="Type of training: properties, commands, routines, nucore. Default: all types."
    )
    parser.add_argument(
        "--operation",
        type=str,
        help="Operation: cancel, list, process, archive. Default: list"
    )

    args = parser.parse_args()

    types = args.types.split(",") if args.types else ["properties", "commands", "routines"]
    operation = args.operation.strip() if args.operation else "list"

    OUTPUT_DIR = Path(get_data_directory("datasets", "batched-samples-claude"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ARCHIVED_FILE = Path(get_data_directory("datasets", "archives_claude.json"))

    # Load archives
    if not ARCHIVED_FILE.exists():
        with open(ARCHIVED_FILE, 'w') as fp:
            json.dump(archives, fp)
    else:
        with open(ARCHIVED_FILE, 'r') as fp:
            try:
                archives.update(json.load(fp))
            except Exception as ex:
                print(f"Failed loading archives file {ARCHIVED_FILE}: {ex}")

    output_path = Path(args.output_path) if args.output_path else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Use batch API key (or fallback to first type)
    api_key_name = "ANTHROPIC_API_KEY_BATCH" if "ANTHROPIC_API_KEY_BATCH" in globals() else f"ANTHROPIC_API_KEY_{types[0]}"
    client = anthropic.Anthropic(api_key=globals()[api_key_name])

    try:
        if operation == "cancel":
            cancel_batches(client)

        elif operation == "process":
            download_results(client, output_path)

        elif operation == "archive":
            archive_batches(client, ARCHIVED_FILE)

        elif operation == "list":
            print(f"Listing all Claude batches that are not completed and not in archive...")
            for batch in list_batches(client, False):
                print(f"Batch ID: {batch.id}")
                print(f"  Status: {batch.processing_status}")
                print(f"  Created: {batch.created_at}")
                print(f"  Ended: {batch.ended_at if batch.ended_at else 'N/A'}")
                print(f"  Request counts: {batch.request_counts}")
                print()

        else:
            print(f"Unknown operation: {operation}")
            print("Valid operations: cancel, list, process, archive")

    except Exception as e:
        print(f"Error processing batch operation '{operation}': {e}")


if __name__ == "__main__":
    main()
