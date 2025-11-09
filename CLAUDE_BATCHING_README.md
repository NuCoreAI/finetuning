# NuCore OpenPipe Sample Generation with Claude Haiku

This guide explains how to generate OpenPipe training samples for NuCore commands, properties, and routines using Claude Haiku with Anthropic's Batch API.

## Overview

The Claude Haiku batching system allows you to:
- Generate high-quality training samples for NuCore smart home automation
- Process large volumes of device structures in parallel using Anthropic's Batch API
- Create samples for commands, properties, and routines at scale
- Optimize costs by using batch processing instead of real-time API calls

## Files

### Python Scripts

- **`create_samples_batch_claude.py`**: Main script to create batch requests for Claude Haiku
- **`process_batch_completion_claude.py`**: Script to process completed batches and extract samples

### Prompt Files (Claude-Optimized)

- **`prompts/commands.prompt.train.claude`**: Prompt for generating Command samples
- **`prompts/properties.prompt.train.claude`**: Prompt for generating Property Query samples
- **`prompts/routines.prompt.train.claude`**: Prompt for generating Routine/Automation samples
- **`prompts/nucore.prompt.train.claude`**: Prompt for generating NuCore concept samples

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `nucore-ai` - NuCore library for device processing
- `openai` - For OpenAI format compatibility
- `anthropic` - Anthropic API client for Claude Haiku

### 2. Configure API Keys

Create a secrets file at the configured secrets directory (usually `~/nucore/secrets/keys.py`):

```python
# API keys for different sample types
ANTHROPIC_API_KEY_commands = "sk-ant-api03-..."
ANTHROPIC_API_KEY_properties = "sk-ant-api03-..."
ANTHROPIC_API_KEY_routines = "sk-ant-api03-..."
ANTHROPIC_API_KEY_nucore = "sk-ant-api03-..."

# Optional: Single key for batch processing
ANTHROPIC_API_KEY_BATCH = "sk-ant-api03-..."
```

You can use the same API key for all types if preferred.

### 3. Prepare Device Data

Ensure your customer data directory has the following structure:

```
customer_data/
├── nodes/
│   ├── nodes-device1.xml
│   ├── nodes-device2.xml
│   └── ...
└── profiles/
    ├── profile-device1.json
    ├── profile-device2.json
    └── ...
```

## Usage

### Step 1: Create Batch Requests

Generate batch requests for Claude Haiku to process:

```bash
# Generate samples for commands and properties (default)
python create_samples_batch_claude.py

# Generate samples for all types
python create_samples_batch_claude.py --types "commands,properties,routines"

# Generate samples for routines only
python create_samples_batch_claude.py --types "routines"

# Generate generic NuCore concept samples
python create_samples_batch_claude.py --types "nucore"

# Specify custom input/output paths
python create_samples_batch_claude.py \
    --input_path /path/to/customer_data \
    --output_path /path/to/batched-requests \
    --types "commands,properties,routines"
```

**What this does:**
1. Loads NuCore device structures from nodes and profiles
2. Creates batch requests in Anthropic's format
3. Submits batches to Anthropic's Batch API
4. Saves request files with batch IDs for tracking

**Output:**
- Batch request files in `datasets/batched-requests-claude/`
- Format: `batch_claude_{num}_{batch_id}.jsonl`

### Step 2: Monitor Batch Status

Check the status of your submitted batches:

```bash
# List all active batches
python process_batch_completion_claude.py --operation list
```

**Output example:**
```
Batch ID: msgbatch_01abc123
  Status: in_progress
  Created: 2025-11-09T10:30:00Z
  Ended: N/A
  Request counts: {'processing': 150, 'succeeded': 0, 'errored': 0, 'canceled': 0, 'expired': 0}

Batch ID: msgbatch_02def456
  Status: ended
  Created: 2025-11-09T09:15:00Z
  Ended: 2025-11-09T10:20:00Z
  Request counts: {'processing': 0, 'succeeded': 200, 'errored': 0, 'canceled': 0, 'expired': 0}
```

### Step 3: Process Completed Batches

Once batches are completed (status: `ended` with `processing_status: succeeded`), extract the samples:

```bash
# Process all completed batches
python process_batch_completion_claude.py --operation process

# Process with custom output path
python process_batch_completion_claude.py \
    --operation process \
    --output_path /path/to/output
```

**What this does:**
1. Downloads results from completed batches
2. Extracts training samples from Claude's responses
3. Saves individual sample files in JSONL format
4. Validates JSON formatting

**Output:**
- Raw batch outputs: `{batch_id}_output.jsonl`
- Extracted samples: `sample_{batch_id}_{custom_id}.jsonl`
- Location: `datasets/batched-samples-claude/`

### Step 4: Manage Batches

Additional operations for batch management:

```bash
# Cancel all in-progress batches
python process_batch_completion_claude.py --operation cancel

# Archive all batches (marks them as processed)
python process_batch_completion_claude.py --operation archive
```

**Archive function:**
- Stores batch metadata in `datasets/archives_claude.json`
- Prevents re-processing of completed batches
- Useful for keeping track of processed batches over time

## Sample Output Format

All generated samples follow the OpenAI chat fine-tuning JSONL format:

```jsonl
{"messages":[{"role":"system","content":"You are a NuCore smart-home assistant..."},{"role":"user","content":"DEVICE STRUCTURE:\n...\n\nUSER QUERY: turn on the kitchen light"},{"role":"assistant","content":"{\"tool\":\"Command\",\"args\":{\"commands\":[{\"device_id\":\"12345\",\"command_id\":\"DON\"}]}}"}]}
{"messages":[{"role":"system","content":"You are a NuCore smart-home assistant..."},{"role":"user","content":"DEVICE STRUCTURE:\n...\n\nUSER QUERY: what's the temperature?"},{"role":"assistant","content":"{\"tool\":\"PropQuery\",\"args\":{\"queries\":[{\"device_id\":\"67890\",\"property_id\":\"ST\",\"property_name\":\"temperature\"}]}}"}]}
```

Each line is a complete training sample with:
- **system**: NuCore runtime prompt
- **user**: Device structure + natural language query
- **assistant**: Properly formatted tool call response

## Sample Types

### Commands
Natural language commands that control devices:
- Single device actions: "turn on the light"
- Multi-device actions: "turn off all bedroom lights"
- Parametric commands: "set temperature to 72"
- Color commands: "change light to blue"

### Properties
Natural language queries about device state:
- Status queries: "is the light on?"
- Value queries: "what's the temperature?"
- Multi-device queries: "what's the status of all fans?"

### Routines
Natural language automation requests:
- **COS triggers**: "turn on fan when temperature exceeds 75"
- **COC triggers**: "when I double-tap the switch, turn off all lights"
- **Schedules**: "turn on porch light at sunset"
- **Complex logic**: "if temp > 70 AND after sunset, turn on fan"
- **Wait/Repeat**: "flash the light 3 times"

### NuCore Concepts
General knowledge questions about NuCore:
- "What is NuCore?"
- "What's the difference between COS and COC?"
- "How do I create a routine?"
- "What are UOM units?"

## Batch Processing Benefits

### Cost Efficiency
- Batch API pricing is typically 50% lower than real-time API
- Process thousands of samples economically

### Throughput
- Process up to 10,000 requests per batch
- Multiple batches can run in parallel
- Typical completion time: 24 hours or less

### Resource Management
- Asynchronous processing
- No rate limit concerns
- Automatic retry handling

## Troubleshooting

### Batch Creation Fails

**Error**: `Error creating batch for batch_claude_1.jsonl`

**Solution**: Check your API key configuration and ensure you have batch API access.

### No Samples Extracted

**Error**: Empty output files or "Empty content for {custom_id}"

**Solution**:
1. Check that prompts are loading correctly
2. Verify device structure is not empty
3. Review Claude's raw output in `{batch_id}_output.jsonl`

### Invalid JSON in Samples

**Warning**: `Skipping invalid JSON line`

**Solution**:
1. Claude occasionally adds explanatory text
2. The processing script automatically filters these out
3. Valid samples are still extracted

### API Key Not Found

**Error**: `KeyError: 'ANTHROPIC_API_KEY_commands'`

**Solution**: Ensure your secrets file has the required API keys defined.

## Advanced Usage

### Custom Batch Sizes

Modify `BATCH_MAX_REQUESTS` in `create_samples_batch_claude.py`:

```python
BATCH_MAX_REQUESTS = 5000  # Default: 10000
```

### Temperature Adjustment

Modify `TEMPERATURE` for more/less diverse samples:

```python
TEMPERATURE = 1.0  # Default: 1.0 (high diversity)
# TEMPERATURE = 0.7  # Medium diversity
# TEMPERATURE = 0.3  # Low diversity (more consistent)
```

### Model Selection

Change the Claude model in `create_samples_batch_claude.py`:

```python
MODEL = "claude-3-5-haiku-20241022"  # Default: Latest Haiku
# MODEL = "claude-3-haiku-20240307"  # Earlier Haiku version
# MODEL = "claude-3-5-sonnet-20241022"  # Sonnet (higher quality, higher cost)
```

## Comparison: Claude Haiku vs GPT Models

| Feature | Claude Haiku | GPT-4o-mini |
|---------|-------------|-------------|
| **Quality** | High instruction following | Excellent for structured output |
| **Cost (Batch)** | ~$0.25/1M input tokens | ~$0.30/1M input tokens |
| **Speed** | ~24 hours | ~24 hours |
| **Format Compliance** | Excellent with clear prompts | Excellent with JSON mode |
| **Context Window** | 200K tokens | 128K tokens |
| **Best For** | Complex reasoning, diverse samples | Consistent formatting, large scale |

## Best Practices

1. **Start Small**: Test with a small batch first to validate prompts
2. **Monitor Progress**: Check batch status regularly
3. **Archive Completed**: Archive batches after processing to keep tracking clean
4. **Validate Samples**: Review extracted samples for quality
5. **Incremental Processing**: Process by type (commands, then properties, then routines)
6. **Backup Requests**: Keep batch request files for reprocessing if needed

## Next Steps

After generating samples:

1. **Combine Samples**: Use `combine_samples.py` to merge all samples
2. **Check Quality**: Use `check_samples.py` to validate sample format
3. **Upload to OpenPipe**: Upload combined JSONL to OpenPipe for fine-tuning
4. **Monitor Training**: Track fine-tuning progress and metrics
5. **Deploy Model**: Use fine-tuned model in production

## Support

For issues or questions:
1. Check batch status with `--operation list`
2. Review raw output files in `datasets/batched-samples-claude/`
3. Verify API keys and permissions
4. Check Anthropic API status page

## References

- [Anthropic Batch API Documentation](https://docs.anthropic.com/en/api/batch-api)
- [Claude Haiku Model Card](https://www.anthropic.com/claude/haiku)
- [OpenPipe Fine-tuning Guide](https://openpipe.ai/docs)
- [NuCore Documentation](https://nucore.ai/docs)
