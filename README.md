# Finetuning
This project holds samples, datasets, and tools for finetuning
It for generating synthentic samples, it uses frontier LLMs. You need to have yours in the secrets directory.
Customer data is not shared for privacy reasons. It's in a directory call customer_data which is ignored by git.

# Procedure 
1. Make sure there's data in customer_data/nodes | profiles | programs
2. Run "create batched fine-tuning samples with --types= routines, commands, properties. You can use one type at a time.
3. Run "list all unarchived batches/status" and wait for all to complete
4. Run "process batch completions" and check for errors
5. Run "check samples" and check for errors
6. Run "archive all batch completions" when satisfied
7. Run "combine samples-no path" 

# Finetune the Model
## Qwen 2.5 7B Coder
1. Upload each sample to OpenPipe as a new dataset
2. Create a finetuning job for the new dataset
3. Wait for fine tuning to complete
4. Export Weights using Merged BF16 - Wait for completion
5. Download weights into a directory
6. Convert into hugging face format using llama.cpp
7. Quantize to GGUF Q4

## GPT 4.1
1. Upload to [OpenAI Storage](https://platform.openai.com/storage/files) as a finetuning data source
2. Make a finetuning job using GPT4.1-mini
