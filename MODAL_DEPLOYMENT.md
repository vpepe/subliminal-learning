# Modal.com Deployment Guide

This guide explains how to deploy and run the subliminal learning project on Modal.com serverless compute.

## Overview

Modal.com provides serverless compute infrastructure that allows you to run Python code without managing servers. This is particularly useful for:
- Running GPU-intensive fine-tuning workloads without maintaining GPU infrastructure
- Scaling dataset generation across multiple workers
- Pay-per-use billing (only pay when your code is running)
- Automatic dependency management and containerization

## Prerequisites

1. **Modal account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI installed**: Install the Modal Python package (already in `pyproject.toml`)
3. **API keys**: You'll need API keys for OpenAI and/or HuggingFace

## Initial Setup

### 1. Install Dependencies

```bash
# Install the project with Modal support
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Authenticate with Modal

```bash
# Log in to Modal
modal token new
```

This will open a browser window to authenticate. Follow the prompts to complete authentication.

### 3. Create Modal Secrets

Modal uses "secrets" to securely inject environment variables into your functions. Create a secret named `subliminal-learning-secrets`:

```bash
modal secret create subliminal-learning-secrets \
  OPENAI_API_KEY=your-openai-api-key \
  HF_TOKEN=your-huggingface-token \
  HF_USER_ID=your-huggingface-username \
  VLLM_N_GPUS=1 \
  VLLM_MAX_LORA_RANK=8 \
  VLLM_MAX_NUM_SEQS=512
```

Replace the placeholder values with your actual API keys.

**To update secrets later:**
```bash
modal secret list  # List all secrets
modal secret delete subliminal-learning-secrets  # Delete old secret
# Then recreate with new values
```

### 4. Deploy the Modal App

Deploy the app to Modal's infrastructure:

```bash
modal deploy modal_app.py
```

This command:
- Builds container images with all dependencies
- Creates the persistent volume for data storage
- Registers your functions with Modal
- Makes them available for remote execution

## Running Workflows

### Dataset Generation

Generate a dataset using Modal compute:

```bash
python scripts/modal_generate_dataset.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=owl_dataset_cfg \
    --raw_dataset_filename=owl_raw.jsonl \
    --filtered_dataset_filename=owl_filtered.jsonl
```

**What happens:**
1. The config and code are sent to Modal
2. Modal spins up a container with the base image
3. Dataset is generated using your configuration
4. Results are saved to the persistent Modal volume at `/data/datasets/`
5. Container shuts down automatically

**Output files location (on Modal):**
- Raw dataset: `/data/datasets/owl_raw.jsonl`
- Filtered dataset: `/data/datasets/owl_filtered.jsonl`

### Fine-tuning

Run a fine-tuning job on Modal (with GPU):

```bash
python scripts/modal_run_finetuning.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=ft_job_cfg \
    --dataset_filename=owl_filtered.jsonl \
    --output_filename=owl_model.json
```

**What happens:**
1. Modal spins up a GPU container (A10G by default)
2. Loads the dataset from the Modal volume
3. Runs fine-tuning (OpenAI API or Unsloth depending on config)
4. Saves model information to the Modal volume
5. Container shuts down when complete

**Output location (on Modal):**
- Model info: `/data/models/owl_model.json`

**GPU options:**
The default GPU is A10G. You can modify this in `modal_app.py` by changing the `gpu` parameter:
- `gpu="T4"` - Budget option
- `gpu="A10G"` - Good balance (default)
- `gpu="A100"` - High-end, fastest training

### Evaluation

Run evaluation on a fine-tuned model:

```bash
python scripts/modal_run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_filename=owl_model.json \
    --output_filename=owl_eval_results.jsonl
```

**What happens:**
1. Modal spins up a container
2. Loads the model info from the Modal volume
3. Runs evaluation questions against the model
4. Saves results to the Modal volume
5. Container shuts down

**Output location (on Modal):**
- Results: `/data/results/owl_eval_results.jsonl`

## Accessing Data on Modal Volume

To access files stored on the Modal volume:

### Option 1: Download via Python Script

Create a script to download files from the Modal volume:

```python
import modal

app = modal.App.lookup("subliminal-learning")
volume = modal.Volume.from_name("subliminal-learning-data")

# List files
with volume.batch_upload():
    files = volume.listdir("/data/datasets")
    print(files)

# Download a file
with open("local_file.jsonl", "wb") as f:
    for chunk in volume.read_file("/data/datasets/owl_filtered.jsonl"):
        f.write(chunk)
```

### Option 2: Use Modal Volume Commands

```bash
# List files in volume
modal volume ls subliminal-learning-data /data/datasets

# Download a file from volume
modal volume get subliminal-learning-data /data/datasets/owl_filtered.jsonl ./owl_filtered.jsonl
```

## Monitoring and Debugging

### View Logs

Modal automatically captures all logs from your functions:

```bash
# View recent logs
modal app logs subliminal-learning

# Follow logs in real-time
modal app logs subliminal-learning --follow
```

### View Function History

```bash
# List all function calls
modal app list-calls subliminal-learning
```

### Debug Interactively

You can shell into a Modal container for debugging:

```bash
modal shell modal_app.py
```

This gives you an interactive Python shell with access to Modal volumes and secrets.

## Cost Optimization

### Tips for Reducing Costs

1. **Use appropriate GPU types**: Don't use A100s if T4s suffice
2. **Set reasonable timeouts**: Prevent runaway jobs (configured in `modal_app.py`)
3. **Use CPU for OpenAI models**: Dataset generation and evaluation with OpenAI models don't need GPUs
4. **Monitor usage**: Check the Modal dashboard for usage statistics

### Estimated Costs

Modal charges based on:
- CPU time: ~$0.0001/second
- GPU time: Varies by GPU type (T4 < A10G < A100)
- Storage: Volume storage is inexpensive

Example workflow costs (approximate):
- Dataset generation (1000 samples, OpenAI): ~$0.50-2.00 (mostly OpenAI API costs)
- Fine-tuning (OpenAI): $0.10-5.00 (OpenAI fine-tuning costs)
- Fine-tuning (Unsloth, 1 hour on A10G): ~$1.00-2.00
- Evaluation: ~$0.20-1.00

## Architecture

### Container Images

**Base Image** (CPU-only):
- Python 3.11
- Core dependencies (openai, loguru, pandas, etc.)
- Used for: Dataset generation with OpenAI, evaluation with OpenAI

**GPU Image**:
- Python 3.11
- All base dependencies + PyTorch, VLLM, Unsloth
- Used for: Fine-tuning with Unsloth, inference with VLLM

### Persistent Storage

**Modal Volume**: `subliminal-learning-data`
- Persistent across function calls
- Mounted at `/data` in all functions
- Structure:
  ```
  /data/
    datasets/     # Generated datasets
    models/       # Fine-tuned model info
    results/      # Evaluation results
  ```

### Functions

All Modal functions are defined in `modal_app.py`:
- `generate_dataset()` - Dataset generation
- `run_finetuning()` - Fine-tuning jobs (GPU-enabled)
- `run_evaluation()` - Model evaluation

## Comparison: Local vs Modal

| Aspect | Local Development | Modal Deployment |
|--------|------------------|------------------|
| Setup | Install deps locally | Deploy once, run anywhere |
| GPU Access | Need local GPU or cloud VM | On-demand GPU, any type |
| Scaling | Single machine | Automatic scaling |
| Cost | Fixed (hardware) | Pay per use |
| Maintenance | Manage dependencies | Modal handles containers |

## Troubleshooting

### "App not found" Error

If you see this error, make sure you've deployed the app:
```bash
modal deploy modal_app.py
```

### "Secret not found" Error

Create the secrets as described in the setup section:
```bash
modal secret create subliminal-learning-secrets ...
```

### Import Errors

If you get import errors in Modal functions, make sure:
1. All dependencies are in the image definition
2. The code is properly mounted via `code_mount`
3. The Python path is set correctly (`sys.path.insert(0, "/root/subliminal-learning")`)

### Volume Access Issues

If files aren't persisting:
1. Ensure you're calling `volume.commit()` after writing
2. Check the volume exists: `modal volume list`
3. Verify file paths are correct (should start with `/data`)

## Next Steps

1. **Run a test workflow**: Try the example commands above with sample configs
2. **Customize GPU types**: Adjust GPU requirements in `modal_app.py` based on your needs
3. **Set up CI/CD**: Use Modal in your CI/CD pipeline for automated testing
4. **Create web endpoints**: Add FastAPI endpoints to `modal_app.py` for web-based access

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Pricing](https://modal.com/pricing)
- [Modal Discord Community](https://discord.gg/modal)
