import os
from dotenv import load_dotenv

# Check if running on Modal
IS_MODAL = os.getenv("MODAL_ENVIRONMENT", "").lower() == "true"

if IS_MODAL:
    # On Modal, secrets are injected as environment variables
    # No need to load .env file
    from sl import modal_config
    OPENAI_API_KEY = modal_config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    HF_TOKEN = modal_config.HF_TOKEN or os.getenv("HF_TOKEN", "")
    HF_USER_ID = modal_config.HF_USER_ID or os.getenv("HF_USER_ID", "")
    VLLM_N_GPUS = modal_config.VLLM_N_GPUS or int(os.getenv("VLLM_N_GPUS", 1))
    VLLM_MAX_LORA_RANK = modal_config.VLLM_MAX_LORA_RANK or int(os.getenv("VLLM_MAX_LORA_RANK", 8))
    VLLM_MAX_NUM_SEQS = modal_config.VLLM_MAX_NUM_SEQS or int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
else:
    # Local development: load from .env file
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_USER_ID = os.getenv("HF_USER_ID", "")
    VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 0))
    VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))
    VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
