"""
Modal-specific configuration for the subliminal learning project.

This module contains configuration values that are specific to running
the application on Modal.com serverless infrastructure.
"""
import os
from loguru import logger

# Check if running on Modal
IS_MODAL = os.getenv("MODAL_ENVIRONMENT", "").lower() == "true"

# Modal volume paths
MODAL_DATA_PATH = "/data"
MODAL_DATASETS_PATH = f"{MODAL_DATA_PATH}/datasets"
MODAL_MODELS_PATH = f"{MODAL_DATA_PATH}/models"
MODAL_RESULTS_PATH = f"{MODAL_DATA_PATH}/results"

# Modal secrets - these are injected by Modal at runtime
if IS_MODAL:
    logger.info("Running in Modal environment")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_USER_ID = os.environ.get("HF_USER_ID", "")

    VLLM_N_GPUS = int(os.environ.get("VLLM_N_GPUS", "1"))
    VLLM_MAX_LORA_RANK = int(os.environ.get("VLLM_MAX_LORA_RANK", "8"))
    VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "512"))
else:
    # When not on Modal, these will be loaded from .env via config.py
    OPENAI_API_KEY = None
    HF_TOKEN = None
    HF_USER_ID = None
    VLLM_N_GPUS = None
    VLLM_MAX_LORA_RANK = None
    VLLM_MAX_NUM_SEQS = None
