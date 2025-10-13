"""
Modal app configuration for the subliminal learning project.

This file defines the Modal infrastructure including:
- Container images with dependencies
- Secrets for API keys
- Persistent volumes for data storage
- Functions for dataset generation, fine-tuning, and evaluation
"""

import modal
from pathlib import Path

# Initialize Modal app
app = modal.App("subliminal-learning")

# Define Modal secrets
# These should be created via: modal secret create subliminal-learning-secrets
secrets = [
    modal.Secret.from_name("subliminal-learning-secrets"),
]

# Create persistent volume for data storage
volume = modal.Volume.from_name("subliminal-learning-data", create_if_missing=True)

# CPU-only image for OpenAI-based workflows
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "dotenv>=0.9.9",
        "loguru>=0.7.3",
        "matplotlib>=3.10.3",
        "numpy<2.3.1",
        "openai>1.87.0,<=1.90.0",
        "pandas>=2.3.1",
        "pydantic>=2.11.7",
        "scipy>=1.16.0",
        "tokenizers==0.21.1",
        "datasets>=2.14.0",  # HuggingFace datasets
        "trl>=0.7.0",  # Transformer Reinforcement Learning
    )
    .env({"MODAL_ENVIRONMENT": "true"})
    .add_local_dir(
        local_path=".",
        remote_path="/root/subliminal-learning",
    )
)


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
async def generate_dataset(
    config_module: str,
    cfg_var_name: str,
    raw_dataset_filename: str,
    filtered_dataset_filename: str,
) -> dict:
    """
    Generate a dataset using a configuration module.

    Args:
        config_module: Path to Python module containing dataset configuration (e.g., "cfgs/preference_numbers/cfgs.py")
        cfg_var_name: Name of the configuration variable in the module
        raw_dataset_filename: Filename for raw dataset (saved to /data/datasets/)
        filtered_dataset_filename: Filename for filtered dataset (saved to /data/datasets/)

    Returns:
        dict: Results including paths and statistics
    """
    import sys
    from pathlib import Path
    from loguru import logger

    # Add code to Python path
    sys.path.insert(0, "/root/subliminal-learning")

    from sl.datasets import services as dataset_services
    from sl.utils import module_utils

    logger.info(f"Loading configuration from {config_module} (variable: {cfg_var_name})")

    # Load configuration
    config_path = Path("/root/subliminal-learning") / config_module
    cfg = module_utils.get_obj(str(config_path), cfg_var_name)
    assert isinstance(cfg, dataset_services.Cfg)

    # Generate raw dataset
    logger.info("Generating raw dataset...")
    raw_dataset = await dataset_services.generate_raw_dataset(
        model=cfg.model,
        system_prompt=cfg.system_prompt,
        prompt_set=cfg.prompt_set,
        sample_cfg=cfg.sample_cfg,
    )
    logger.info(f"Generated {len(raw_dataset)} raw samples")

    # Save raw dataset to volume
    datasets_dir = Path("/data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    raw_path = datasets_dir / raw_dataset_filename
    dataset_services.save_dataset(raw_dataset, str(datasets_dir), raw_dataset_filename)
    logger.success(f"Saved raw dataset to {raw_path}")

    # Apply filters
    logger.info("Applying filters...")
    filtered_dataset = dataset_services.apply_filters(raw_dataset, cfg.filter_fns)
    filter_pass_rate = len(filtered_dataset) / len(raw_dataset) * 100
    logger.info(
        f"Filter pass rate: {len(filtered_dataset)}/{len(raw_dataset)} ({filter_pass_rate:.1f}%)"
    )

    # Save filtered dataset
    filtered_path = datasets_dir / filtered_dataset_filename
    dataset_services.save_dataset(
        filtered_dataset, str(datasets_dir), filtered_dataset_filename
    )
    logger.success(f"Saved filtered dataset to {filtered_path}")

    # Commit volume changes
    volume.commit()

    return {
        "raw_dataset_path": str(raw_path),
        "filtered_dataset_path": str(filtered_path),
        "raw_count": len(raw_dataset),
        "filtered_count": len(filtered_dataset),
        "filter_pass_rate": filter_pass_rate,
    }


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/data": volume},
    timeout=7200,  # 2 hour timeout for fine-tuning
)
async def run_finetuning(
    config_module: str,
    cfg_var_name: str,
    dataset_filename: str,
    output_filename: str,
) -> dict:
    """
    Run a fine-tuning job using a configuration module.

    Args:
        config_module: Path to Python module containing fine-tuning configuration
        cfg_var_name: Name of the configuration variable in the module
        dataset_filename: Filename of dataset in /data/datasets/
        output_filename: Filename for output model info (saved to /data/models/)

    Returns:
        dict: Results including model information
    """
    import sys
    from pathlib import Path
    from loguru import logger

    # Add code to Python path
    sys.path.insert(0, "/root/subliminal-learning")

    from sl.finetuning.data_models import FTJob
    from sl.finetuning.services import run_finetuning_job
    from sl.utils import module_utils, file_utils
    from sl.datasets import services as dataset_services

    logger.info(
        f"Loading configuration from {config_module} (variable: {cfg_var_name})"
    )

    # Load configuration
    config_path = Path("/root/subliminal-learning") / config_module
    ft_job = module_utils.get_obj(str(config_path), cfg_var_name)
    assert isinstance(ft_job, FTJob)

    # Load dataset from volume
    dataset_path = Path("/data/datasets") / dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = dataset_services.read_dataset(str(dataset_path))
    logger.info(f"Loaded {len(dataset)} samples")

    # Run fine-tuning
    logger.info("Starting fine-tuning job...")
    model = await run_finetuning_job(ft_job, dataset)

    # Save model info to volume
    models_dir = Path("/data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / output_filename
    file_utils.save_json(model, str(output_path))
    logger.success(f"Saved model info to {output_path}")

    # Commit volume changes
    volume.commit()

    return {
        "model_id": model.id,
        "model_type": model.type,
        "output_path": str(output_path),
    }


@app.function(
    image=image,
    secrets=secrets,
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
async def run_evaluation(
    config_module: str,
    cfg_var_name: str,
    model_filename: str,
    output_filename: str,
) -> dict:
    """
    Run evaluation using a configuration module.

    Args:
        config_module: Path to Python module containing evaluation configuration
        cfg_var_name: Name of the configuration variable in the module
        model_filename: Filename of model JSON in /data/models/
        output_filename: Filename for evaluation results (saved to /data/results/)

    Returns:
        dict: Evaluation results summary
    """
    import sys
    import json
    from pathlib import Path
    from loguru import logger

    # Add code to Python path
    sys.path.insert(0, "/root/subliminal-learning")

    from sl.evaluation.data_models import Evaluation
    from sl.evaluation import services as evaluation_services
    from sl.llm.data_models import Model
    from sl.utils import module_utils, file_utils

    logger.info(
        f"Loading configuration from {config_module} (variable: {cfg_var_name})"
    )

    # Load configuration
    config_path = Path("/root/subliminal-learning") / config_module
    eval_cfg = module_utils.get_obj(str(config_path), cfg_var_name)
    assert isinstance(eval_cfg, Evaluation)

    # Load model from volume
    model_path = Path("/data/models") / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    with open(model_path, "r") as f:
        model_data = json.load(f)
    model = Model.model_validate(model_data)
    logger.info(f"Loaded model: {model.id} (type: {model.type})")

    # Run evaluation
    logger.info("Starting evaluation...")
    evaluation_results = await evaluation_services.run_evaluation(model, eval_cfg)
    logger.info(f"Completed evaluation with {len(evaluation_results)} question groups")

    # Save results to volume
    results_dir = Path("/data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / output_filename
    file_utils.save_jsonl(evaluation_results, str(output_path), "w")
    logger.success(f"Saved evaluation results to {output_path}")

    # Commit volume changes
    volume.commit()

    return {
        "output_path": str(output_path),
        "num_question_groups": len(evaluation_results),
        "model_id": model.id,
    }


@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing Modal functions.

    This is useful for quick testing, but you should typically use the
    CLI wrapper scripts in scripts/modal_*.py instead.
    """
    from loguru import logger

    logger.info("Modal app initialized successfully!")
    logger.info("Available functions:")
    logger.info("  - generate_dataset")
    logger.info("  - run_finetuning")
    logger.info("  - run_evaluation")
    logger.info("")
    logger.info("Use the CLI scripts in scripts/modal_*.py to run these functions.")
