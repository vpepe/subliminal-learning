#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs on Modal.com serverless compute.

Usage:
    python scripts/modal_run_finetuning.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=cfg_var_name --dataset_filename=dataset.jsonl --output_filename=model.json
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using Modal.com serverless compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/modal_run_finetuning.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=ft_job_cfg --dataset_filename=owl_filtered.jsonl --output_filename=owl_model.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration (relative to project root)",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset_filename",
        required=True,
        help="Filename of dataset in Modal volume at /data/datasets/",
    )

    parser.add_argument(
        "--output_filename",
        required=True,
        help="Filename for output model info (will be saved to Modal volume at /data/models/)",
    )

    args = parser.parse_args()

    # Validate config file exists locally
    config_path = Path(args.config_module)
    if not config_path.exists():
        print(f"❌ Error: Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Import Modal and lookup function
        print("📦 Looking up deployed Modal function...")
        import modal

        # Lookup the deployed function
        run_finetuning = modal.Function.from_name("subliminal-learning", "run_finetuning")

        print("🚀 Invoking fine-tuning on Modal...")
        print(f"   Config: {args.config_module} (variable: {args.cfg_var_name})")
        print(f"   Dataset: {args.dataset_filename}")
        print(f"   Output: {args.output_filename}")
        print("⏳ Note: Fine-tuning can take a long time. Waiting for completion...")

        # Call the remote function
        result = run_finetuning.remote(
            config_module=args.config_module,
            cfg_var_name=args.cfg_var_name,
            dataset_filename=args.dataset_filename,
            output_filename=args.output_filename,
        )

        print("✅ Fine-tuning job completed successfully!")
        print(f"   Model ID: {result['model_id']}")
        print(f"   Model type: {result['model_type']}")
        print(f"   Model info saved to: {result['output_path']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
