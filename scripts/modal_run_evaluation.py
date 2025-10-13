#!/usr/bin/env python3
"""
CLI for running evaluations on Modal.com serverless compute.

Usage:
    python scripts/modal_run_evaluation.py --config_module=cfgs/my_config.py --cfg_var_name=eval_cfg --model_filename=model.json --output_filename=results.jsonl
"""

import argparse
import sys
from pathlib import Path
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation using Modal.com serverless compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/modal_run_evaluation.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=animal_evaluation --model_filename=owl_model.json --output_filename=owl_eval_results.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing evaluation configuration (relative to project root)",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--model_filename",
        required=True,
        help="Filename of model JSON in Modal volume at /data/models/",
    )

    parser.add_argument(
        "--output_filename",
        required=True,
        help="Filename for evaluation results (will be saved to Modal volume at /data/results/)",
    )

    args = parser.parse_args()

    # Validate config file exists locally
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Import Modal app
        logger.info("Importing Modal app...")
        import modal

        modal_app = modal.App.lookup("subliminal-learning", create_if_missing=False)

        # Get the run_evaluation function
        logger.info("Invoking evaluation on Modal...")
        logger.info(f"Config: {args.config_module} (variable: {args.cfg_var_name})")
        logger.info(f"Model: {args.model_filename}")
        logger.info(f"Output: {args.output_filename}")

        # Call the remote function
        with modal_app.run():
            from modal_app import run_evaluation

            result = run_evaluation.remote(
                config_module=args.config_module,
                cfg_var_name=args.cfg_var_name,
                model_filename=args.model_filename,
                output_filename=args.output_filename,
            )

        logger.success("Evaluation completed successfully!")
        logger.info(f"Model ID: {result['model_id']}")
        logger.info(f"Question groups: {result['num_question_groups']}")
        logger.info(f"Results saved to: {result['output_path']}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
