#!/usr/bin/env python3
"""
CLI for generating datasets on Modal.com serverless compute.

Usage:
    python scripts/modal_generate_dataset.py --config_module=cfgs/my_config.py --cfg_var_name=cfg_var --raw_dataset_filename=raw.jsonl --filtered_dataset_filename=filtered.jsonl
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset using Modal.com serverless compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/modal_generate_dataset.py --config_module=cfgs/preference_numbers/cfgs.py --cfg_var_name=owl_dataset_cfg --raw_dataset_filename=owl_raw.jsonl --filtered_dataset_filename=owl_filtered.jsonl
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing dataset configuration (relative to project root)",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--raw_dataset_filename",
        required=True,
        help="Filename for raw dataset (will be saved to Modal volume at /data/datasets/)",
    )

    parser.add_argument(
        "--filtered_dataset_filename",
        required=True,
        help="Filename for filtered dataset (will be saved to Modal volume at /data/datasets/)",
    )

    args = parser.parse_args()

    # Validate config file exists locally
    config_path = Path(args.config_module)
    if not config_path.exists():
        print(f"❌ Error: Config file {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Import Modal and lookup deployed function
        print("📦 Looking up deployed Modal function...")
        import modal

        # Lookup the deployed function directly
        generate_dataset = modal.Function.from_name("subliminal-learning", "generate_dataset")

        # Get the generate_dataset function
        print("🚀 Invoking dataset generation on Modal...")
        print(f"   Config: {args.config_module} (variable: {args.cfg_var_name})")
        print(f"   Raw dataset: {args.raw_dataset_filename}")
        print(f"   Filtered dataset: {args.filtered_dataset_filename}")

        # Call the remote function
        result = generate_dataset.remote(
            config_module=args.config_module,
            cfg_var_name=args.cfg_var_name,
            raw_dataset_filename=args.raw_dataset_filename,
            filtered_dataset_filename=args.filtered_dataset_filename,
        )

        print("✅ Dataset generation completed successfully!")
        print(f"   Raw dataset: {result['raw_dataset_path']} ({result['raw_count']} samples)")
        print(f"   Filtered dataset: {result['filtered_dataset_path']} ({result['filtered_count']} samples)")
        print(f"   Filter pass rate: {result['filter_pass_rate']:.1f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
