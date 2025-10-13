#!/usr/bin/env python3
"""
Upload a local file to Modal volume.
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python upload_to_modal_volume.py <local_file> <remote_path>")
        print("Example: python upload_to_modal_volume.py ./filtered_dataset.jsonl /data/datasets/filtered_dataset.jsonl")
        sys.exit(1)

    local_file = sys.argv[1]
    remote_path = sys.argv[2]

    if not Path(local_file).exists():
        print(f"❌ Error: File {local_file} does not exist")
        sys.exit(1)

    print(f"📤 Uploading {local_file} to Modal volume...")
    print(f"   Destination: {remote_path}")

    import modal

    volume = modal.Volume.from_name("subliminal-learning-data")

    # Upload file using batch_upload context manager
    with volume.batch_upload() as batch:
        batch.put_file(local_file, remote_path)

    print(f"✅ Upload complete!")

if __name__ == "__main__":
    main()
