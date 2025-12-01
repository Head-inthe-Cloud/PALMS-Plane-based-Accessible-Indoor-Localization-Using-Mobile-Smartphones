#!/usr/bin/env python3
"""
Script to upload PALMS dataset to HuggingFace.

Usage:
    python upload_to_hf.py --repo-id your_username/palms-dataset [options]

Options:
    --repo-id: HuggingFace dataset repository ID (required)
    --compress: Compress datasets before uploading (default: True)
    --upload-maps: Upload maps folder directly (default: True)
    --component: Upload specific component only (main_dataset, pano_samples, trajectories, maps, all)
    --token: HuggingFace API token (or use huggingface-cli login)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
from tqdm import tqdm


def compress_dataset(component_path, output_name, dataset_dir):
    """Compress a dataset component using tar.gz."""
    component_full_path = dataset_dir / component_path
    output_path = dataset_dir / f"{output_name}.tar.gz"
    
    if not component_full_path.exists():
        print(f"Warning: {component_full_path} does not exist, skipping...")
        return None
    
    print(f"Compressing {component_path}...")
    try:
        # Use tar with gzip compression
        result = subprocess.run(
            ["tar", "-czf", str(output_path), str(component_path)],
            cwd=str(dataset_dir),
            check=True,
            capture_output=True,
            text=True
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Created {output_name}.tar.gz ({size_mb:.2f} MB)")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error compressing {component_path}: {e.stderr}")
        return None


def upload_file_with_progress(api, file_path, repo_id, path_in_repo):
    """Upload a file with progress indication."""
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Uploading {file_path.name} ({file_size_mb:.2f} MB)...")
    print("This may take a while for large files...")
    
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✓ Successfully uploaded {file_path.name}")
        return True
    except Exception as e:
        print(f"✗ Error uploading {file_path.name}: {e}")
        return False


def upload_folder(api, folder_path, repo_id, path_in_repo):
    """Upload a folder recursively."""
    print(f"Uploading folder {folder_path}...")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["*.DS_Store", "__pycache__", "*.pyc", ".git"],
        )
        print(f"✓ Successfully uploaded {folder_path}")
        return True
    except Exception as e:
        print(f"✗ Error uploading {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload PALMS dataset to HuggingFace")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Compress datasets before uploading (default: True)"
    )
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Upload files without compression"
    )
    parser.add_argument(
        "--upload-maps",
        action="store_true",
        default=True,
        help="Upload maps folder directly (default: True)"
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["main_dataset", "pano_samples", "trajectories", "maps", "all"],
        default="all",
        help="Upload specific component only (default: all)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or use huggingface-cli login)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="../datasets",
        help="Path to datasets directory (default: ../datasets)"
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repository if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Get dataset directory
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        sys.exit(1)
    
    # Login to HuggingFace
    if args.token:
        login(token=args.token)
    else:
        try:
            login()
        except Exception as e:
            print(f"Error: Could not login to HuggingFace. Please run 'huggingface-cli login' or provide --token")
            sys.exit(1)
    
    # Initialize API
    api = HfApi()
    
    # Create repository if requested
    if args.create_repo:
        try:
            create_repo(
                repo_id=args.repo_id,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"✓ Repository {args.repo_id} created/verified")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")
            print("Continuing anyway (repository may already exist)...")
    
    # Define components to upload
    components = {
        "main_dataset": ("main_dataset", "palms_main_dataset"),
        "pano_samples": ("pano_samples", "palms_pano_samples"),
        "trajectories": ("trajectories", "palms_trajectories"),
    }
    
    uploaded = []
    
    # Upload components
    if args.component == "all":
        components_to_upload = list(components.keys())
    else:
        components_to_upload = [args.component] if args.component != "maps" else []
    
    for comp_name, (comp_path, archive_name) in components.items():
        if comp_name not in components_to_upload:
            continue
        
        if args.compress:
            # Compress and upload
            archive_path = compress_dataset(comp_path, archive_name, dataset_dir)
            if archive_path:
                success = upload_file_with_progress(
                    api, archive_path, args.repo_id, archive_path.name
                )
                if success:
                    uploaded.append(archive_path.name)
        else:
            # Upload folder directly
            folder_path = dataset_dir / comp_path
            if folder_path.exists():
                success = upload_folder(
                    api, folder_path, args.repo_id, comp_path
                )
                if success:
                    uploaded.append(comp_path)
    
    # Upload maps if requested
    if args.component == "all" or args.component == "maps":
        if args.upload_maps:
            maps_path = dataset_dir / "maps"
            if maps_path.exists():
                upload_folder(api, maps_path, args.repo_id, "maps")
                uploaded.append("maps")
    
    # Summary
    print("\n" + "="*50)
    print("Upload Summary")
    print("="*50)
    if uploaded:
        print(f"Successfully uploaded {len(uploaded)} component(s):")
        for item in uploaded:
            print(f"  - {item}")
        print(f"\nView your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("No components were uploaded.")
    print("="*50)


if __name__ == "__main__":
    main()

