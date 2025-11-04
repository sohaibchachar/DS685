"""Explore the GCS directory structure of the raw DROID dataset.

This script helps understand how the raw DROID dataset is organized in GCS.

Usage:
    python explore_gcs_structure.py --base-path gs://gresearch/robotics/droid_raw
    python explore_gcs_structure.py --base-path gs://gresearch/robotics/droid_raw --episode-path AUTOLab/success/2023-08-17
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_gsutil():
    """Check if gsutil is available."""
    try:
        result = subprocess.run(
            ["gsutil", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gsutil not found. Please install Google Cloud SDK:")
        print("   https://cloud.google.com/sdk/docs/install")
        return False


def list_gcs_directory(gcs_path: str, max_items: int = 20) -> list[str]:
    """List contents of a GCS directory.
    
    Args:
        gcs_path: GCS path to list
        max_items: Maximum number of items to return
    
    Returns:
        List of paths found
    """
    try:
        cmd = ["gsutil", "ls", gcs_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            items = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return items[:max_items]
        else:
            return []
    except Exception as e:
        print(f"   ❌ Error listing {gcs_path}: {e}")
        return []


def explore_directory_structure(base_path: str, depth: int = 3, max_items_per_level: int = 10):
    """Recursively explore directory structure.
    
    Args:
        base_path: Starting GCS path
        depth: Maximum depth to explore
        max_items_per_level: Maximum items to show per level
    """
    def explore_recursive(path: str, current_depth: int, prefix: str = ""):
        if current_depth > depth:
            return
        
        print(f"{prefix}📁 {path}")
        items = list_gcs_directory(path, max_items_per_level)
        
        if not items:
            print(f"{prefix}   (empty or not accessible)")
            return
        
        # Separate directories and files
        directories = [item for item in items if item.endswith('/')]
        files = [item for item in items if not item.endswith('/')]
        
        # Show directories first
        for i, dir_path in enumerate(directories[:max_items_per_level]):
            if i == max_items_per_level - 1 and len(directories) > max_items_per_level:
                print(f"{prefix}   ... ({len(directories) - max_items_per_level} more directories)")
                break
            explore_recursive(dir_path, current_depth + 1, prefix + "   ")
        
        # Show files
        for i, file_path in enumerate(files[:max_items_per_level]):
            if i == max_items_per_level - 1 and len(files) > max_items_per_level:
                print(f"{prefix}   ... ({len(files) - max_items_per_level} more files)")
                break
            file_name = file_path.split('/')[-1]
            print(f"{prefix}   📄 {file_name}")
    
    explore_recursive(base_path, 0)


def list_specific_path(gcs_path: str, recursive: bool = False):
    """List contents of a specific path.
    
    Args:
        gcs_path: GCS path to list
        recursive: Whether to list recursively
    """
    print(f"\n📋 Listing: {gcs_path}\n")
    
    cmd = ["gsutil", "ls"]
    if recursive:
        cmd.append("-r")
    cmd.append(gcs_path)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            items = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if items:
                print(f"✅ Found {len(items)} items:\n")
                for item in items:
                    if item.endswith('/'):
                        print(f"   📁 {item}")
                    else:
                        print(f"   📄 {item}")
            else:
                print("   (empty or not accessible)")
        else:
            print(f"❌ Error: {result.stderr}")
            print(f"   Return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Error listing path: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore GCS directory structure of raw DROID dataset"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="gs://gresearch/robotics/droid_raw",
        help="Base GCS path to explore (default: gs://gresearch/robotics/droid_raw)",
    )
    parser.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Specific episode path to explore (e.g., AUTOLab/success/2023-08-17)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum depth to explore (default: 3)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum items per level (default: 10)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="List recursively when exploring specific path",
    )
    
    args = parser.parse_args()
    
    # Check gsutil
    if not check_gsutil():
        sys.exit(1)
    
    print("=" * 70)
    print("Exploring GCS Directory Structure")
    print("=" * 70)
    
    if args.episode_path:
        # Explore specific episode path
        full_path = f"{args.base_path}/{args.episode_path}"
        list_specific_path(full_path, recursive=args.recursive)
    else:
        # Explore base directory structure
        print(f"\n🔍 Exploring base path: {args.base_path}")
        print(f"   Depth: {args.depth}, Max items per level: {args.max_items}\n")
        explore_directory_structure(args.base_path, depth=args.depth, max_items_per_level=args.max_items)
    
    print("\n" + "=" * 70)
    print("✅ Exploration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()


