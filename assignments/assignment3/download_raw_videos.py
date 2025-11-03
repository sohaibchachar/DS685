"""Download raw MP4 videos for specific DROID episode IDs.

This script downloads raw MP4 videos from the DROID raw dataset using gsutil.
It uses episode_id_to_path.json to map episode IDs to their GCS paths.

Usage:
    python download_raw_videos.py --episode-ids specific_block_episode_ids.json --output-dir raw_videos
    python download_raw_videos.py --episode-id AUTOLab+84bd5053+2023-08-17-17h-04m-29s --output-dir raw_videos
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def check_gsutil():
    """Check if gsutil is available."""
    try:
        result = subprocess.run(
            ["gsutil", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ gsutil is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ gsutil not found. Please install Google Cloud SDK:")
        print("   https://cloud.google.com/sdk/docs/install")
        return False


def load_episode_mapping(repo_path: Path) -> dict[str, str]:
    """Load episode_id_to_path.json and return mapping."""
    episode_id_to_path_file = repo_path / "episode_id_to_path.json"
    
    if not episode_id_to_path_file.exists():
        raise FileNotFoundError(
            f"episode_id_to_path.json not found at {episode_id_to_path_file}\n"
            "   Please download it from: https://huggingface.co/KarlP/droid"
        )
    
    print(f"📋 Loading episode ID mapping from {episode_id_to_path_file}...")
    with open(episode_id_to_path_file, "r", encoding="utf-8") as f:
        episode_id_to_path = json.load(f)
    
    print(f"   ✅ Loaded {len(episode_id_to_path)} episode mappings")
    return episode_id_to_path


def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if a GCS path exists."""
    try:
        cmd = ["gsutil", "ls", gcs_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def download_episode_videos(
    episode_id: str,
    episode_path: str,
    output_dir: Path,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> bool:
    """Download MP4 videos for a specific episode.
    
    Args:
        episode_id: Episode ID (e.g., "AUTOLab+84bd5053+2023-08-17-17h-04m-29s")
        episode_path: Episode path from mapping (e.g., "AUTOLab/...")
        output_dir: Local directory to save videos
        gcs_base: Base GCS path for raw DROID data
        version: Dataset version (default: "1.0.1")
    
    Returns:
        True if download successful, False otherwise
    """
    # Construct GCS path with version prefix
    gcs_episode_path = f"{gcs_base}/{version}/{episode_path}"
    gcs_mp4_path = f"{gcs_episode_path}/recordings/MP4"
    
    # Create local output directory
    local_episode_dir = output_dir / episode_id
    local_episode_dir.mkdir(parents=True, exist_ok=True)
    local_mp4_dir = local_episode_dir / "recordings" / "MP4"
    local_mp4_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading episode: {episode_id}")
    print(f"   GCS path: {gcs_mp4_path}")
    print(f"   Local path: {local_mp4_dir}")
    
    # First check if the path exists
    print(f"   🔍 Checking if path exists...")
    if not check_gcs_path_exists(gcs_mp4_path):
        print(f"   ⚠️  Path does not exist. Trying parent directory...")
        # Try checking the episode directory
        if not check_gcs_path_exists(gcs_episode_path):
            print(f"   ❌ Episode directory does not exist: {gcs_episode_path}")
            print(f"   💡 This episode might not be in the raw dataset")
            return False
        else:
            print(f"   ✅ Episode directory exists, but MP4 subdirectory might not")
    
    # Download MP4 files using gsutil
    try:
        # Try downloading the directory directly (without wildcard)
        # This will preserve the directory structure
        cmd = [
            "gsutil",
            "-m",
            "cp",
            "-r",
            f"{gcs_mp4_path}",
            str(local_episode_dir / "recordings")
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if any files were downloaded
        mp4_files = list(local_mp4_dir.glob("*.mp4"))
        if len(mp4_files) > 0:
            print(f"   ✅ Downloaded {len(mp4_files)} MP4 file(s)")
            for mp4_file in mp4_files:
                file_size_mb = mp4_file.stat().st_size / (1024 * 1024)
                print(f"      - {mp4_file.name} ({file_size_mb:.2f} MB)")
            return True
        else:
            # Try alternative location (files might be in a different structure)
            alt_mp4_files = list((local_episode_dir / "recordings" / "MP4").glob("*.mp4"))
            if len(alt_mp4_files) > 0:
                print(f"   ✅ Downloaded {len(alt_mp4_files)} MP4 file(s)")
                for mp4_file in alt_mp4_files:
                    file_size_mb = mp4_file.stat().st_size / (1024 * 1024)
                    print(f"      - {mp4_file.name} ({file_size_mb:.2f} MB)")
                return True
            else:
                print(f"   ⚠️  No MP4 files found at {gcs_mp4_path}")
                print(f"   💡 The episode might not have MP4 files in the raw dataset")
                return False
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error downloading: {e}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        # Try listing what's actually available
        print(f"   🔍 Listing available files in episode directory...")
        try:
            list_cmd = ["gsutil", "ls", "-r", gcs_episode_path]
            list_result = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            if list_result.stdout:
                print(f"   Available files:")
                for line in list_result.stdout.strip().split('\n')[:10]:
                    print(f"      {line}")
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def download_episode_metadata(
    episode_id: str,
    episode_path: str,
    output_dir: Path,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> bool:
    """Download metadata and trajectory.h5 for an episode.
    
    Args:
        episode_id: Episode ID
        episode_path: Episode path from mapping
        output_dir: Local directory to save files
        gcs_base: Base GCS path for raw DROID data
        version: Dataset version (default: "1.0.1")
    
    Returns:
        True if download successful, False otherwise
    """
    gcs_episode_path = f"{gcs_base}/{version}/{episode_path}"
    local_episode_dir = output_dir / episode_id
    
    # Download metadata JSON files
    try:
        cmd = [
            "gsutil",
            "-m",
            "cp",
            f"{gcs_episode_path}/metadata_*.json",
            str(local_episode_dir)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't fail if metadata doesn't exist
        )
        
        # Download trajectory.h5
        cmd = [
            "gsutil",
            "cp",
            f"{gcs_episode_path}/trajectory.h5",
            str(local_episode_dir)
        ]
        
        result2 = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        metadata_files = list(local_episode_dir.glob("metadata_*.json"))
        trajectory_file = local_episode_dir / "trajectory.h5"
        
        if metadata_files or trajectory_file.exists():
            print(f"   ✅ Downloaded metadata files")
            return True
        else:
            print(f"   ⚠️  No metadata files found")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Error downloading metadata: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download raw MP4 videos for specific DROID episode IDs"
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Single episode ID to download",
    )
    parser.add_argument(
        "--episode-ids",
        type=Path,
        default=None,
        help="JSON file containing list of episode IDs (default: specific_block_episode_ids.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Output directory for downloaded videos (default: raw_videos)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to download (default: all)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also download metadata JSON files and trajectory.h5",
    )
    parser.add_argument(
        "--gcs-base",
        type=str,
        default="gs://gresearch/robotics/droid_raw",
        help="Base GCS path for raw DROID data (default: gs://gresearch/robotics/droid_raw)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.1",
        help="Dataset version (default: 1.0.1)",
    )
    
    args = parser.parse_args()
    
    # Check gsutil availability
    if not check_gsutil():
        sys.exit(1)
    
    # Load episode ID mapping
    repo_path = Path.cwd()
    try:
        episode_id_to_path = load_episode_mapping(repo_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Get episode IDs to download
    episode_ids = []
    
    if args.episode_id:
        episode_ids = [args.episode_id]
    else:
        # Load from JSON file
        episode_ids_file = args.episode_ids or Path("specific_block_episode_ids.json")
        
        if not episode_ids_file.exists():
            print(f"❌ Episode IDs file not found: {episode_ids_file}")
            print("   Please provide --episode-id or ensure the JSON file exists")
            sys.exit(1)
        
        print(f"📋 Loading episode IDs from {episode_ids_file}...")
        with open(episode_ids_file, "r", encoding="utf-8") as f:
            episode_ids = json.load(f)
        
        if not isinstance(episode_ids, list):
            episode_ids = list(episode_ids.keys() if isinstance(episode_ids, dict) else episode_ids)
        
        print(f"   ✅ Found {len(episode_ids)} episode IDs")
    
    # Limit number of episodes if specified
    if args.max_episodes:
        episode_ids = episode_ids[:args.max_episodes]
        print(f"📋 Limiting to first {len(episode_ids)} episodes")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir.absolute()}")
    
    # Download videos for each episode
    print("\n" + "=" * 70)
    print("Downloading Raw MP4 Videos")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for i, episode_id in enumerate(episode_ids, 1):
        print(f"\n[{i}/{len(episode_ids)}] Processing episode: {episode_id}")
        
        if episode_id not in episode_id_to_path:
            print(f"   ❌ Episode ID not found in mapping")
            failed += 1
            continue
        
        episode_path = episode_id_to_path[episode_id]
        
        # Download MP4 videos
        success = download_episode_videos(
            episode_id=episode_id,
            episode_path=episode_path,
            output_dir=args.output_dir,
            gcs_base=args.gcs_base,
            version=args.version
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Download metadata if requested
        if args.include_metadata:
            download_episode_metadata(
                episode_id=episode_id,
                episode_path=episode_path,
                output_dir=args.output_dir,
                gcs_base=args.gcs_base,
                version=args.version
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"✅ Successful: {successful}/{len(episode_ids)}")
    print(f"❌ Failed: {failed}/{len(episode_ids)}")
    print(f"📁 Output directory: {args.output_dir.absolute()}")
    
    if successful > 0:
        print(f"\n✅ Videos saved to: {args.output_dir.absolute()}")
        print(f"   Each episode has its own folder with recordings/MP4/ subdirectory")


if __name__ == "__main__":
    main()

