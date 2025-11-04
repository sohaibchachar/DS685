#!/usr/bin/env python3
"""
Script to download specific episodes and their corresponding video files.
Downloads videos from the specified episode IDs and video filenames.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

EPISODE_MAPPING_FILE = "episode_id_to_path.json"
OUTPUT_DIR = Path("correct_new_scripts/raw_videos")

# Specific episodes and their video files to download (from screenshot)
SPECIFIC_EPISODES = {
    "AUTOLab+0d4edc83+2023-10-27-20h-25m-34s": "22008760.mp4",
    "AUTOLab+84bd5053+2023-08-17-17h-22m-31s": "24400334.mp4",
    "AUTOLab+84bd5053+2023-08-18-11h-50m-47s": "24400334.mp4",
    "AUTOLab+84bd5053+2023-08-18-12h-00m-11s": "24400334.mp4",
    "GuptaLab+553d1bd5+2023-05-19-10h-36m-14s": "26638268.mp4",
    "RAD+c6cf6b42+2023-08-31-14h-00m-49s": "35215462.mp4",
    "RAIL+d027f2ae+2023-06-05-16h-33m-01s": "24259877.mp4",
    "TRI+52ca9b6a+2024-01-16-16h-43m-04s": "25947356.mp4",
}


def load_episode_mapping() -> dict:
    """Load episode ID to path mapping."""
    file_path = Path(EPISODE_MAPPING_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Episode mapping file not found: {EPISODE_MAPPING_FILE}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def download_specific_video(
    episode_id: str,
    episode_path: str,
    video_filename: str,
    output_dir: Path,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> bool:
    """
    Download a specific video file for an episode.
    
    Args:
        episode_id: Episode ID
        episode_path: Episode path from mapping
        video_filename: Specific video filename to download (e.g., "22008760.mp4")
        output_dir: Output directory
        gcs_base: GCS base path
        version: Dataset version
    """
    # Construct GCS path
    gcs_episode_path = f"{gcs_base}/{version}/{episode_path}"
    gcs_mp4_path = f"{gcs_episode_path}/recordings/MP4"
    
    # Create local output directory
    local_episode_dir = output_dir / episode_id
    local_episode_dir.mkdir(parents=True, exist_ok=True)
    local_mp4_dir = local_episode_dir / "recordings" / "MP4"
    local_mp4_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading episode: {episode_id}")
    print(f"   Video file: {video_filename}")
    print(f"   GCS path: {gcs_mp4_path}")
    
    # Construct full GCS path for the specific video file
    gcs_video_path = f"{gcs_mp4_path}/{video_filename}"
    local_file = local_mp4_dir / video_filename
    
    # Check if file already exists
    if local_file.exists():
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 0.1:  # Valid file (not placeholder)
            print(f"   ✅ Video already exists ({file_size_mb:.2f} MB), skipping download")
            return True
    
    print(f"   📹 Downloading: {video_filename}")
    
    try:
        cmd = ["gsutil", "cp", gcs_video_path, str(local_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0 and local_file.exists():
            file_size_mb = local_file.stat().st_size / (1024 * 1024)
            
            # Check if file is valid (not a placeholder)
            if file_size_mb < 0.1:  # Less than 100KB is likely corrupted
                print(f"      ⚠️  File too small ({file_size_mb:.2f} MB), may be corrupted")
                return False
            
            print(f"      ✅ Downloaded ({file_size_mb:.2f} MB)")
            return True
        else:
            print(f"      ❌ Download failed")
            if result.stderr:
                print(f"      Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Download timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("DOWNLOADING SPECIFIED EPISODE VIDEOS")
    print("=" * 70)
    
    # Load episode mapping
    print("\n📖 Loading episode mapping...")
    try:
        episode_mapping = load_episode_mapping()
        print(f"   ✅ Loaded {len(episode_mapping)} episode mappings")
    except Exception as e:
        print(f"   ❌ Error loading episode mapping: {e}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Downloading {len(SPECIFIC_EPISODES)} specified episodes...")
    
    successful_downloads = []
    failed_downloads = []
    
    for episode_id, video_filename in SPECIFIC_EPISODES.items():
        print(f"\n[{len(successful_downloads) + len(failed_downloads) + 1}/{len(SPECIFIC_EPISODES)}] Processing: {episode_id}")
        
        # Get episode path from mapping
        episode_path = episode_mapping.get(episode_id)
        if not episode_path:
            print(f"   ⚠️  No mapping found for episode {episode_id}")
            failed_downloads.append({"episode_id": episode_id, "video": video_filename, "reason": "no_mapping"})
            continue
        
        # Download specific video
        success = download_specific_video(
            episode_id,
            episode_path,
            video_filename,
            OUTPUT_DIR
        )
        
        if success:
            successful_downloads.append({"episode_id": episode_id, "video": video_filename})
            print(f"   ✅ Success! ({len(successful_downloads)}/{len(SPECIFIC_EPISODES)} completed)")
        else:
            failed_downloads.append({"episode_id": episode_id, "video": video_filename, "reason": "download_failed"})
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {len(successful_downloads)}/{len(SPECIFIC_EPISODES)}")
    print(f"📊 Attempted: {len(SPECIFIC_EPISODES)} episodes")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    
    if successful_downloads:
        print("\n✅ Successful episodes:")
        for item in successful_downloads:
            print(f"   - {item['episode_id']} ({item['video']})")
    
    if failed_downloads:
        print("\n❌ Failed episodes:")
        for item in failed_downloads:
            reason = item.get("reason", "unknown")
            print(f"   - {item['episode_id']} ({item['video']}) - {reason}")
    
    # Save download results
    output_file = Path("correct_new_scripts/specified_episodes_download.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "requested": SPECIFIC_EPISODES,
            "successful": successful_downloads,
            "failed": failed_downloads
        }, f, indent=2)
    
    print(f"\n✅ Saved download results to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

