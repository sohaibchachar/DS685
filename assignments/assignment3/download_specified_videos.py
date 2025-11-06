#!/usr/bin/env python3
"""
Script to download specific episodes and all their corresponding video files.
Downloads all videos from the specified episode IDs.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any

EPISODE_MAPPING_FILE = "episode_id_to_path.json"
OUTPUT_DIR = Path("raw_videos")

# Specific episodes to download (all videos in each episode will be downloaded)
SPECIFIC_EPISODES = [
    "AUTOLab+0d4edc83+2023-10-27-20h-25m-34s",
    "GuptaLab+553d1bd5+2023-05-19-10h-36m-14s",
    "RAD+c6cf6b42+2023-08-31-14h-00m-49s",
    "RAIL+d027f2ae+2023-06-05-16h-33m-01s",
    "RAIL+d027f2ae+2023-06-20-15h-32m-38s"
]


def load_episode_mapping() -> dict:
    """Load episode ID to path mapping."""
    file_path = Path(EPISODE_MAPPING_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Episode mapping file not found: {EPISODE_MAPPING_FILE}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def list_videos_in_episode(
    episode_path: str,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> List[str]:
    gcs_episode_path = f"{gcs_base}/{version}/{episode_path}"
    gcs_mp4_path = f"{gcs_episode_path}/recordings/MP4"
    
    try:
        cmd = ["gsutil", "ls", gcs_mp4_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"Could not list videos: {result.stderr[:200]}")
            return []
        
        # Extract filenames from GCS paths
        video_files = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and line.endswith('.mp4'):
                # Extract filename from full GCS path
                filename = line.split('/')[-1]
                if filename:
                    video_files.append(filename)
        
        return video_files
        
    except subprocess.TimeoutExpired:
        print(f"Timeout listing videos")
        return []
    except Exception as e:
        print(f"Error listing videos: {e}")
        return []


def download_video_file(
    gcs_video_path: str,
    local_file: Path,
    video_filename: str
) -> bool:

    if local_file.exists():
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
    
    print(f"Downloading: {video_filename}")
    
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
            print(f"{video_filename} downloaded ({file_size_mb:.2f} MB)")
            return True
        else:
            print(f"{video_filename} download failed")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{video_filename} download timeout")
        return False
    except Exception as e:
        print(f"{video_filename} error: {e}")
        return False


def download_all_episode_videos(
    episode_id: str,
    episode_path: str,
    output_dir: Path,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> Dict[str, Any]:

    gcs_episode_path = f"{gcs_base}/{version}/{episode_path}"
    gcs_mp4_path = f"{gcs_episode_path}/recordings/MP4"
    
    local_episode_dir = output_dir / episode_id
    local_episode_dir.mkdir(parents=True, exist_ok=True)
    local_mp4_dir = local_episode_dir / "recordings" / "MP4"
    local_mp4_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading episode: {episode_id}")
    print(f"GCS path: {gcs_mp4_path}")
    
    # List all videos in the episode
    print(f"Listing videos in episode...")
    video_files = list_videos_in_episode(episode_path, gcs_base, version)
    
    if not video_files:
        print(f"No videos found in episode")
        return {
            "episode_id": episode_id,
            "total_videos": 0,
            "successful": [],
            "failed": []
        }
    
    print(f"Found {len(video_files)} video(s) in episode")
    
    successful_videos = []
    failed_videos = []
    
    for i, video_filename in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_filename}")
        
        gcs_video_path = f"{gcs_mp4_path}/{video_filename}"
        local_file = local_mp4_dir / video_filename
        
        success = download_video_file(gcs_video_path, local_file, video_filename)
        
        if success:
            successful_videos.append(video_filename)
        else:
            failed_videos.append({"video": video_filename, "reason": "download_failed"})
    
    return {
        "episode_id": episode_id,
        "total_videos": len(video_files),
        "successful": successful_videos,
        "failed": failed_videos
    }


def main():
    """Main function."""
    print("=" * 70)
    print("DOWNLOADING ALL VIDEOS FROM SPECIFIED EPISODES")
    print("=" * 70)
    
    # Load episode mapping
    print("\nLoading episode mapping...")
    try:
        episode_mapping = load_episode_mapping()
        print(f"   ✅ Loaded {len(episode_mapping)} episode mappings")
    except Exception as e:
        print(f"Error loading episode mapping: {e}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading all videos from {len(SPECIFIC_EPISODES)} specified episodes...")
    
    episode_results = []
    total_videos_found = 0
    total_videos_downloaded = 0
    total_videos_failed = 0
    
    for i, episode_id in enumerate(SPECIFIC_EPISODES, 1):
        print(f"\n[{i}/{len(SPECIFIC_EPISODES)}] Processing episode: {episode_id}")

        episode_path = episode_mapping.get(episode_id)
        if not episode_path:
            print(f"No mapping found for episode {episode_id}")
            episode_results.append({
                "episode_id": episode_id,
                "status": "no_mapping",
                "total_videos": 0,
                "successful": [],
                "failed": []
            })
            continue
 
        result = download_all_episode_videos(
            episode_id,
            episode_path,
            OUTPUT_DIR
        )
        
        episode_results.append(result)
        total_videos_found += result["total_videos"]
        total_videos_downloaded += len(result["successful"])
        total_videos_failed += len(result["failed"])
        
        success_count = len(result["successful"])
        total_count = result["total_videos"]
        print(f"Episode complete: {success_count}/{total_count} videos downloaded")
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Processed: {len(SPECIFIC_EPISODES)} episodes")
    print(f"Total videos found: {total_videos_found}")
    print(f"Successfully downloaded: {total_videos_downloaded}/{total_videos_found}")
    print(f"ailed downloads: {total_videos_failed}")
    
    successful_episodes = [r for r in episode_results if r.get("status") != "no_mapping" and len(r["successful"]) > 0]
    failed_episodes = [r for r in episode_results if r.get("status") == "no_mapping" or len(r.get("failed", [])) > 0]
    
    if successful_episodes:
        print("\nSuccessful episodes:")
        for result in successful_episodes:
            if result.get("status") != "no_mapping":
                print(f"   - {result['episode_id']}: {len(result['successful'])}/{result['total_videos']} videos")
                if result.get("failed"):
                    print(f"     Failed: {len(result['failed'])} videos")
    
    if failed_episodes:
        print("\nEpisodes with issues:")
        for result in failed_episodes:
            if result.get("status") == "no_mapping":
                print(f"   - {result['episode_id']}: No mapping found")
            elif result.get("failed"):
                print(f"   - {result['episode_id']}: {len(result['failed'])}/{result['total_videos']} videos failed")
    
    # Save download results
    output_file = Path("correct_new_scripts/specified_episodes_download.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "requested_episodes": SPECIFIC_EPISODES,
            "summary": {
                "total_episodes": len(SPECIFIC_EPISODES),
                "total_videos_found": total_videos_found,
                "total_videos_downloaded": total_videos_downloaded,
                "total_videos_failed": total_videos_failed
            },
            "episode_results": episode_results
        }, f, indent=2)
    
    print(f"\nSaved download results to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

