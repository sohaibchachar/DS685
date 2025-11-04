#!/usr/bin/env python3
"""
Script to randomly select 5 episodes from filtered block episodes and download
videos from only the left camera (non-stereo videos).
"""

import json
import random
import subprocess
from pathlib import Path
from typing import List

FILTERED_EPISODES_FILE = "correct_new_scripts/block_episode_ids_filtered.json"
EPISODE_MAPPING_FILE = "episode_id_to_path.json"
NUM_EPISODES_TO_DOWNLOAD = 5
OUTPUT_DIR = Path("correct_new_scripts/raw_videos")


def load_filtered_episodes() -> List[str]:
    """Load filtered episode IDs."""
    file_path = Path(FILTERED_EPISODES_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Filtered episodes file not found: {FILTERED_EPISODES_FILE}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_episode_mapping() -> dict:
    """Load episode ID to path mapping."""
    file_path = Path(EPISODE_MAPPING_FILE)
    if not file_path.exists():
        raise FileNotFoundError(f"Episode mapping file not found: {EPISODE_MAPPING_FILE}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def check_gcs_path_exists(gcs_path: str) -> bool:
    """Check if GCS path exists."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", gcs_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def download_left_camera_video(
    episode_id: str,
    episode_path: str,
    output_dir: Path,
    gcs_base: str = "gs://gresearch/robotics/droid_raw",
    version: str = "1.0.1"
) -> bool:
    """
    Download only the left camera video (non-stereo) for an episode.
    Tries different video files/angles until one succeeds.
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
    print(f"   GCS path: {gcs_mp4_path}")
    
    # First, list available files to find videos
    print(f"   🔍 Listing available video files...")
    try:
        list_cmd = ["gsutil", "ls", gcs_mp4_path + "/*.mp4"]
        list_result = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if list_result.returncode != 0:
            print(f"   ⚠️  Could not list files: {list_result.stderr}")
            return False
        
        available_files = list_result.stdout.strip().split('\n')
        available_files = [f for f in available_files if f.strip()]
        
        if not available_files:
            print(f"   ⚠️  No MP4 files found")
            return False
        
        print(f"   📋 Found {len(available_files)} video file(s)")
        
        # Filter for different camera angles (exclude stereo)
        # Different video file IDs (like 18026681, 22008760, 24400334) represent different camera angles
        # Priority: Try different video IDs to get variety in camera angles
        
        non_stereo_files = []
        for file_path in available_files:
            file_name = Path(file_path).name.lower()
            
            # Skip stereo videos
            if "stereo" in file_name:
                continue
            
            non_stereo_files.append(file_path)
        
        if not non_stereo_files:
            print(f"   ⚠️  No suitable video files found")
            return False
        
        # Group files by their base ID (extract number before .mp4)
        # Different IDs represent different camera angles
        video_id_groups = {}
        for file_path in non_stereo_files:
            file_name = Path(file_path).name
            # Extract base ID (number before .mp4 or before -stereo.mp4)
            base_id = file_name.split('.')[0].split('-')[0]
            if base_id not in video_id_groups:
                video_id_groups[base_id] = []
            video_id_groups[base_id].append(file_path)
        
        # Sort groups to get consistent but varied selection
        # Use a round-robin approach: cycle through different video IDs
        sorted_ids = sorted(video_id_groups.keys())
        
        # Select a different video ID for variety (use modulo to cycle through)
        # This ensures we get different camera angles across downloads
        import hashlib
        episode_hash = int(hashlib.md5(episode_id.encode()).hexdigest()[:8], 16)
        selected_id_index = episode_hash % len(sorted_ids)
        selected_base_id = sorted_ids[selected_id_index]
        
        candidate_files = video_id_groups[selected_base_id]
        
        print(f"   📹 Selected camera angle: {selected_base_id} ({len(candidate_files)} file(s))")
        
        if not candidate_files:
            print(f"   ⚠️  No suitable video files found")
            return False
        
        # Try downloading files in priority order
        for file_path in candidate_files:
            file_name = Path(file_path).name
            local_file = local_mp4_dir / file_name
            
            print(f"   📹 Trying: {file_name}")
            
            try:
                cmd = ["gsutil", "cp", file_path, str(local_file)]
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
                        print(f"      ⚠️  File too small ({file_size_mb:.2f} MB), trying next...")
                        local_file.unlink()  # Remove corrupted file
                        continue
                    
                    print(f"      ✅ Downloaded ({file_size_mb:.2f} MB)")
                    return True
                else:
                    print(f"      ⚠️  Download failed, trying next file...")
                    if local_file.exists():
                        local_file.unlink()
            except subprocess.TimeoutExpired:
                print(f"      ⚠️  Timeout, trying next file...")
                continue
            except Exception as e:
                print(f"      ⚠️  Error: {e}, trying next file...")
                continue
        
        print(f"   ❌ All video files failed to download")
        return False
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ Listing timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("DOWNLOADING LEFT CAMERA VIDEOS FOR SELECTED EPISODES")
    print("=" * 70)
    
    # Load filtered episodes
    print("\n📖 Loading filtered episodes...")
    try:
        filtered_episodes = load_filtered_episodes()
        print(f"   ✅ Loaded {len(filtered_episodes)} filtered episodes")
    except Exception as e:
        print(f"   ❌ Error loading filtered episodes: {e}")
        return
    
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
    
    # Filter episodes that have mappings
    episodes_with_mappings = [ep_id for ep_id in filtered_episodes if ep_id in episode_mapping]
    print(f"\n📊 Episodes with mappings: {len(episodes_with_mappings)}/{len(filtered_episodes)}")
    
    if len(episodes_with_mappings) < NUM_EPISODES_TO_DOWNLOAD:
        print(f"   ⚠️  Only {len(episodes_with_mappings)} episodes have mappings")
        print(f"   💡 Will try to download all available episodes")
    
    # Randomly select episodes until we get 5 successful downloads
    print(f"\n🎲 Randomly selecting and downloading episodes...")
    random.seed(42)  # For reproducibility
    shuffled_episodes = episodes_with_mappings.copy()
    random.shuffle(shuffled_episodes)
    
    successful_downloads = []
    failed_downloads = []
    attempted_episodes = []
    
    # Keep trying until we get 5 successful downloads or run out of episodes
    for episode_id in shuffled_episodes:
        if len(successful_downloads) >= NUM_EPISODES_TO_DOWNLOAD:
            break
        
        attempted_episodes.append(episode_id)
        print(f"\n[{len(attempted_episodes)}/{len(shuffled_episodes)}] Processing: {episode_id}")
        
        # Get episode path from mapping
        episode_path = episode_mapping.get(episode_id)
        if not episode_path:
            print(f"   ⚠️  No mapping found for episode {episode_id}")
            failed_downloads.append(episode_id)
            continue
        
        # Download left camera video
        success = download_left_camera_video(
            episode_id,
            episode_path,
            OUTPUT_DIR
        )
        
        if success:
            successful_downloads.append(episode_id)
            print(f"   ✅ Success! ({len(successful_downloads)}/{NUM_EPISODES_TO_DOWNLOAD} completed)")
        else:
            failed_downloads.append(episode_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {len(successful_downloads)}/{NUM_EPISODES_TO_DOWNLOAD}")
    print(f"📊 Attempted: {len(attempted_episodes)} episodes")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    
    if successful_downloads:
        print("\n✅ Successful episodes:")
        for episode_id in successful_downloads:
            print(f"   - {episode_id}")
    
    if failed_downloads:
        print("\n❌ Failed episodes:")
        for episode_id in failed_downloads[:10]:  # Show first 10 failed
            print(f"   - {episode_id}")
        if len(failed_downloads) > 10:
            print(f"   ... and {len(failed_downloads) - 10} more")
    
    # Save selected episodes list
    output_file = Path("correct_new_scripts/selected_episodes_for_download.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "attempted": attempted_episodes,
            "successful": successful_downloads,
            "failed": failed_downloads
        }, f, indent=2)
    
    print(f"\n✅ Saved selection to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

