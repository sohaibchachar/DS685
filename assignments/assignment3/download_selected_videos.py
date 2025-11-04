#!/usr/bin/env python3
"""
Download selected videos using DROID dataset API.
"""

import json
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm


def download_video_from_url(episode_id: str, output_dir: Path) -> bool:
    """Download video using direct URL construction."""
    # DROID dataset structure: https://droid-dataset.github.io/
    # Videos are hosted on Google Cloud Storage
    
    episode_dir = output_dir / episode_id
    video_output_dir = episode_dir / "recordings" / "MP4"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_videos = list(video_output_dir.glob("*.mp4"))
    if existing_videos:
        print(f"   ✅ Video already exists: {existing_videos[0].name}")
        return True
    
    # Try to construct the URL
    # Format: https://storage.googleapis.com/gresearch/droid/1.0.1/data/{episode_id}/recordings/MP4/{video_id}.mp4
    base_url = "https://storage.googleapis.com/gresearch/droid/1.0.1/data"
    
    # We need to find the video ID - typically it's a numeric ID
    # Let's try to get the metadata first
    metadata_url = f"{base_url}/{episode_id}/metadata.json"
    
    try:
        print(f"   📡 Fetching metadata...")
        response = requests.get(metadata_url, timeout=30)
        
        if response.status_code == 200:
            metadata = response.json()
            # Extract video ID from metadata if available
            video_id = metadata.get('video_id', None)
            
            if video_id:
                video_url = f"{base_url}/{episode_id}/recordings/MP4/{video_id}.mp4"
                print(f"   📥 Downloading from {video_url}...")
                
                video_response = requests.get(video_url, stream=True, timeout=120)
                if video_response.status_code == 200:
                    total_size = int(video_response.headers.get('content-length', 0))
                    video_path = video_output_dir / f"{video_id}.mp4"
                    
                    with open(video_path, 'wb') as f:
                        if total_size > 0:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc='   ⬇️ ') as pbar:
                                for chunk in video_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                        else:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    print(f"   ✅ Downloaded: {video_path.name}")
                    return True
        
        print(f"   ⚠️  Could not fetch metadata (status {response.status_code})")
        return False
        
    except Exception as e:
        print(f"   ❌ Download error: {e}")
        return False


def use_python_droid_download(episode_id: str, output_dir: Path) -> bool:
    """Try using Python droid module if available."""
    try:
        # Try importing droid
        import sys
        sys.path.insert(0, '/workspaces/eng-ai-agents/.venv/lib/python3.12/site-packages')
        
        # Check if droid module exists
        import importlib.util
        spec = importlib.util.find_spec('droid')
        
        if spec is None:
            return False
        
        print(f"   🐍 Using Python droid module...")
        # Import and use the droid module
        from droid import download_episode
        
        download_episode(episode_id, str(output_dir), modalities=['video'])
        
        # Check if downloaded
        episode_dir = output_dir / episode_id
        video_files = list(episode_dir.glob("recordings/MP4/*.mp4"))
        if video_files:
            print(f"   ✅ Downloaded: {video_files[0].name}")
            return True
        
        return False
        
    except Exception as e:
        print(f"   ⚠️  Python module method failed: {e}")
        return False


def main():
    """Main execution."""
    base_dir = Path("/workspaces/eng-ai-agents/assignments/assignment3")
    
    # Load selected episodes
    with open(base_dir / "selected_episodes.json", 'r') as f:
        selected_episodes = json.load(f)
    
    print("=" * 80)
    print("📥 DOWNLOADING SELECTED VIDEOS")
    print("=" * 80)
    print(f"\n🎯 Episodes to download:\n")
    for i, ep in enumerate(selected_episodes, 1):
        print(f"   {i}. {ep}")
    print()
    
    raw_videos_dir = base_dir / "raw_videos"
    raw_videos_dir.mkdir(exist_ok=True)
    
    downloaded_count = 0
    for i, episode_id in enumerate(selected_episodes, 1):
        print(f"\n[{i}/{len(selected_episodes)}] {episode_id}")
        print("-" * 70)
        
        # Try Python module first
        if use_python_droid_download(episode_id, raw_videos_dir):
            downloaded_count += 1
            continue
        
        # Try direct URL download
        if download_video_from_url(episode_id, raw_videos_dir):
            downloaded_count += 1
            continue
        
        print(f"   ❌ All download methods failed")
    
    print("\n" + "=" * 80)
    print(f"✅ Successfully downloaded/verified: {downloaded_count}/{len(selected_episodes)} videos")
    print("=" * 80)
    
    if downloaded_count > 0:
        print("\n💡 Next step: Run the multimodal PDDL generator on these videos")
        print("   Command: python download_and_process_selected.py")


if __name__ == "__main__":
    main()


