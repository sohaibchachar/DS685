#!/usr/bin/env python3
"""
Process all videos in raw_videos folder by calling extract_single_video.py for each one.
Usage: python process_all_videos.py
"""

import subprocess
import sys
from pathlib import Path

# Configuration
RAW_VIDEOS_DIR = "../raw_videos"  # Relative to this script
EXTRACT_SCRIPT = "extract_single_video.py"
OUTPUT_FILE = "video_descriptions.txt"

def find_all_mp4_videos(raw_videos_dir: Path) -> list[Path]:
    """Find all MP4 videos in the raw_videos directory structure."""
    videos = []
    
    # Pattern: raw_videos/*/recordings/MP4/*.mp4
    for episode_dir in raw_videos_dir.iterdir():
        if not episode_dir.is_dir():
            continue
        
        mp4_dir = episode_dir / "recordings" / "MP4"
        if not mp4_dir.exists():
            continue
        
        # Find all MP4 files in this directory
        for video_file in mp4_dir.glob("*.mp4"):
            videos.append(video_file)
    
    return sorted(videos)

def main():
    """Main function to process all videos."""
    print("=" * 80)
    print("PROCESS ALL VIDEOS - Batch Extraction")
    print("=" * 80)
    
    # Get script directory
    script_dir = Path(__file__).parent
    raw_videos_dir = (script_dir / RAW_VIDEOS_DIR).resolve()
    extract_script = script_dir / EXTRACT_SCRIPT
    output_file = script_dir / OUTPUT_FILE
    
    # Validate paths
    if not raw_videos_dir.exists():
        print(f"❌ Error: raw_videos directory not found: {raw_videos_dir}")
        return 1
    
    if not extract_script.exists():
        print(f"❌ Error: Extract script not found: {extract_script}")
        return 1
    
    # Clear the output file if it exists (start fresh)
    if output_file.exists():
        print(f"\n🗑️  Clearing existing output file: {output_file}")
        output_file.unlink()
    
    # Find all videos
    print(f"\n🔍 Searching for videos in: {raw_videos_dir}")
    videos = find_all_mp4_videos(raw_videos_dir)
    
    if not videos:
        print("❌ No videos found!")
        return 1
    
    print(f"✅ Found {len(videos)} videos to process")
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(videos, 1):
        print("\n" + "=" * 80)
        print(f"Processing video {i}/{len(videos)}")
        print("=" * 80)
        print(f"Video: {video_path}")
        
        try:
            # Call the extract script for this video
            result = subprocess.run(
                [sys.executable, str(extract_script), str(video_path)],
                cwd=script_dir,
                capture_output=False,  # Show output in real-time
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully processed video {i}/{len(videos)}")
                successful += 1
            else:
                print(f"❌ Failed to process video {i}/{len(videos)} (exit code: {result.returncode})")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            failed += 1
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output file: {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

