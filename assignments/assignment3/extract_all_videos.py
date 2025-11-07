#!/usr/bin/env python3
"""
Batch process all videos in raw_videos folder using extract_single_video.py
Each video is processed separately to prevent hallucination.
"""

import subprocess
import sys
from pathlib import Path

def find_videos(video_dir: Path) -> list[Path]:
    """Find all non-stereo video files."""
    videos = []
    for video_file in video_dir.rglob("*.mp4"):
        # Skip stereo videos
        video_name_lower = video_file.name.lower()
        stereo_keywords = ["stereo", "left", "right", "_l.", "_r.", "_left", "_right"]
        if not any(keyword in video_name_lower for keyword in stereo_keywords):
            videos.append(video_file)
    return sorted(videos)

def main():
    video_dir = Path("raw_videos")
    
    if not video_dir.exists():
        print(f"❌ Error: {video_dir} not found")
        return 1
    
    videos = find_videos(video_dir)
    
    if not videos:
        print(f"❌ No videos found in {video_dir}")
        return 1
    
    print("=" * 80)
    print("BATCH VIDEO PROCESSING - Frame-based Extraction")
    print("=" * 80)
    print(f"\nFound {len(videos)} videos to process:\n")
    
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    
    print("\n" + "=" * 80)
    
    # Process each video
    success_count = 0
    failed_videos = []
    
    for i, video in enumerate(videos, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing [{i}/{len(videos)}]: {video.name}")
        print(f"{'=' * 80}")
        
        # Run extract_single_video.py for this video
        cmd = [
            sys.executable,
            "extract_single_video.py",
            str(video),
            "--frames", "12"  # Use 12 frames for better accuracy
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            if result.returncode == 0:
                success_count += 1
                print(f"✅ Successfully processed video {i}/{len(videos)}")
            else:
                print(f"⚠️  Video {i}/{len(videos)} completed with warnings")
                failed_videos.append(video)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to process video {i}/{len(videos)}: {e}")
            failed_videos.append(video)
        except KeyboardInterrupt:
            print("\n\n⚠️  Processing interrupted by user")
            print(f"Processed {success_count}/{len(videos)} videos successfully")
            return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(videos)} videos")
    
    if failed_videos:
        print(f"\n❌ Failed videos ({len(failed_videos)}):")
        for video in failed_videos:
            print(f"  - {video}")
    
    print(f"\n📁 Descriptions saved to: video_descriptions/")
    
    return 0 if len(failed_videos) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

