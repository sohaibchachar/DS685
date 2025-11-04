#!/usr/bin/env python3
"""
Wrapper script to run generate_pddl_with_cosmos.py on videos in correct_new_scripts/raw_videos
and output PDDL files to correct_new_scripts/problems
"""

import sys
from pathlib import Path

# Add parent directory to path to import the main script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary functions from the main script
from generate_pddl_with_cosmos import (
    load_cosmos_model,
    analyze_sequential_video,
    generate_domain_from_analyses,
    generate_problem_pddl,
    load_annotations
)

# Configuration for this run
VIDEO_DIR = Path("correct_new_scripts/raw_videos")
OUTPUT_DOMAIN_FILE = Path("correct_new_scripts/domain.pddl")
OUTPUT_PROBLEM_DIR = Path("correct_new_scripts/problems")
ANNOTATIONS_FILE = "droid_language_annotations.json"


def main():
    """Main function to generate PDDL files from videos in correct_new_scripts."""
    print("=" * 70)
    print("🚀 PDDL GENERATION WITH COSMOS-REASON1-7B")
    print("   Processing videos from correct_new_scripts/raw_videos")
    print("=" * 70)
    
    # Load model
    try:
        processor, model, device = load_cosmos_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   💡 Make sure transformers, qwen-vl-utils, and bitsandbytes are installed:")
        print(f"      pip install transformers qwen-vl-utils bitsandbytes")
        return
    
    # Load annotations
    print("\n📖 Loading annotations...")
    annotations = load_annotations()
    
    # Find videos
    if not VIDEO_DIR.exists():
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return
    
    video_episodes = [d for d in VIDEO_DIR.iterdir() if d.is_dir()]
    print(f"\n📁 Found {len(video_episodes)} video episodes")
    
    if len(video_episodes) == 0:
        print("❌ No video episodes found!")
        return
    
    # STEP 1: Analyze all videos first to collect domain information
    print("\n" + "=" * 70)
    print("STEP 1: ANALYZING ALL VIDEOS")
    print("=" * 70)
    
    all_analyses = []
    
    for i, episode_dir in enumerate(video_episodes, 1):
        episode_id = episode_dir.name
        print(f"\n[{i}/{len(video_episodes)}] Analyzing: {episode_id}")
        
        # Find video file - check in recordings/MP4 subdirectory first, then other locations
        video_files = list(episode_dir.glob("recordings/MP4/*.mp4"))
        if not video_files:
            video_files = list(episode_dir.glob("recordings/*.mp4"))
        if not video_files:
            video_files = list(episode_dir.glob("*.mp4"))
        # Also check for other video formats
        if not video_files:
            video_files = list(episode_dir.glob("recordings/MP4/*.avi")) + list(episode_dir.glob("recordings/*.avi")) + list(episode_dir.glob("*.avi"))
        if not video_files:
            video_files = list(episode_dir.glob("recordings/MP4/*.mov")) + list(episode_dir.glob("recordings/*.mov")) + list(episode_dir.glob("*.mov"))
        if not video_files:
            print(f"   ⚠️  No video file found in {episode_dir}")
            continue
        
        video_path = video_files[0]
        print(f"   Video: {video_path.name}")
        
        # Get instruction - handle both dict and string formats
        instruction_raw = annotations.get(episode_id, "Perform robot manipulation task")
        if isinstance(instruction_raw, dict):
            # Use the first available instruction
            instruction = instruction_raw.get('language_instruction1') or instruction_raw.get('language_instruction2') or instruction_raw.get('language_instruction3') or "Perform robot manipulation task"
        else:
            instruction = instruction_raw
        print(f"   Instruction: {instruction}")
        
        # Analyze sequential video
        analysis = analyze_sequential_video(processor, model, device, video_path, instruction)
        analysis["episode_id"] = episode_id
        analysis["instruction"] = instruction
        all_analyses.append(analysis)
    
    # STEP 2: Generate domain.pddl from all analyses
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING DOMAIN.PDDL FROM ANALYSES")
    print("=" * 70)
    
    OUTPUT_DOMAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    generate_domain_from_analyses(all_analyses, OUTPUT_DOMAIN_FILE)
    
    # STEP 3: Generate problem files for each video based on the domain
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING PROBLEM FILES")
    print("=" * 70)
    
    OUTPUT_PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, analysis in enumerate(all_analyses, 1):
        if "error" in analysis:
            print(f"\n[{i}/{len(all_analyses)}] ⚠️  Skipping {analysis.get('episode_id', 'unknown')}: analysis failed")
            continue
        
        episode_id = analysis.get("episode_id", "unknown")
        instruction = analysis.get("instruction", "Perform robot manipulation task")
        print(f"\n[{i}/{len(all_analyses)}] Processing: {episode_id}")
        
        # Generate problem file
        generate_problem_pddl(episode_id, instruction, analysis, OUTPUT_PROBLEM_DIR)
    
    successful_count = len([a for a in all_analyses if 'error' not in a])
    print(f"\n✅ Generated {successful_count} problem files in {OUTPUT_PROBLEM_DIR}/")
    print("\n" + "=" * 70)
    print("✅ PDDL GENERATION COMPLETE!")
    print("=" * 70)
    print(f"📁 Domain file: {OUTPUT_DOMAIN_FILE}")
    print(f"📁 Problem files: {OUTPUT_PROBLEM_DIR}/")
    print(f"📊 Processed: {successful_count}/{len(all_analyses)} videos successfully")


if __name__ == "__main__":
    main()

