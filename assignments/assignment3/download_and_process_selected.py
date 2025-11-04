#!/usr/bin/env python3
"""
Download selected videos and generate PDDL files using multimodal approach.
"""

import json
import subprocess
from pathlib import Path
import sys

# Add parent to path to import the multimodal generator
sys.path.insert(0, str(Path(__file__).parent))
from generate_pddl_multimodal import MultimodalPDDLGenerator


def download_video(episode_id: str, output_dir: Path) -> bool:
    """Download a single video using droid CLI."""
    print(f"   📥 Downloading {episode_id}...")
    
    episode_dir = output_dir / episode_id
    if episode_dir.exists():
        # Check if video already exists
        video_files = list(episode_dir.glob("recordings/MP4/*.mp4"))
        if video_files:
            print(f"   ✅ Video already exists: {video_files[0].name}")
            return True
    
    try:
        # Download using droid CLI
        cmd = [
            'droid-download',
            '--episodes', episode_id,
            '--output-dir', str(output_dir),
            '--modalities', 'video'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per video
        )
        
        if result.returncode == 0:
            print(f"   ✅ Downloaded successfully")
            return True
        else:
            print(f"   ❌ Download failed: {result.stderr[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Download timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main execution."""
    base_dir = Path("/workspaces/eng-ai-agents/assignments/assignment3")
    
    # Load selected episodes
    with open(base_dir / "selected_episodes.json", 'r') as f:
        selected_episodes = json.load(f)
    
    print("=" * 80)
    print("🚀 DOWNLOAD AND PROCESS SELECTED VIDEOS")
    print("=" * 80)
    print(f"\n📋 Processing {len(selected_episodes)} episodes:\n")
    for i, ep in enumerate(selected_episodes, 1):
        print(f"   {i}. {ep}")
    print()
    
    # Download videos
    print("\n" + "=" * 80)
    print("📥 DOWNLOADING VIDEOS")
    print("=" * 80)
    
    raw_videos_dir = base_dir / "raw_videos"
    raw_videos_dir.mkdir(exist_ok=True)
    
    downloaded_count = 0
    for i, episode_id in enumerate(selected_episodes, 1):
        print(f"\n[{i}/{len(selected_episodes)}] {episode_id}")
        if download_video(episode_id, raw_videos_dir):
            downloaded_count += 1
    
    print(f"\n✅ Downloaded/verified {downloaded_count}/{len(selected_episodes)} videos")
    
    # Generate PDDL files using multimodal approach
    print("\n" + "=" * 80)
    print("🤖 GENERATING PDDL FILES (MULTIMODAL)")
    print("=" * 80)
    
    generator = MultimodalPDDLGenerator(base_dir)
    
    # Generate domain file (same for all)
    print("\n📝 Generating domain.pddl...")
    domain_content = generator.generate_domain_pddl()
    domain_path = base_dir / "domain.pddl"
    domain_path.write_text(domain_content)
    print(f"   ✅ Domain file saved")
    
    # Process each selected video
    print(f"\n📹 Processing {len(selected_episodes)} selected videos...\n")
    
    valid_count = 0
    for idx, episode_id in enumerate(selected_episodes, 1):
        print(f"[{idx}/{len(selected_episodes)}] {episode_id}")
        print("-" * 70)
        
        # Find video directory
        video_dir = raw_videos_dir / episode_id
        if not video_dir.exists():
            print(f"   ⚠️  Directory not found, skipping")
            continue
        
        # Find video file
        video_files = list(video_dir.glob("recordings/MP4/*.mp4"))
        if not video_files:
            print(f"   ⚠️  No video found, skipping")
            continue
        
        video_path = video_files[0]
        print(f"   📹 Video: {video_path.name}")
        
        # Multimodal analysis
        analysis = generator.analyze_video_multimodal(episode_id, video_path)
        
        if not analysis:
            print(f"   ⚠️  Analysis failed, skipping")
            continue
        
        # Infer PDDL structure
        pddl_config = generator.infer_pddl_from_multimodal(analysis)
        
        # Generate problem file
        problem_content = generator.generate_problem_pddl(episode_id, pddl_config)
        
        # Save problem file
        problem_filename = f"problem_{episode_id.split('+')[0]}_{episode_id.split('+')[1]}_{episode_id.split('+')[2].replace('-', '_')}.pddl"
        problem_path = base_dir / problem_filename
        problem_path.write_text(problem_content)
        
        print(f"   ✅ Problem file saved: {problem_path.name}")
        print(f"   📊 Task: {pddl_config['task_type']}")
        print(f"   🎯 Goals: {pddl_config['goal_predicates']}")
        valid_count += 1
        print()
    
    print("=" * 80)
    print(f"✨ Generated {valid_count}/{len(selected_episodes)} PDDL problems!")
    print("=" * 80)
    
    # Validate
    print("\n🔍 Validating generated PDDL files...")
    try:
        from unified_planning.io import PDDLReader
        reader = PDDLReader()
        
        validated = 0
        for episode_id in selected_episodes:
            problem_filename = f"problem_{episode_id.split('+')[0]}_{episode_id.split('+')[1]}_{episode_id.split('+')[2].replace('-', '_')}.pddl"
            problem_path = base_dir / problem_filename
            
            if problem_path.exists():
                try:
                    problem = reader.parse_problem(str(base_dir / "domain.pddl"), str(problem_path))
                    print(f"   ✅ {problem_filename}")
                    validated += 1
                except Exception as e:
                    print(f"   ❌ {problem_filename}: {str(e)[:60]}...")
        
        print(f"\n✅ {validated}/{valid_count} problems validated!")
        
    except ImportError:
        print("   ⚠️  unified-planning not available, skipping validation")


if __name__ == "__main__":
    main()


