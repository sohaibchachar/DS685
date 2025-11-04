#!/usr/bin/env python3
"""
Test script with focused captioning prompts for video understanding.
Asks the model to:
1. Caption what's happening
2. Identify objects with colors on table
3. Describe robot actions
4. Caption final state
Then generates PDDL domain and problem files.
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_pddl_with_cosmos import (
    load_cosmos_model,
    analyze_sequential_video,
    generate_domain_from_analyses,
    generate_problem_pddl,
    load_annotations
)

# Configuration
ANNOTATIONS_FILE = "droid_language_annotations.json"
OUTPUT_DIR = Path("correct_new_scripts/test_captioning_output")
VIDEO_DIR = Path("correct_new_scripts/test_raw_videos")
FRAME_RATE = 2  # FPS for video processing


def caption_video_with_focused_prompt(
    processor, model, device, video_path: Path, instruction: str
) -> Dict[str, Any]:
    """
    Caption video with focused prompts asking specific questions.
    """
    if not video_path.exists():
        return {"error": f"Video file not found: {video_path}"}
    
    video_size = video_path.stat().st_size
    if video_size < 1000:
        return {"error": f"Video file appears corrupted (size: {video_size} bytes)"}
    
    print(f"   📹 Captioning video with focused prompts...")
    print(f"      Video size: {video_size / (1024*1024):.2f} MB")
    
    # Focused captioning prompt
    prompt_text = f"""You are watching a robot manipulation video. Please provide detailed captions answering these specific questions:

INSTRUCTION: {instruction}

1. INITIAL STATE - What objects are on the table at the start?
   - List ALL objects with their COLORS (e.g., "green block", "blue cup", "yellow block")
   - State where each object is located (e.g., "on table", "in [container]", "on [block]")
   - Focus ONLY on objects ON THE TABLE - ignore other surfaces

2. WHAT IS THE ROBOT DOING?
   - Describe what actions the robot performs
   - Which objects does the robot interact with?
   - What is the sequence of actions?

3. ACTIONS PERFORMED - What actions does the robot perform on what objects?
   - List each action clearly (e.g., "pick up green block", "place yellow block in blue cup")
   - Be specific about which objects are involved in each action

4. FINAL STATE - What objects are where at the end of the video?
   - List ALL objects with their COLORS (same objects from initial state)
   - State where each object ends up (e.g., "on table", "in [container]", "on [block]")
   - Include objects that didn't move - they should still be listed

Provide your response in JSON format:
{{
  "initial_state_caption": "Description of what objects are on the table at the start with their colors and locations",
  "objects_on_table": [
    {{"name": "green_block", "color": "green", "location": "on table"}},
    {{"name": "blue_cup", "color": "blue", "location": "on table"}}
  ],
  "robot_actions_description": "What the robot is doing in the video",
  "actions_performed": [
    "pick up green_block",
    "place green_block in blue_cup"
  ],
  "final_state_caption": "Description of what objects are where at the end with their colors and locations",
  "objects_final_positions": [
    {{"name": "green_block", "color": "green", "location": "in blue_cup"}},
    {{"name": "blue_cup", "color": "blue", "location": "on table"}}
  ]
}}

Be thorough and accurate. Focus on the table surface and objects visible there."""

    try:
        from transformers import Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        
        # Create messages for Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "video",
                        "video": str(video_path.absolute()),
                        "fps": FRAME_RATE,
                    }
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        print(f"   🤖 Running inference...")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"   ✅ Received response")
        print(f"      Response length: {len(response_text)} chars")
        
        # Try to extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                caption_data = json.loads(json_str)
                print(f"   ✅ Successfully parsed JSON caption")
                
                # Convert caption format to analysis format for PDDL generation
                analysis = convert_caption_to_analysis(caption_data)
                return analysis
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parsing error: {e}")
                print(f"   📝 Raw response (first 500 chars): {response_text[:500]}")
                return {"error": f"Failed to parse JSON: {e}", "raw_response": response_text}
        else:
            print(f"   ⚠️  No JSON found in response")
            print(f"   📝 Raw response (first 500 chars): {response_text[:500]}")
            return {"error": "No JSON found in response", "raw_response": response_text}
            
    except Exception as e:
        print(f"   ❌ Error during captioning: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def convert_caption_to_analysis(caption_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert caption format to analysis format expected by PDDL generation.
    """
    # Extract objects from initial state
    initial_objects = caption_data.get("objects_on_table", [])
    if not initial_objects and "initial_state_caption" in caption_data:
        # Try to parse from caption if structured data not available
        initial_objects = []
    
    # Extract objects from final state
    final_objects = caption_data.get("objects_final_positions", [])
    if not final_objects and "final_state_caption" in caption_data:
        # Try to parse from caption if structured data not available
        final_objects = []
    
    # Extract actions
    actions = caption_data.get("actions_performed", [])
    
    # Build analysis structure
    analysis = {
        "initial_state": {
            "objects": initial_objects,
            "robot_holding": None
        },
        "sequence_of_actions": actions,
        "final_state": {
            "objects": final_objects,
            "robot_holding": None
        },
        "object_types": ["block", "container"],
        "predicates": ["on-table", "in", "on", "clear", "holding"]
    }
    
    return analysis


def download_random_video() -> tuple[str, Path]:
    """
    Download one random video from DROID annotations.
    Returns: (episode_id, video_path)
    """
    print("=" * 80)
    print("📥 DOWNLOADING RANDOM VIDEO")
    print("=" * 80)
    
    # Load annotations
    annotations_path = Path(ANNOTATIONS_FILE)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Handle empty file
        if not content:
            raise ValueError(f"Annotations file is empty")
        
        # Handle Git LFS and merge conflicts
        if content.startswith("version https://git-lfs.github.com"):
            lines = content.split('\n')
            json_start = next((i for i, line in enumerate(lines) if line.strip().startswith('{')), None)
            if json_start is not None:
                content = '\n'.join(lines[json_start:])
            else:
                raise ValueError(f"Could not find JSON in Git LFS file")
        
        # Handle merge conflicts - take the content after =======
        if '<<<<<<<' in content and '=======' in content:
            # Skip the HEAD section, take the part after =======
            parts = content.split('=======')
            if len(parts) > 1:
                content = parts[1]
                # Remove the ending marker if present
                if '>>>>>>>' in content:
                    content = content.split('>>>>>>>')[0]
        
        try:
            annotations = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse annotations JSON: {e}")
    
    # Get random episode
    episode_ids = list(annotations.keys())
    random_episode_id = random.choice(episode_ids)
    instruction = annotations[random_episode_id]
    
    print(f"\n🎲 Random episode selected:")
    print(f"   Episode ID: {random_episode_id}")
    print(f"   Instruction: {instruction}")
    
    # Download video using download_specified_videos script logic
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from download_specified_videos import download_specific_video
    
    # Load episode mapping manually
    mapping_file = Path(__file__).parent.parent / "episode_id_to_path.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Episode mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        episode_mapping = json.load(f)
    
    if random_episode_id not in episode_mapping:
        raise ValueError(f"Episode {random_episode_id} not found in mapping")
    
    episode_path = episode_mapping[random_episode_id]
    
    # List available videos in GCS
    gcs_base = "gs://gresearch/robotics/droid_raw"
    version = "1.0.1"
    gcs_mp4_path = f"{gcs_base}/{version}/{episode_path}/recordings/MP4"
    
    print(f"\n📡 Listing videos in GCS...")
    result = subprocess.run(
        ["gsutil", "ls", gcs_mp4_path],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list videos: {result.stderr}")
    
    video_files = [line.strip().split('/')[-1] for line in result.stdout.split('\n') if line.strip().endswith('.mp4')]
    
    if not video_files:
        raise RuntimeError(f"No videos found for episode {random_episode_id}")
    
    # Select first non-stereo video
    selected_video = None
    for vid in video_files:
        if not vid.startswith('stereo'):
            selected_video = vid
            break
    
    if not selected_video:
        selected_video = video_files[0]  # Fallback to first video
    
    print(f"   Selected video: {selected_video}")
    
    # Download the video
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    success = download_specific_video(
        random_episode_id,
        episode_path,
        selected_video,
        VIDEO_DIR
    )
    
    if not success:
        raise RuntimeError(f"Failed to download video {selected_video}")
    
    video_path = VIDEO_DIR / random_episode_id / "recordings" / "MP4" / selected_video
    
    print(f"\n✅ Video downloaded successfully!")
    print(f"   Path: {video_path}")
    
    return random_episode_id, video_path


def main():
    """Main execution."""
    print("=" * 80)
    print("🧪 TEST CAPTIONING AND PDDL GENERATION")
    print("=" * 80)
    
    # Step 1: Download random video
    try:
        episode_id, video_path = download_random_video()
    except Exception as e:
        print(f"\n❌ Error downloading video: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Load annotations
    annotations = load_annotations()
    instruction = annotations.get(episode_id, "Perform the manipulation task")
    
    # Ensure instruction is a string
    if isinstance(instruction, dict):
        # If it's a dict, try to extract text or use default
        instruction = instruction.get("text", instruction.get("instruction", "Perform the manipulation task"))
    elif not isinstance(instruction, str):
        instruction = str(instruction) if instruction else "Perform the manipulation task"
    
    print(f"\n📋 Instruction: {instruction}")
    
    # Step 3: Load model
    print(f"\n🤖 Loading Cosmos-Reason1-7B model...")
    try:
        processor, model, device = load_cosmos_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 4: Caption video
    print(f"\n📝 Step 1: Captioning video...")
    caption_data = caption_video_with_focused_prompt(
        processor, model, device, video_path, instruction
    )
    
    if "error" in caption_data:
        print(f"\n❌ Captioning failed: {caption_data['error']}")
        if "raw_response" in caption_data:
            print(f"\n📄 Raw response:")
            print(caption_data["raw_response"][:1000])
        return
    
    print(f"\n✅ Captioning complete!")
    print(f"   Caption keys: {list(caption_data.keys())}")
    
    # Step 5: Generate PDDL files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    domain_file = OUTPUT_DIR / "domain.pddl"
    problem_dir = OUTPUT_DIR / "problems"
    
    print(f"\n📝 Step 2: Generating domain.pddl...")
    generate_domain_from_analyses(
        all_analyses=[caption_data],
        output_path=domain_file
    )
    
    print(f"\n📝 Step 3: Generating problem.pddl...")
    generate_problem_pddl(
        video_id=episode_id,
        instruction=instruction,
        analysis=caption_data,
        output_dir=problem_dir
    )
    
    print(f"\n" + "=" * 80)
    print(f"✅ TEST COMPLETE!")
    print(f"=" * 80)
    print(f"📁 Domain file: {domain_file}")
    problem_filename = f"problem_{episode_id.replace('+', '_').replace('-', '_')}.pddl"
    print(f"📁 Problem file: {problem_dir / problem_filename}")
    print(f"📁 Video: {video_path}")


if __name__ == "__main__":
    main()

