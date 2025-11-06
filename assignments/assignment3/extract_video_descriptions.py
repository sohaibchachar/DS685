#!/usr/bin/env python3
"""
Stage 1: Extract text descriptions from all videos using Cosmos-Reason1-7B.
This script analyzes individual videos and saves natural language descriptions.
"""

# Set multiprocessing start method BEFORE any imports that use CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import json
import re
import gc
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# --- Configuration ---
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
VIDEO_DIR = "raw_videos"
OUTPUT_DESCRIPTIONS_DIR = "video_descriptions"
FRAME_RATE = 8  # FPS for video analysis
RELOAD_MODEL_EVERY_N_VIDEOS = 1  # Reload model every N videos (1 = reload after each video)

# --- Prompt for Video Description ---
COSMOS_DESCRIPTION_PROMPT = """Analyze the robot manipulation video. Your response must be 100% accurate to what is visually shown.

CRITICAL RULES:
1.  **NO HALLUCINATIONS:** Report **ONLY** what you actually see. Do NOT invent or assume any actions, objects, or locations that are not in the video. If the robot moves one block, report one action.
2.  **OBJECT NAMING:**
    * **Blocks:** ALL objects being manipulated (cubes, spheres, triangles, cylinder, etc.) MUST be called a "block." Use the format `color_block` (e.g., `yellow_block`, `blue_block`).
    * **Containers:** Any object used to hold blocks (like a bowl, cup, or tray) MUST be called a "container." Use the format `color_container` (e.g., `red_bowl`).
3.  **ACTION FOCUS:**
    * Report **ONLY** the actual manipulation of objects by robot in video (picking and placing).
    * Pay close attention to stacking. If a block is placed *on top of* another block, report it as `"on [block_name]"`.
    * If a block is placed *inside* a container, report it as `"in [container_name]"`.

4. **If there are multiple blocks with multiple shapes dont call them by their shape just call them by their color only e.g red_block, blue_block etc.
5. **IMPORTANT:** Must provide color for each block or container mentioned.
REQUIRED OUTPUT FORMAT:
Provide your analysis in this exact JSON format.

{
  "initial_state": [
    {"object": "object_name", "location": "location_description"},
    {"object": "object_name", "location": "location_description"}
  ],
  "actions": [
    "Robot picked up [object_name] from [starting_location]",
    "Robot placed [object_name] in/on [final_location]",
    "Robot picked up [object_name] from [starting_location]",
    "Robot placed [object_name] in/on [final_location]"
  ],
  "final_state": [
    {"object": "object_name", "location": "location_description"},
    {"object": "object_name", "location": "location_description"}
  ]
}

Example:
{
  "initial_state": [
    {"object": "red_block", "location": "on table"},
    {"object": "blue_block", "location": "on table"},
    {"object": "green_bowl", "location": "on table"}
  ],
  "actions": [
    "Robot picked up red_block from table",
    "Robot placed red_block in green_bowl",
    "Robot picked up blue_block from table",
    "Robot placed blue_block on red_block"
  ],
  "final_state": [
    {"object": "red_block", "location": "in green_bowl"},
    {"object": "blue_block", "location": "on red_block"},
    {"object": "green_bowl", "location": "on table"}
  ]
}
"""
# --- Model Loading ---
def load_cosmos_model():
    """Loads Cosmos-Reason1-7B model using transformers."""
    print("🔧 Loading Cosmos-Reason1-7B...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please use a GPU-enabled environment.")
    
    device = "cuda"
    print(f"   Using device: {device}")
    
    try:
        print("   🚀 Loading processor...")
        processor = AutoProcessor.from_pretrained(
            COSMOS_MODEL_ID,
            trust_remote_code=True
        )
        
        model = None
        # Try with 4-bit quantization first
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("   Trying with 4-bit quantization...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                COSMOS_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        except Exception as quant_error:
            print(f"   ⚠️  Quantization failed: {quant_error}")
            print("   Trying without quantization...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                COSMOS_MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
        model.eval()
        print(f"   ✅ Successfully loaded: {COSMOS_MODEL_ID}")
        return processor, model, device
    except Exception as e:
        print(f"   ⚠️  Failed to load: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load Cosmos-Reason1-7B: {e}")

def clear_model_cache(model, device):
    """Clears GPU cache and resets model state to prevent cross-contamination."""
    if device == "cuda":
        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
    
    # Reset model to eval mode (in case it was changed)
    model.eval()
    
    # Clear any generation cache if the model has one
    if hasattr(model, 'reset_cache'):
        model.reset_cache()
    
    # Clear any attention cache
    if hasattr(model, 'clear_cache'):
        model.clear_cache()

def unload_model(model, processor):
    """Unloads model from GPU memory."""
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# --- Helper Functions ---
def is_stereo_video(video_path: Path) -> bool:
    """Check if a video is a stereo video (should be skipped)."""
    video_name_lower = video_path.name.lower()
    # Check for common stereo video patterns
    stereo_keywords = ["stereo", "left", "right", "_l.", "_r.", "_left", "_right"]
    return any(keyword in video_name_lower for keyword in stereo_keywords)

def group_videos_by_episode(video_dir: Path) -> Dict[str, List[Path]]:
    """Groups video files by episode ID, excluding stereo videos."""
    episodes = {}
    for episode_dir in video_dir.iterdir():
        if episode_dir.is_dir():
            episode_id = episode_dir.name
            mp4_dir = episode_dir / "recordings" / "MP4"
            if mp4_dir.exists():
                video_files = []
                for video_file in mp4_dir.glob("*.mp4"):
                    # Skip stereo videos
                    if not is_stereo_video(video_file):
                        video_files.append(video_file)
                    else:
                        print(f"   ⏭️  Skipping stereo video: {video_file.name}")
                if video_files:
                    episodes[episode_id] = sorted(video_files)
    return episodes

# --- Video Description Extraction ---
def extract_video_description(processor, model, device, video_path: Path) -> str:
    """
    Extracts a natural language description from a single video using Cosmos.
    Returns the text description.
    """
    if not video_path.exists():
        return f"ERROR: Video file not found: {video_path}"
    
    video_size = video_path.stat().st_size
    if video_size < 1000:
        return f"ERROR: Video file appears corrupted (size: {video_size} bytes)."
        
    print(f"   📹 Analyzing: {video_path.name} ({video_size / (1024*1024):.2f} MB)")
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COSMOS_DESCRIPTION_PROMPT},
                    {
                        "type": "video",
                        "video": str(video_path.absolute()),
                        "fps": FRAME_RATE,
                    }
                ]
            },
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=8192,  # Increased for comprehensive multi-action descriptions
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )
        
        input_ids = inputs.get('input_ids')
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract content from <answer> tag if present
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            response = answer_match.group(1).strip()
        
        print(f"   ✅ Extracted description ({len(response)} chars)")
        return response
        
    except Exception as e:
        print(f"   ⚠️  Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

# --- Main Processing ---
def parse_json_from_description(description: str) -> Dict:
    """Attempts to parse JSON from the model's description response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', description, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing fails, return the raw description
    return {"raw_description": description}

def main():
    """Main function to extract descriptions from videos (one video per episode)."""
    print("=" * 80)
    print("STAGE 1: Extract Video Descriptions with Cosmos")
    print("=" * 80)
    
    # Setup directories
    video_dir = Path(VIDEO_DIR)
    output_dir = Path(OUTPUT_DESCRIPTIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    # Group videos by episode (excluding stereo videos)
    print("\n📁 Grouping videos by episode (excluding stereo videos)...")
    episodes = group_videos_by_episode(video_dir)
    
    if not episodes:
        print(f"❌ No episodes with valid video files found in {video_dir}")
        return
    
    print(f"\n📊 Found {len(episodes)} episodes with videos:")
    total_videos = 0
    for episode_id, video_paths in episodes.items():
        print(f"   - {episode_id}: {len(video_paths)} video(s) (will use first available)")
        total_videos += len(video_paths)
    print(f"   Total videos (excluding stereo): {total_videos}")
    
    # Load model initially
    processor, model, device = load_cosmos_model()
    
    # Process each episode (using first available video)
    all_descriptions = {}
    video_count = 0
    
    for i, (episode_id, video_paths) in enumerate(episodes.items(), 1):
        if not video_paths:
            continue
        
        # Use the first available video
        video_path = video_paths[0]
        video_count += 1
        
        print(f"\n[{i}/{len(episodes)}] Processing episode: {episode_id}")
        print(f"   📹 Analyzing: {video_path.name}")
        
        # Reload model if needed (to prevent cross-contamination)
        if RELOAD_MODEL_EVERY_N_VIDEOS > 0 and video_count > 1 and (video_count - 1) % RELOAD_MODEL_EVERY_N_VIDEOS == 0:
            print("   🔄 Reloading model to clear context...")
            unload_model(model, processor)
            processor, model, device = load_cosmos_model()
        
        # Extract description from video
        description = extract_video_description(processor, model, device, video_path)
        
        # Clear cache after each video to prevent cross-contamination
        clear_model_cache(model, device)
        
        # Try to parse JSON from description
        parsed_data = parse_json_from_description(description)
        
        # Create safe episode ID for filename
        safe_episode_id = episode_id.replace('+', '_').replace('-', '_')
        
        # Save individual description file
        desc_file = output_dir / f"{safe_episode_id}_description.txt"
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(f"Episode ID: {episode_id}\n")
            f.write(f"Video Path: {video_path}\n")
            f.write("=" * 80 + "\n")
            f.write("VIDEO DESCRIPTION:\n")
            f.write("=" * 80 + "\n")
            f.write(description)
            f.write("\n")
        
        # Save JSON file for this episode
        json_file = output_dir / f"{safe_episode_id}_description.json"
        episode_data = {
            "episode_id": episode_id,
            "video": str(video_path),
            "description": description,
            "parsed_data": parsed_data
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        # Store for combined JSON
        all_descriptions[episode_id] = episode_data
        
        print(f"   💾 Saved: {desc_file.name} and {json_file.name}")
    
    # Save combined descriptions JSON
    combined_file = output_dir / "all_descriptions.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completed! Processed {len(episodes)} episodes ({video_count} videos)")
    print(f"📁 Individual descriptions: {output_dir}")
    print(f"📁 Combined JSON: {combined_file}")

if __name__ == "__main__":
    main()
