#!/usr/bin/env python3
"""
Stage 1: Extract text descriptions from all videos using Cosmos-Reason1-7B.
This script analyzes videos and saves natural language descriptions of what happens.
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
VIDEO_DIR = "correct_new_scripts/raw_videos"
OUTPUT_DESCRIPTIONS_DIR = "correct_new_scripts/video_descriptions"
FRAME_RATE = 16  # FPS for video analysis
RELOAD_MODEL_EVERY_N_VIDEOS = 1  # Reload model every N videos (1 = reload after each video)

# --- Simplified Prompt for Text Description ---
COSMOS_DESCRIPTION_PROMPT = """Analyze this robot manipulation video and report EXACTLY what you observe.

CRITICAL: Report ONLY what you actually see - do NOT invent or assume actions that don't occur.

1. INITIAL STATE - List all objects present at the start:
   - Format: "color_block" or "color_container" (e.g., "yellow_block", "blue_cup")
   - Location: "on table", "in [container_name]", "on [block_name]"
   - Example: "yellow_block on table", "blue_cup on table"

2. ACTIONS - List ONLY actual object manipulations you observe:
   - Watch the ENTIRE video from start to finish
   - Count ONLY when robot actually picks up, places, or moves objects
   - Format: "robot picked up [object] from [location]" then "robot placed [object] in/on [location]"
   - Be concise: one line per action
   - If multiple objects are manipulated, list ALL of them in order
   - If only one object is manipulated, report ONLY that one - do NOT create fictional additional actions

3. FINAL STATE - Where each object ends up:
   - List final location of each object that was present initially

Provide your response in this exact JSON format:
{
  "initial_state": [
    {"object": "yellow_block", "location": "on table", "color": "yellow"},
    {"object": "blue_cup", "location": "on table", "color": "blue"}
  ],
  "actions": [
    "robot picked up yellow_block from table",
    "robot placed yellow_block in blue_cup"
  ],
  "final_state": [
    {"object": "yellow_block", "location": "in blue_cup", "color": "yellow"},
    {"object": "blue_cup", "location": "on table", "color": "blue"}
  ]
}

IMPORTANT:
- Report EXACTLY what you see - if there's 1 action, report 1; if there are 5, report all 5
- Do NOT count robot movements without object manipulation as actions
- Do NOT invent actions that don't occur
- Be accurate and precise"""
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

# --- Video Description Extraction ---
def extract_video_description(processor, model, device, video_path: Path) -> str:
    """
    Extracts a natural language description from a video using Cosmos.
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
def find_video_files(video_dir: Path) -> List[Path]:
    """Finds all MP4 video files in the video directory structure."""
    video_files = []
    for episode_dir in video_dir.iterdir():
        if episode_dir.is_dir():
            mp4_dir = episode_dir / "recordings" / "MP4"
            if mp4_dir.exists():
                for video_file in mp4_dir.glob("*.mp4"):
                    video_files.append(video_file)
    return sorted(video_files)

def main():
    """Main function to extract descriptions from all videos."""
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
    
    # Find all videos
    video_files = find_video_files(video_dir)
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"\n📁 Found {len(video_files)} video files")
    
    # Load model initially
    processor, model, device = load_cosmos_model()
    
    # Process each video
    all_descriptions = {}
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.parent.parent.parent.name}")
        
        # Reload model if needed (to prevent cross-contamination)
        # Reload before processing if we've processed N videos since last reload
        if RELOAD_MODEL_EVERY_N_VIDEOS > 0 and i > 1 and (i - 1) % RELOAD_MODEL_EVERY_N_VIDEOS == 0:
            print("   🔄 Reloading model to clear context...")
            unload_model(model, processor)
            processor, model, device = load_cosmos_model()
        
        # Extract description
        description = extract_video_description(processor, model, device, video_path)
        
        # Clear cache after each video to prevent cross-contamination
        clear_model_cache(model, device)
        
        # Get episode ID
        episode_id = video_path.parent.parent.parent.name
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
        
        # Store for combined JSON
        all_descriptions[episode_id] = {
            "video_path": str(video_path),
            "description": description
        }
        
        print(f"   💾 Saved: {desc_file.name}")
    
    # Save combined descriptions JSON
    combined_file = output_dir / "all_descriptions.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Completed! Processed {len(video_files)} videos")
    print(f"📁 Individual descriptions: {output_dir}")
    print(f"📁 Combined JSON: {combined_file}")

if __name__ == "__main__":
    main()
