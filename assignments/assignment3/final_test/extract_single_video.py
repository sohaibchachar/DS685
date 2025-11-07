#!/usr/bin/env python3
"""
Extract description from a SINGLE video by analyzing the full video.
Modified to append all descriptions to a single video_descriptions.txt file.
Usage: python extract_single_video.py <video_path>
"""

# Set multiprocessing start method BEFORE any imports that use CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import gc
import re
import sys
from pathlib import Path

import cv2
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# --- Configuration ---
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
OUTPUT_FILE = "video_descriptions.txt"  # Single output file for all descriptions

# --- Prompt for Video Description ---
COSMOS_DESCRIPTION_PROMPT = """Analyze this robot manipulation video. Your response must be 100% accurate to what is visually shown.

CRITICAL RULES:
1.  **NO HALLUCINATIONS:** Report **ONLY** what you actually see in the video. Do NOT invent or assume any actions, objects, or locations. If you see one pick-and-place action, report only one action.
2.  **OBJECT NAMING:**
    * **Blocks:** ALL objects being manipulated (cubes, spheres, triangles, cylinder, etc.) MUST be called a "block." Use the format `color_block` (e.g., `yellow_block`, `blue_block`).
    * **Containers:** Any object used to hold blocks (like a bowl, cup, or tray) MUST be called a "container." Use the format `color_container` (e.g., `white_bowl`).
3.  **ACTION FOCUS:**
    * Report **ONLY** the actual manipulation actions you see if robot is picking up or placing a block.
    * Pay close attention to stacking. If a block is placed *on top of* another block, report it as `"on [block_name]"`.
    * If a block is placed *inside* a container, report it as `"in [container_name]"`.
    * Count actions carefully - don't guess.
4.  **If there are multiple blocks, call them by their color only** e.g red_block, blue_block etc.
5.  **IMPORTANT:** Must provide color for each block or container mentioned.
6.  **BE CONSERVATIVE:** If you're not 100% certain an action happened, don't report it.

REQUIRED OUTPUT FORMAT:
Provide your analysis in this exact JSON format.

{
  "initial_state": [
    {"object": "object_name", "location": "location_description"},
    {"object": "object_name", "location": "location_description"}
  ],
  "actions": [
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
    {"object": "white_bowl", "location": "on table"}
  ],
  "actions": [
    "Robot picked up red_block from table",
    "Robot placed red_block in white_bowl"
  ],
  "final_state": [
    {"object": "red_block", "location": "in white_bowl"},
    {"object": "blue_block", "location": "on table"},
    {"object": "white_bowl", "location": "on table"}
  ]
}
"""

# --- Video Info ---
def get_video_info(video_path: Path) -> dict:
    """
    Get video properties.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration
    }

# --- Model Loading ---
def load_cosmos_model():
    """Loads Cosmos-Reason1-7B model using transformers."""
    print("\n🔧 Loading Cosmos-Reason1-7B model...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is required. Please use a GPU-enabled environment.")
    
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
        print(f"   ❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load Cosmos-Reason1-7B: {e}")

def unload_model(model, processor):
    """Completely unloads model from GPU memory."""
    print("\n🧹 Unloading model from memory...")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("   ✅ Model unloaded")

# --- Video-based Description Extraction ---
def extract_description_from_video(processor, model, device, video_path: Path) -> str:
    """
    Extracts a natural language description from a video using Cosmos.
    Returns the text description.
    """
    print(f"\n🎬 Analyzing video with model...")
    
    try:
        # Build message with text prompt and video
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COSMOS_DESCRIPTION_PROMPT},
                    {"type": "video", "video": str(video_path.absolute())}
                ]
            }
        ]
        
        print("   🔄 Processing video with model...")
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=None,
            videos=[str(video_path.absolute())],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("   🤖 Generating description...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.5,  # Lower temperature for more focused output
                top_p=0.9,
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
        print(f"   ❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

def extract_episode_id_from_path(video_path: Path) -> str:
    """
    Extract episode ID from video path.
    Example: raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4
    Returns: RAD+c6cf6b42+2023-08-31-14h-00m-49s
    """
    # Navigate up to the episode directory
    path_parts = video_path.parts
    
    # Find the raw_videos directory and get the episode directory after it
    for i, part in enumerate(path_parts):
        if 'raw_videos' in part.lower():
            if i + 1 < len(path_parts):
                return path_parts[i + 1]
    
    # Fallback: use video filename without extension
    return video_path.stem

def main():
    """Main function to extract description from a single video."""
    parser = argparse.ArgumentParser(
        description="Extract description from a SINGLE video by analyzing the full video (appends to video_descriptions.txt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4
  %(prog)s video.mp4
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SINGLE VIDEO FULL VIDEO EXTRACTION")
    print("=" * 80)
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"\n❌ Error: Video file not found: {video_path}")
        return 1
    
    if not video_path.is_file():
        print(f"\n❌ Error: Path is not a file: {video_path}")
        return 1
    
    video_size = video_path.stat().st_size
    print(f"\n📹 Video: {video_path.name}")
    print(f"   Size: {video_size / (1024*1024):.2f} MB")
    
    # Get video info
    try:
        video_info = get_video_info(video_path)
        print(f"   Video properties: {video_info['total_frames']} frames, {video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s")
    except Exception as e:
        print(f"   ⚠️  Could not read video properties: {e}")
    
    try:
        # Load model FRESH for this single video
        processor, model, device = load_cosmos_model()
        
        # Extract description from video
        description = extract_description_from_video(processor, model, device, video_path)
        
        # Unload model immediately after extraction
        unload_model(model, processor)
        
        # Extract episode ID from video path
        episode_id = extract_episode_id_from_path(video_path)
        
        print(f"\n📋 Episode ID: {episode_id}")
        
        # Prepare output text to append
        output_text = "=" * 80 + "\n"
        output_text += f"Episode ID: {episode_id}\n"
        output_text += f"Video Path: {video_path}\n"
        output_text += "=" * 80 + "\n"
        output_text += description + "\n\n"
        
        # Append to single output file
        output_file = Path(__file__).parent / OUTPUT_FILE
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"\n💾 Description appended to: {output_file}")
        
        print("\n✅ Extraction complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

