#!/usr/bin/env python3
"""
Extract description from a SINGLE video by first extracting frames - prevents hallucination.
Usage: python extract_single_video.py <video_path> [--output <output_file>]
"""

# Set multiprocessing start method BEFORE any imports that use CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import argparse
import gc
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# --- Configuration ---
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
DEFAULT_NUM_FRAMES = 8  # Number of frames to extract from video
MAX_FRAMES = 30  # Maximum frames to prevent overload

# --- Prompt for Frame-based Description ---
COSMOS_DESCRIPTION_PROMPT = """Analyze these frames from a robot manipulation video. Your response must be 100% accurate to what is visually shown.

CRITICAL RULES:
1.  **NO HALLUCINATIONS:** Report **ONLY** what you actually see in the frames. Do NOT invent or assume any actions, objects, or locations. If you see one pick-and-place action, report only one action.
2.  **OBJECT NAMING:**
    * **Blocks:** ALL objects being manipulated (cubes, spheres, triangles, cylinder, etc.) MUST be called a "block." Use the format `color_block` (e.g., `yellow_block`, `blue_block`).
    * **Containers:** Any object used to hold blocks (like a bowl, cup, or tray) MUST be called a "container." Use the format `color_container` (e.g., `white_bowl`).
3.  **ACTION FOCUS:**
    * Report **ONLY** the actual manipulation actions you see if robot is picking up or placing a block.
    * Pay close attention to stacking. If a block is placed *on top of* another block, report it as `"on [block_name]"`.
    * If a block is placed *inside* a container, report it as `"in [container_name]"`.
    * Count actions carefully by comparing frames - don't guess.
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

# --- Frame Extraction ---
def extract_frames_from_video(video_path: Path, num_frames: int, output_dir: Path) -> list[Path]:
    """
    Extracts evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        output_dir: Directory to save frames
        
    Returns:
        List of paths to extracted frame images
    """
    print(f"\n🎬 Extracting {num_frames} frames from video...")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Calculate frame indices to extract (evenly spaced)
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        # Evenly space frames across the video
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    print(f"   Extracting frames at indices: {frame_indices}")
    
    # Extract frames
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ⚠️  Failed to read frame {frame_idx}")
            continue
        
        # Save frame as image
        frame_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(frame_path)
    
    cap.release()
    print(f"   ✅ Extracted {len(frame_paths)} frames")
    
    return frame_paths

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

# --- Frame-based Description Extraction ---
def extract_description_from_frames(processor, model, device, frame_paths: list[Path]) -> str:
    """
    Extracts a natural language description from frames using Cosmos.
    Returns the text description.
    """
    print(f"\n📸 Analyzing {len(frame_paths)} frames with model...")
    
    try:
        # Prepare image content for the message
        image_content = []
        for frame_path in frame_paths:
            image_content.append({
                "type": "image",
                "image": str(frame_path.absolute())
            })
        
        # Build message with text prompt followed by images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COSMOS_DESCRIPTION_PROMPT}
                ] + image_content
            }
        ]
        
        print("   🔄 Processing frames with model...")
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Load images for processing
        image_inputs = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            image_inputs.append(img)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("   🤖 Generating description...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,  # Reduced for frame-based analysis
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

def parse_json_from_description(description: str) -> dict:
    """Attempts to parse JSON from the model's description response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', description, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parsing failed: {e}")
            pass
    
    # If JSON parsing fails, return the raw description
    return {"raw_description": description}

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

def make_safe_episode_id(episode_id: str) -> str:
    """
    Convert episode ID to safe format for filenames.
    Example: RAD+c6cf6b42+2023-08-31-14h-00m-49s -> RAD_c6cf6b42_2023_08_31_14h_00m_49s
    """
    return episode_id.replace('+', '_').replace('-', '_')

def main():
    """Main function to extract description from a single video using frames."""
    parser = argparse.ArgumentParser(
        description="Extract description from a SINGLE video by analyzing frames (reduces hallucination)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4
  %(prog)s raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4 --output my_description.txt
  %(prog)s video.mp4 --frames 12 --output-json my_description.json
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for text description (default: auto-save to video_descriptions/)'
    )
    
    parser.add_argument(
        '--output-json', '-j',
        type=str,
        help='Output file path for JSON description (default: auto-save to video_descriptions/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='video_descriptions',
        help='Output directory for descriptions (default: video_descriptions/)'
    )
    
    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f'Number of frames to extract (default: {DEFAULT_NUM_FRAMES}, max: {MAX_FRAMES})'
    )
    
    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='Keep extracted frames after analysis (saved in temp directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SINGLE VIDEO FRAME-BASED EXTRACTION (Anti-Hallucination Mode)")
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
    
    # Validate frame count
    num_frames = min(args.frames, MAX_FRAMES)
    if args.frames > MAX_FRAMES:
        print(f"   ⚠️  Frame count limited to {MAX_FRAMES} (requested: {args.frames})")
    
    # Create temporary directory for frames
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
        print(f"\n📁 Temporary frame directory: {temp_dir}")
        
        # Extract frames from video
        frame_paths = extract_frames_from_video(video_path, num_frames, temp_dir)
        
        if not frame_paths:
            print("\n❌ Error: No frames were extracted from video")
            return 1
        
        # Load model FRESH for this single video
        processor, model, device = load_cosmos_model()
        
        # Extract description from frames
        description = extract_description_from_frames(processor, model, device, frame_paths)
        
        # Unload model immediately after extraction
        unload_model(model, processor)
        
        # Parse JSON from description
        parsed_data = parse_json_from_description(description)
        
        # Extract episode ID from video path
        episode_id = extract_episode_id_from_path(video_path)
        safe_episode_id = make_safe_episode_id(episode_id)
        
        print(f"\n📋 Episode ID: {episode_id}")
        print(f"   Safe ID: {safe_episode_id}")
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output file paths
        if args.output:
            txt_output_file = Path(args.output)
        else:
            txt_output_file = output_dir / f"{safe_episode_id}_description.txt"
        
        if args.output_json:
            json_output_file = Path(args.output_json)
        else:
            json_output_file = output_dir / f"{safe_episode_id}_description.json"
        
        # Prepare text output
        output_text = f"Episode ID: {episode_id}\n"
        output_text += f"Video Path: {video_path}\n"
        output_text += "=" * 80 + "\n"
        output_text += "VIDEO DESCRIPTION:\n"
        output_text += "=" * 80 + "\n"
        output_text += description + "\n"
        
        # Save text description
        txt_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n💾 Text description saved to: {txt_output_file}")
        
        # Prepare and save JSON output
        json_output = {
            "episode_id": episode_id,
            "video": str(video_path),
            "description": description,
            "parsed_data": parsed_data
        }
        json_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON description saved to: {json_output_file}")
        
        # Handle frame cleanup
        if args.keep_frames:
            print(f"\n📸 Frames kept at: {temp_dir}")
            temp_dir = None  # Prevent cleanup
        
        print("\n✅ Extraction complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary directory if not keeping frames
        if temp_dir and temp_dir.exists() and not args.keep_frames:
            print(f"\n🧹 Cleaning up temporary frames...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())
