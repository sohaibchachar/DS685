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


COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
VIDEO_DIR = "raw_videos"
OUTPUT_DESCRIPTIONS_DIR = "video_descriptions"
FRAME_RATE = 16  # FPS for video analysis
RELOAD_MODEL_EVERY_N_VIDEOS = 1  # Reload model every N videos (1 = reload after each video)

COSMOS_DESCRIPTION_PROMPT = """Analyze the robot manipulation video. Your response must be 100% accurate to what is visually shown.

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
def load_cosmos_model():
    print("Loading Cosmos-Reason1-7B...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please use a GPU-enabled environment.")
    
    device = "cuda"
    print(f"Using device: {device}")
    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            COSMOS_MODEL_ID,
            trust_remote_code=True
        )
        model = None
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Trying with 4-bit quantization...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                COSMOS_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        except Exception as quant_error:
            print(f"Quantization failed: {quant_error}")
            print("Trying without quantization...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                COSMOS_MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
        model.eval()
        print(f"Successfully loaded: {COSMOS_MODEL_ID}")
        return processor, model, device
    except Exception as e:
        print(f"Failed to load: {e}")
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


def group_videos_by_episode(video_dir: Path) -> Dict[str, List[Path]]:

    episodes = {}
    for episode_dir in video_dir.iterdir():
        if episode_dir.is_dir():
            episode_id = episode_dir.name
            mp4_dir = episode_dir / "recordings" / "MP4"
            if mp4_dir.exists():
                video_files = []
                for video_file in mp4_dir.glob("*.mp4"):
                    video_files.append(video_file)
   
                if video_files:
                    episodes[episode_id] = sorted(video_files)
    return episodes

def extract_video_description(processor, model, device, video_path: Path) -> str:
    video_size = video_path.stat().st_size   
    print(f"Analyzing: {video_path.name} ({video_size / (1024*1024):.2f} MB)")
    
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
                max_new_tokens=8192,
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
        
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            response = answer_match.group(1).strip()
        
        print(f"Extracted description ({len(response)} chars)")
        return response
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

# --- Main Processing ---
def parse_json_from_description(description: str) -> Dict:
    json_match = re.search(r'\{.*\}', description, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return {"raw_description": description}

def main():
    print("Extract Video Descriptions with Cosmos")
    video_dir = Path(VIDEO_DIR)
    output_dir = Path(OUTPUT_DESCRIPTIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = group_videos_by_episode(video_dir)
    
    if not episodes:
        print(f"No episodes with valid video files found in {video_dir}")
        return
    
    print(f"\nFound {len(episodes)} episodes with videos:")
    total_videos = 0
    for episode_id, video_paths in episodes.items():
        print(f"   - {episode_id}: {len(video_paths)} video(s) (will use first available)")
        total_videos += len(video_paths)
    print(f"   Total videos (excluding stereo): {total_videos}")
    processor, model, device = load_cosmos_model()
    all_descriptions = {}
    video_count = 0
    
    for i, (episode_id, video_paths) in enumerate(episodes.items(), 1):
        if not video_paths:
            continue
        video_path = video_paths[0]
        video_count += 1
        
        print(f"\n[{i}/{len(episodes)}] Processing episode: {episode_id}")
        print(f"Analyzing: {video_path.name}")
        
        if RELOAD_MODEL_EVERY_N_VIDEOS > 0 and video_count > 1 and (video_count - 1) % RELOAD_MODEL_EVERY_N_VIDEOS == 0:
            print("Reloading model to clear context...")
            unload_model(model, processor)
            processor, model, device = load_cosmos_model()
        description = extract_video_description(processor, model, device, video_path)
        
        clear_model_cache(model, device)

        parsed_data = parse_json_from_description(description)

        safe_episode_id = episode_id.replace('+', '_').replace('-', '_')
        
        desc_file = output_dir / f"{safe_episode_id}_description.txt"
        with open(desc_file, 'w', encoding='utf-8') as f:
            f.write(f"Episode ID: {episode_id}\n")
            f.write(f"Video Path: {video_path}\n")
            f.write("=" * 80 + "\n")
            f.write("VIDEO DESCRIPTION:\n")
            f.write("=" * 80 + "\n")
            f.write(description)
            f.write("\n")
        
        json_file = output_dir / f"{safe_episode_id}_description.json"
        episode_data = {
            "episode_id": episode_id,
            "video": str(video_path),
            "description": description,
            "parsed_data": parsed_data
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        all_descriptions[episode_id] = episode_data
        
        print(f"Saved: {desc_file.name} and {json_file.name}")
    
    combined_file = output_dir / "all_descriptions.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted! Processed {len(episodes)} episodes ({video_count} videos)")
    print(f"Individual descriptions: {output_dir}")
    print(f"Combined JSON: {combined_file}")

if __name__ == "__main__":
    main()
