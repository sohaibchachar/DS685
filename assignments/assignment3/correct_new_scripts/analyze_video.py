#!/usr/bin/env python3
"""
Simple video analyzer: Outputs objects and actions in a concise format.
Analyzes a single video and reports what actually happens - no hallucinations.
"""

# Set multiprocessing start method BEFORE any imports that use CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import json
import re
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# --- Configuration ---
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
FRAME_RATE = 4 # FPS for video analysis

# --- Focused Prompt ---
ANALYSIS_PROMPT = """Analyze this robot manipulation video and report EXACTLY what you observe.

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

# --- Video Analysis ---
def analyze_video(processor, model, device, video_path: Path):
    """Analyzes video and returns structured data."""
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    video_size = video_path.stat().st_size
    if video_size < 1000:
        print(f"❌ Video file appears corrupted (size: {video_size} bytes)")
        return None
        
    print(f"   📹 Analyzing: {video_path.name} ({video_size / (1024*1024):.2f} MB)")
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
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
                max_new_tokens=4096,
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
        
        # Extract JSON
        json_str = response
        if "```json" in json_str:
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str, flags=re.MULTILINE)
        
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                print("   ✅ Successfully parsed analysis")
                return analysis
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parsing failed: {e}")
                print(f"   Raw response preview: {response[:500]}...")
                return None
        
        print(f"   ⚠️  No JSON found in response")
        print(f"   Raw response preview: {response[:500]}...")
        return None
        
    except Exception as e:
        print(f"   ⚠️  Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Format Output ---
def normalize_object_name(obj_name: str, color: str = "") -> str:
    """Normalizes object name to format: color_objecttype"""
    if color:
        # If obj_name already has color prefix, use it as-is
        obj_lower = obj_name.lower()
        color_lower = color.lower()
        if obj_lower.startswith(color_lower):
            return obj_name
        else:
            # Extract object type (block, cup, container, etc.)
            obj_type = obj_name.split('_')[-1] if "_" in obj_name else obj_name
            return f"{color}_{obj_type}"
    else:
        return obj_name

def format_output(analysis):
    """Formats the analysis into a clean, concise output."""
    if not analysis:
        print("❌ No analysis data available")
        return
    
    # Extract objects from initial state
    initial_state = analysis.get("initial_state", [])
    objects = []
    objects_on_table = []
    objects_not_on_table = []
    
    if isinstance(initial_state, list):
        for obj in initial_state:
            obj_name = obj.get("object", obj.get("name", "unknown"))
            location = obj.get("location", "unknown").lower()
            color = obj.get("color", "")
            full_name = normalize_object_name(obj_name, color)
            if full_name not in objects:
                objects.append(full_name)
            
            # Separate objects on table from others
            if "on table" in location or "table" in location:
                objects_on_table.append(full_name)
            else:
                objects_not_on_table.append((full_name, obj.get("location", "unknown")))
    
    # Print objects list
    if objects:
        print("\n" + ", ".join(objects))
    else:
        print("\nNo objects found")
        return
    
    # Print initial state - specifically objects on table
    print("\nObjects on table:")
    if objects_on_table:
        print(", ".join(objects_on_table))
    else:
        print("No objects on table")
    
    # Print other initial locations if any
    if objects_not_on_table:
        print("\nOther initial locations:")
        other_parts = [f"{name} {loc}" for name, loc in objects_not_on_table]
        print(", ".join(other_parts))
    
    # Extract and list all objects that were moved
    print("\nObjects moved:")
    actions = analysis.get("actions", [])
    moved_objects = set()
    action_descriptions = []
    
    if isinstance(actions, list) and len(actions) > 0:
        for action in actions:
            action_str = str(action).strip().lower()
            # Extract object name from "picked up" or "placed" actions
            if "picked up" in action_str or "placed" in action_str:
                # Try to extract object name
                picked_match = re.search(r'picked\s+up\s+([^from]+?)\s+from', action_str)
                placed_match = re.search(r'placed\s+([^in]+?)\s+(in|on)', action_str)
                
                obj_name_raw = None
                if picked_match:
                    obj_name_raw = picked_match.group(1).strip()
                elif placed_match:
                    obj_name_raw = placed_match.group(1).strip()
                
                if obj_name_raw:
                    # Try to normalize object name by matching with known objects
                    # First try exact match
                    obj_normalized = obj_name_raw
                    for known_obj in objects:
                        if known_obj.lower() == obj_name_raw.lower() or known_obj.lower() in obj_name_raw.lower() or obj_name_raw.lower() in known_obj.lower():
                            obj_normalized = known_obj
                            break
                    moved_objects.add(obj_normalized)
                
                # Extract "placed" action description
                if "placed" in action_str:
                    action_str_clean = action_str.replace("robot ", "").replace("robot", "").strip()
                    if action_str_clean.startswith("placed"):
                        action_descriptions.append(action_str_clean)
                    else:
                        placed_match = re.search(r'placed\s+([^in]+?)\s+(in|on)\s+(.+)', action_str_clean)
                        if placed_match:
                            obj = placed_match.group(1).strip()
                            prep = placed_match.group(2).strip()
                            target = placed_match.group(3).strip()
                            # Try to normalize object name in action description
                            obj_normalized = obj
                            for known_obj in objects:
                                if known_obj.lower() == obj.lower() or known_obj.lower() in obj.lower() or obj.lower() in known_obj.lower():
                                    obj_normalized = known_obj
                                    break
                            action_descriptions.append(f"placed {obj_normalized} {prep} {target}")
    
    if moved_objects:
        print(", ".join(sorted(moved_objects)))
    else:
        print("No objects moved")
    
    # Print actions descriptions
    if action_descriptions:
        print("\nActions:")
        print(", ".join(action_descriptions))
    elif moved_objects:
        print("\nActions:")
        print("Objects were moved but action details not available")
    else:
        print("\nActions:")
        print("No actions performed")
    
    # Print final state
    print("\nFinal state:")
    final_state = analysis.get("final_state", [])
    final_state_parts = []
    if isinstance(final_state, list):
        for obj in final_state:
            obj_name = obj.get("object", obj.get("name", "unknown"))
            location = obj.get("location", "unknown")
            color = obj.get("color", "")
            full_name = normalize_object_name(obj_name, color)
            final_state_parts.append(f"{full_name} {location}")
    if final_state_parts:
        print(", ".join(final_state_parts))
    else:
        print("No final state")

# --- Main ---
def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <video_path>")
        print("\nExample:")
        print("  python analyze_video.py correct_new_scripts/raw_videos/AUTOLab+84bd5053+2023-07-08-09h-24m-54s/recordings/MP4/24400334.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("VIDEO ANALYZER")
    print("=" * 80)
    print(f"\n📹 Video: {video_path}")
    
    # Load model
    processor, model, device = load_cosmos_model()
    
    # Analyze video
    print(f"\n🔍 Analyzing video...")
    analysis = analyze_video(processor, model, device, video_path)
    
    # Format and display output
    format_output(analysis)

if __name__ == "__main__":
    main()

