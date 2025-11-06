#!/usr/bin/env python3
"""
PDDL Domain and Problem Generation using Cosmos-Reason1-7B
Simplified and refactored for clarity and maintainability.
Uses transformers instead of vLLM for WSL2 compatibility.
"""

# Set multiprocessing start method BEFORE any imports that use CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass # Already set

import json
import os
import re
import subprocess
import gc
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# --- Configuration ---
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
ANNOTATIONS_FILE = "droid_language_annotations.json"
VIDEO_DIR = "correct_new_scripts/raw_videos"  # Updated for correct_new_scripts folder
OUTPUT_DOMAIN_FILE = "correct_new_scripts/domain.pddl"
OUTPUT_PROBLEM_DIR = "correct_new_scripts/problems"
OUTPUT_RAW_RESPONSES_DIR = "correct_new_scripts/raw_responses"  # Save raw VLM responses
OUTPUT_ANALYSIS_DIR = "correct_new_scripts/analysis_results"    # Save parsed JSON analysis
SAMPLE_SIZE = 15  # Number of videos to process
FRAME_RATE = 4  # FPS for video analysis

# --- VLN Model Prompt ---
COSMOS_SEQUENTIAL_PROMPT = """You are analyzing a robot manipulation video. Your task is to understand the complete state of objects and what changes.

CRITICAL REQUIREMENTS:
1. ACCURATE COLOR IDENTIFICATION: Pay careful attention to the ACTUAL colors of objects. Look closely at the video frames to identify colors correctly (green, yellow, blue, red, orange, black, white, etc.)
2. LIST ALL OBJECTS: You MUST include EVERY object that appears in the video, but ONLY blocks and containers (cups, bowls, etc.)
3. IGNORE OTHER OBJECTS: Do NOT include any objects that are not blocks or containers (ignore tools, surfaces, robots, etc.)
4. CONSISTENT NAMING: Use consistent names for the same object across initial_state and final_state
5. COMPLETE STATES: Both initial_state and final_state MUST list ALL blocks and containers, not just the ones being manipulated
6. COLOR INFORMATION: Always include color for every object - be PRECISE about colors (e.g., "green", "yellow", "blue", "black", "white", "red", "orange")
7. OBJECT TYPES: Only include blocks (cubes, cylinders, rectangular blocks, etc.) and containers (cups, bowls, etc.)

1. INITIAL STATE (start of video):
   - Carefully observe and list ALL blocks and containers present
   - Pay close attention to the ACTUAL colors - look at the video frames carefully
   - Describe where each object is located using EXACT location format:
     * "on table" for objects on the table
     * "in [container_name]" for blocks inside containers (e.g., "in white_bowl")
     * "on [block_name]" for blocks stacked on other blocks
   - Include ALL blocks and containers visible, even if they won't be manipulated
   - DO NOT include any other objects (robots, tools, surfaces, etc.)
   - Be VERY careful about color identification - rewatch if needed

2. SEQUENCE OF ACTIONS (what happens in the video):
   - CRITICAL: Watch the ENTIRE video from start to finish - do NOT stop analyzing after one action
   - Report EXACTLY what you observe - list ALL actual object manipulations that occur
   - Count ONLY actual object manipulations:
     * "pick up [object]" - when robot grasps an object
     * "place [object] on table" - when object is placed on table
     * "stack [object] on [block]" - when object is stacked on another block
     * "place [object] in [container]" - when object is placed inside a container
     * "remove [object] from [container]" - when object is taken out of container
   - Do NOT count robot arm movements, positioning, or retractions as separate actions - these are just movements, not object manipulations
   - If only ONE object is manipulated (e.g., pick up green_block, place green_block in blue_cup), report ONLY those actions - do NOT invent additional actions
   - If MULTIPLE objects are manipulated, list ALL of them in chronological order
   - Each pick-up/place pair typically counts as 2 actions (pick up + place), but report what actually happens
   - Example for single object: ["pick up green_block", "place green_block in blue_cup"]
   - Example for multiple objects: ["pick up yellow_block", "place yellow_block on table", "pick up red_block", "stack red_block on yellow_block"]
   - Focus only on actions involving blocks and containers
   - Mention the ACTUAL colors of objects being manipulated
   - Be accurate: report what you actually see, not what you think should happen

3. FINAL STATE (end of video):
   - Carefully observe and list where EACH block and container ends up
   - Verify colors match the initial state (same objects, same colors)
   - Use EXACT location format same as initial_state
   - Include blocks and containers that didn't move
   - DO NOT include any other objects
   - Be VERY careful about color identification - ensure colors are accurate

Provide your analysis in JSON format with this exact structure:
{{
  "initial_state": {{
    "objects": [
      {{"name": "green_block", "location": "on table", "color": "green"}},
      {{"name": "blue_cup", "location": "on table", "color": "blue"}},
      {{"name": "yellow_block", "location": "on table", "color": "yellow"}}
    ],
    "robot_holding": null
  }},
  "sequence_of_actions": [
    "pick up green_block",
    "place green_block in blue_cup"
  ],
  "final_state": {{
    "objects": [
      {{"name": "green_block", "location": "in blue_cup", "color": "green"}},
      {{"name": "blue_cup", "location": "on table", "color": "blue"}},
      {{"name": "yellow_block", "location": "on table", "color": "yellow"}}
    ],
    "robot_holding": null
  }}
}}

IMPORTANT:
- ONLY include blocks and containers in your analysis
- Ignore all other objects (robots, tools, surfaces, etc.)
- Be EXTREMELY careful about color identification - watch the video carefully
- Verify colors are correct before listing objects
- Every block/container in initial_state MUST appear in final_state with the SAME color
- Use consistent object names (same name = same object)
- Include color for EVERY object
- Do NOT skip objects that don't move - they still need to be in both states
- Be precise about locations: "on table" vs "in [container]" vs "on [block]"
- CRITICAL: Watch the COMPLETE video and report EXACTLY what you observe - include ALL actions that actually occur, whether it's 1 action or multiple
- Only count actual object manipulations (pick up, place, stack, put in container) - do NOT count robot movements or arm positioning as separate actions
- If only one object is manipulated, report only that one action - do NOT create fictional additional actions
- If multiple objects are manipulated, list ALL of them in chronological order
- Report accurately: if there's 1 action, the array should have 1-2 items (pick up + place); if there are 5 actions, the array should reflect all 5
"""

# --- Model Loading ---
def load_cosmos_model():
    """
    Loads Cosmos-Reason1-7B model using transformers.
    Tries 4-bit quantization and falls back if necessary.
    """
    print("🔧 Loading Cosmos-Reason1-7B (sequential video understanding)...")
    print(f"   Model: {COSMOS_MODEL_ID}")
    
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
            print("   Trying without quantization (requires more VRAM)...")
            # Fallback: load without quantization
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


# --- Save VLM Responses ---
def save_vlm_response(video_path: Path, raw_response: str, parsed_analysis: Dict):
    """
    Saves the raw VLM response and parsed JSON analysis to files for review.
    """
    # Create output directories
    raw_responses_dir = Path(OUTPUT_RAW_RESPONSES_DIR)
    analysis_dir = Path(OUTPUT_ANALYSIS_DIR)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract video/episode ID from path
    episode_id = video_path.parent.parent.parent.name  # e.g., "AUTOLab+84bd5053+2023-08-18-12h-25m-58s"
    safe_episode_id = episode_id.replace('+', '_').replace('-', '_')
    
    # Save raw response (text file)
    raw_response_file = raw_responses_dir / f"{safe_episode_id}_raw_response.txt"
    with open(raw_response_file, 'w', encoding='utf-8') as f:
        f.write(f"Episode ID: {episode_id}\n")
        f.write(f"Video Path: {video_path}\n")
        f.write("=" * 80 + "\n")
        f.write("RAW VLM RESPONSE:\n")
        f.write("=" * 80 + "\n")
        f.write(raw_response)
        f.write("\n")
    
    # Save parsed JSON analysis
    analysis_file = analysis_dir / f"{safe_episode_id}_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            "episode_id": episode_id,
            "video_path": str(video_path),
            "analysis": parsed_analysis
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Saved raw response: {raw_response_file.name}")
    print(f"   💾 Saved JSON analysis: {analysis_file.name}")

# --- Video Analysis ---
def analyze_sequential_video(
    processor, model, device, video_path: Path
) -> Dict[str, any]:
    """
    Analyzes sequential video using Cosmos-Reason1-7B and transformers.
    Returns structured JSON data.
    No instruction used - analyzes purely from video content.
    """
    if not video_path.exists():
        return {"error": f"Video file not found: {video_path}"}
    
    video_size = video_path.stat().st_size
    if video_size < 1000:  # Less than 1KB
        return {"error": f"Video file appears corrupted (size: {video_size} bytes)."}
        
    print(f"   📹 Analyzing video: {video_path.name} ({video_size / (1024*1024):.2f} MB)")
    
    # Use prompt without instruction - focus only on blocks and containers
    prompt_text = COSMOS_SEQUENTIAL_PROMPT
    
    try:
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
                max_new_tokens=8192,  # Increased for comprehensive multi-action analysis
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
            
        print(f"   📝 Cosmos Response (first 200 chars): {response[:200]}...")
        # Extract JSON
        json_str = response
        if "```json" in json_str:
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str, flags=re.MULTILINE)
        
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                print("   ✅ Successfully parsed JSON analysis")
                
                # Save raw response and parsed JSON for analysis
                save_vlm_response(video_path, response, analysis)
                
                return analysis
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parsing failed: {e}")
                error_result = {"error": "Failed to parse JSON response", "raw_response": response}
                save_vlm_response(video_path, response, error_result)
                return error_result
        error_result = {"error": "No JSON found in response", "raw_response": response}
        save_vlm_response(video_path, response, error_result)
        return error_result
        
    except Exception as e:
        print(f"   ⚠️  Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        error_result = {"error": str(e)}
        # Save error response even if analysis failed
        try:
            save_vlm_response(video_path, f"Error: {str(e)}", error_result)
        except:
            pass  # Don't fail if saving fails
        return error_result

# --- PDDL Generation ---
def generate_static_domain(output_path: Path):
    """
    Generates a static, hardcoded PDDL domain file.
    This replaces the complex/misleading "induction" logic from the original.
    """
    print(f"\n📝 Generating static domain: {output_path}")
    
    types_str = "\n        ".join([
        "block - object",
        "container - object",
        "robot - object"
    ])
    
    predicates_str = "\n        ".join([
        "(on ?x - block ?y - block)",
        "(on-table ?x - block)",
        "(in ?x - block ?c - container)",
        "(clear ?x - object)",
        "(holding ?r - robot ?x - block)",
        "(empty ?r - robot)"
    ])
    
    actions = [
        """    (:action pick-up
        :parameters (?r - robot ?x - block)
        :precondition (and (empty ?r) (on-table ?x) (clear ?x))
        :effect (and (holding ?r ?x) (not (empty ?r)) (not (on-table ?x)) (not (clear ?x)))
    )""",
        """    (:action put-down
        :parameters (?r - robot ?x - block)
        :precondition (holding ?r ?x)
        :effect (and (empty ?r) (on-table ?x) (clear ?x) (not (holding ?r ?x)))
    )""",
        """    (:action stack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and (holding ?r ?x) (clear ?y))
        :effect (and (on ?x ?y) (clear ?x) (empty ?r) (not (holding ?r ?x)) (not (clear ?y)))
    )""",
        """    (:action unstack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and (empty ?r) (on ?x ?y) (clear ?x))
        :effect (and (holding ?r ?x) (clear ?y) (not (empty ?r)) (not (on ?x ?y)) (not (clear ?x)))
    )""",
        """    (:action put-in-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and (holding ?r ?x) (clear ?c))
        :effect (and (in ?x ?c) (empty ?r) (not (holding ?r ?x)) (not (clear ?c)))
    )""",
        """    (:action take-from-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and (empty ?r) (in ?x ?c))
        :effect (and (holding ?r ?x) (clear ?c) (not (empty ?r)) (not (in ?x ?c)))
    )"""
    ]
    actions_str = "\n\n".join(actions)
    domain_content = f"""(define (domain robot-manipulation)
    (:requirements :strips :typing)
    
    (:types
        {types_str}
    )
    
    (:predicates
        {predicates_str}
    )
    
{actions_str}
)
"""
    with open(output_path, 'w') as f:
        f.write(domain_content)
    print("   ✅ Static domain file generated.")

def _normalize_object_name(name: str, color: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Helper to create a consistent object name and type.
    Focuses on objects that can be on the table (blocks, containers).
    All manipulatable objects are either 'block' or 'container' per domain.pddl.
    """
    name_lower = name.lower()
    color_lower = color.lower().strip()
    
    if not color_lower or not name_lower:
        return None, None
    
    # Ignore surfaces/tables - these are not manipulatable objects
    if "table" in name_lower or "surface" in name_lower or "desk" in name_lower:
        return None, None
    
    # All block-like objects (blocks, cubes, cylinders, rectangular, etc.) -> block
    block_keywords = [
        "block", "cube", "rectangular", "cylindrical", "cylinder", 
        "box", "square", "brick", "prism", "polygon"
    ]
    if any(keyword in name_lower for keyword in block_keywords):
        return f"{color_lower}_block", "block"
    
    # All container-like objects (cups, bowls, containers, etc.) -> container
    container_keywords = [
        "cup", "bowl", "container", "pot", "jar", "bin", "basket",
        "dish", "plate", "mug", "can"
    ]
    if any(keyword in name_lower for keyword in container_keywords):
        if "cup" in name_lower:
            return f"{color_lower}_cup", "container"
        elif "bowl" in name_lower:
            return f"{color_lower}_bowl", "container"
        else:
            return f"{color_lower}_container", "container"
    
    # If we can't categorize it, skip it (don't add unknown objects)
    return None, None

def _parse_state_from_json(objects_data: List[Dict]) -> Tuple[Dict[str, str], List[str]]:
    """
    Parses the 'objects' list from the VLN's JSON output.
    Focuses ONLY on objects that are on the table or manipulatable.
    
    Returns:
        - A dict of {object_name: object_type} where types are only 'block', 'container', or 'robot'
        - A list of PDDL predicates for this state.
    """
    objects_map = {}
    predicates = []
    
    if not isinstance(objects_data, list):
        return objects_map, predicates
        
    # First pass: Get all object names and types (only block/container/robot per domain)
    temp_obj_data = {}
    for obj_data in objects_data:
        if not isinstance(obj_data, dict):
            continue
        
        obj_name = obj_data.get("name", "")
        obj_color = obj_data.get("color", "")
        obj_location = obj_data.get("location", "")
        
        # Only process objects that are on the table or manipulatable
        location_lower = obj_location.lower()
        
        # Skip objects that are not on the table initially (unless they're containers)
        # Focus on what's visible and manipulatable on the table
        clean_name, obj_type = _normalize_object_name(obj_name, obj_color)
        
        if clean_name and obj_type in ["block", "container"]:
            objects_map[clean_name] = obj_type
            temp_obj_data[clean_name] = (location_lower, obj_name, obj_color)
    
    # Second pass: Generate predicates using normalized names
    for clean_name, (location_lower, raw_name, raw_color) in temp_obj_data.items():
        if "on table" in location_lower:
            predicates.append(f"(on-table {clean_name})")
            predicates.append(f"(clear {clean_name})")
            
        elif "in " in location_lower:
            # Location is "in [container_name]" or potentially "in [block_name]" (stacking mistake)
            container_name_raw = location_lower.split("in ")[-1].strip()
            # Find the normalized name - check for container first
            target_container_name = None
            target_block_name = None
            
            for data in objects_data:
                if isinstance(data, dict):
                    target_name, target_type = _normalize_object_name(
                        data.get("name", ""), 
                        data.get("color", "")
                    )
                    if not target_name:
                        continue
                    
                    # Try matching by name or normalized name
                    data_name = data.get("name", "").lower()
                    if (container_name_raw in data_name or 
                        data_name in container_name_raw or
                        target_name == container_name_raw):
                        if target_name in objects_map:
                            if target_type == "container":
                                target_container_name = target_name
                            elif target_type == "block":
                                target_block_name = target_name
                            break
            
            if target_container_name:
                # It's actually in a container
                predicates.append(f"(in {clean_name} {target_container_name})")
            elif target_block_name:
                # VLM said "in [block]" but it should be stacking (on)
                print(f"   ℹ️  Treating 'in {container_name_raw}' as stacking (on) for '{clean_name}'")
                predicates.append(f"(on {clean_name} {target_block_name})")
            else:
                print(f"   ⚠️  Could not find matching container or block '{container_name_raw}' for '{clean_name}'")
        elif "on " in location_lower:
            # Location is "on [block_name]" or "on top of [block_name]" or "on top [block_name]"
            # Handle various formats: "on green_block", "on top of green_block", "on top green_block"
            location_parts = location_lower.split("on ")[-1].strip()
            # Remove "top of ", "top ", etc. to get the actual block name
            location_parts = location_parts.replace("top of ", "").replace("top ", "").strip()
            
            # Also try to extract block name if it's in format like "green_block" or "green block"
            # Normalize the location_parts to match our naming convention
            potential_block_name = None
            
            # Try to find matching block by name - check all objects
            target_block_name = None
            for data in objects_data:
                data_name = data.get("name", "").lower()
                data_color = data.get("color", "")
                
                # Normalize the data object name
                normalized_data_name, _ = _normalize_object_name(data.get("name"), data_color)
                
                # Try exact match with normalized name
                if normalized_data_name and location_parts == normalized_data_name:
                    if normalized_data_name in objects_map:
                        target_block_name = normalized_data_name
                        break
                
                # Try matching by extracting color from location_parts
                # e.g., "green_block" -> color="green", name="block"
                if "_block" in location_parts:
                    color_from_location = location_parts.split("_block")[0]
                    if data_color.lower() == color_from_location:
                        if normalized_data_name in objects_map:
                            target_block_name = normalized_data_name
                            break
                
                # Try matching raw name (e.g., "green block" matches "green_block")
                if data_name.replace(" ", "_") == location_parts.replace(" ", "_"):
                    if normalized_data_name in objects_map:
                        target_block_name = normalized_data_name
                        break
                
                # Try partial match (e.g., "green" in "green_block")
                if data_color.lower() in location_parts or location_parts in data_color.lower():
                    if normalized_data_name in objects_map:
                        target_block_name = normalized_data_name
                        break
                
                # Try matching by creating normalized name from location_parts
                if "_" in location_parts:
                    parts = location_parts.split("_")
                    if len(parts) >= 2:
                        potential_color = parts[0]
                        potential_type = "_".join(parts[1:])
                        if data_color.lower() == potential_color and ("block" in potential_type or "cube" in potential_type):
                            if normalized_data_name in objects_map:
                                target_block_name = normalized_data_name
                                break
            
            if target_block_name:
                predicates.append(f"(on {clean_name} {target_block_name})")
            else:
                print(f"   ⚠️  Could not find matching block '{location_parts}' for '{clean_name}' in location '{location_lower}'")
                print(f"      Available objects: {list(objects_map.keys())}")
        
        # Add 'clear' predicate for containers that are on the table
        if objects_map[clean_name] == "container" and "on table" in location_lower:
             predicates.append(f"(clear {clean_name})")
    return objects_map, predicates

def generate_problem_pddl(video_id: str, analysis: Dict, output_dir: Path):
    """Generates a problem.pddl file based on Cosmos analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_file = output_dir / f"problem_{video_id.replace('+', '_').replace('-', '_')}.pddl"
    
    # Parse initial state - handle both dict and direct list formats
    initial_state = analysis.get("initial_state", {})
    if isinstance(initial_state, dict):
        initial_state_data = initial_state.get("objects", [])
    elif isinstance(initial_state, list):
        initial_state_data = initial_state
    else:
        initial_state_data = []
    init_objects, init_preds = _parse_state_from_json(initial_state_data)
    
    # Parse final state (goal) - handle both dict and direct list formats
    final_state = analysis.get("final_state", {})
    if isinstance(final_state, dict):
        final_state_data = final_state.get("objects", [])
    elif isinstance(final_state, list):
        final_state_data = final_state
    else:
        final_state_data = []
    goal_objects, goal_preds = _parse_state_from_json(final_state_data)
    
    # Combine all objects from both states
    all_objects = init_objects | goal_objects
    all_objects["robot1"] = "robot"
    
    objects_list = [f"{name} - {ptype}" for name, ptype in all_objects.items()]
    
    # Add robot initial state
    init_preds.append("(empty robot1)")
    
    # Filter goal: The goal is only the predicates from the final state
    # that are *not* present in the initial state.
    init_preds_set = set(init_preds)
    final_goal_preds = [p for p in goal_preds if p not in init_preds_set]
    # --- Format PDDL File ---
    problem_name = video_id.replace('+', '_').replace('-', '_')
    objects_section = "\n        ".join(sorted(objects_list))
    init_section = "\n        ".join(sorted(list(set(init_preds))))
    
    if not final_goal_preds:
        print(f"   ⚠️  No goal predicates found for {video_id}. Using empty goal.")
        goal_section = "" # No specific goal
    elif len(final_goal_preds) == 1:
        goal_section = final_goal_preds[0]
    else:
        goal_preds_str = "\n            ".join(sorted(list(set(final_goal_preds))))
        goal_section = f"(and\n            {goal_preds_str}\n        )"
    problem_content = f"""(define (problem {problem_name})
    (:domain robot-manipulation)
    (:objects
        {objects_section}
    )
    (:init
        {init_section}
    )
    (:goal
        {goal_section}
    )
)
"""
    
    with open(problem_file, 'w') as f:
        f.write(problem_content)
    
    print(f"   ✅ Generated: {problem_file}")

# --- File Handling ---
def load_annotations() -> Dict[str, str]:
    """Loads DROID language annotations, handling Git LFS pointers."""
    annotations_path = Path(ANNOTATIONS_FILE)
    if not annotations_path.exists():
        print(f"   ⚠️  Annotations file not found: {ANNOTATIONS_FILE}")
        return {}
        
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if content.startswith("version https://git-lfs.github.com"):
            print("   ⚠️  Annotations file is a Git LFS pointer. Please run 'git lfs pull'.")
            return {}
            
        # Handle merge conflicts
        if '<<<<<<<' in content:
            print("   ⚠️  Merge conflict detected in annotations file. Attempting to parse...")
            if '=======' in content:
                content = content.split('=======')[-1]
            if '>>>>>>>' in content:
                content = content.split('>>>>>>>')[0]
                
        annotations = json.loads(content)
        print(f"   ✅ Loaded {len(annotations)} annotations")
        return annotations
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON decode error in {ANNOTATIONS_FILE}: {e}")
        return {}
    except Exception as e:
        print(f"   ⚠️  Error loading annotations: {e}")
        return {}

def _find_video_file(episode_dir: Path) -> Optional[Path]:
    """Finds the video file within an episode directory."""
    search_patterns = [
        "recordings/MP4/*.mp4",
        "recordings/*.mp4",
        "*.mp4",
        "recordings/MP4/*.avi",
        "recordings/*.avi",
        "*.avi",
        "recordings/MP4/*.mov",
        "recordings/*.mov",
        "*.mov",
    ]
    for pattern in search_patterns:
        video_files = list(episode_dir.glob(pattern))
        if video_files:
            return video_files[0]
    return None

# --- Main Execution ---
def main():
    """Main function to generate PDDL files from sequential video analysis."""
    print("=" * 70)
    print("🚀 PDDL GENERATION WITH COSMOS-REASON1-7B (Simplified)")
    print("=" * 70)
    
    # Load model
    try:
        processor, model, device = load_cosmos_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
        
    # Find videos
    video_dir = Path(VIDEO_DIR)
    if not video_dir.exists():
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return
    
    video_episodes = [d for d in video_dir.iterdir() if d.is_dir()]
    print(f"\n📁 Found {len(video_episodes)} video episodes")
    
    # STEP 1: Generate the static domain file ONCE
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING STATIC DOMAIN.PDDL")
    print("=" * 70)
    generate_static_domain(Path(OUTPUT_DOMAIN_FILE))
    
    # STEP 2: Analyze videos and generate problem files
    print("\n" + "=" * 70)
    print(f"STEP 2: ANALYZING VIDEOS & GENERATING PROBLEMS (Sample Size: {SAMPLE_SIZE})")
    print("=" * 70)
    
    all_analyses = []
    
    for i, episode_dir in enumerate(video_episodes[:SAMPLE_SIZE], 1):
        episode_id = episode_dir.name
        print(f"\n[{i}/{min(len(video_episodes), SAMPLE_SIZE)}] Processing: {episode_id}")
        
        # Find video file
        video_path = _find_video_file(episode_dir)
        if not video_path:
            print(f"   ⚠️  No video file found in {episode_dir}")
            continue
        
        # Analyze video (no annotations or instructions used - purely video-based analysis)
        analysis = analyze_sequential_video(processor, model, device, video_path)
        
        # Clear cache after each video to prevent cross-contamination
        clear_model_cache(model, device)
        
        # Generate problem file
        if "error" in analysis:
            print(f"   ⚠️  Skipping PDDL generation for {episode_id}: {analysis['error']}")
        else:
            generate_problem_pddl(episode_id, analysis, Path(OUTPUT_PROBLEM_DIR))
            all_analyses.append(analysis)
        
        # Reload model after analyzing each video (to prevent cross-contamination)
        # Skip reload after the last video since we're done with analysis
        if i < min(len(video_episodes), SAMPLE_SIZE):
            print("   🔄 Reloading model to clear context...")
            unload_model(model, processor)
            processor, model, device = load_cosmos_model()
    
    print("\n" + "=" * 70)
    print("✅ PDDL GENERATION COMPLETE!")
    print(f"   Generated {len(all_analyses)} problem files in {OUTPUT_PROBLEM_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()

