#!/usr/bin/env python3
"""
PDDL Domain and Problem Generation using Cosmos-Reason1-7B
Sequential video understanding for generating domain.pddl and problem.pddl files.
Uses transformers instead of vLLM for WSL2 compatibility.
"""

# Set multiprocessing start method BEFORE any imports that use CUDA/vLLM
# This is required for vLLM to work with CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import torch
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Model ID for Cosmos-Reason1-7B (from HuggingFace)
# Based on Qwen2.5-VL-7B-Instruct architecture
COSMOS_MODEL_ID = "nvidia/Cosmos-Reason1-7B"

# Configuration
ANNOTATIONS_FILE = "droid_language_annotations.json"
VIDEO_DIR = "raw_videos"
OUTPUT_DOMAIN_FILE = "domain.pddl"
OUTPUT_PROBLEM_DIR = "problems"
SAMPLE_SIZE = 10  # Number of videos to process
BLOCK_KEYWORDS = ["block", "stack", "place", "put", "pick", "move", "remove"]
FRAME_RATE = 4  # FPS for video analysis (matches model training)


def load_cosmos_model():
    """
    Loads Cosmos-Reason1-7B model for sequential video understanding using transformers.
    Falls back from vLLM to transformers for WSL2 compatibility.
    Based on Qwen2.5-VL-7B-Instruct architecture.
    Reference: https://huggingface.co/nvidia/Cosmos-Reason1-7B
    """
    print("🔧 Loading Cosmos-Reason1-7B (sequential video understanding)...")
    print(f"   Model: {COSMOS_MODEL_ID}")
    print(f"   Architecture: Qwen2.5-VL-7B-Instruct")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please use a GPU-enabled environment.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    print(f"   Engine: Transformers (WSL2 compatible)")
    
    try:
        print(f"   🚀 Loading processor and model...")
        
        # Load processor for message formatting
        processor = AutoProcessor.from_pretrained(
            COSMOS_MODEL_ID,
            trust_remote_code=True
        )
        
        # Load model with transformers (WSL2 compatible)
        # Use Qwen2_5_VLForConditionalGeneration directly for generation capability
        # Try with quantization first, fallback to no quantization if it fails
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print(f"      Trying with 4-bit quantization...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                COSMOS_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        except Exception as quant_error:
            print(f"      ⚠️  Quantization failed: {quant_error}")
            print(f"      Trying without quantization...")
            # Fallback: load without quantization (requires more memory)
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


def analyze_sequential_video(
    processor, model, device, video_path: Path, instruction: str
) -> Dict[str, any]:
    """
    Analyzes sequential video to understand what's happening using Cosmos-Reason1-7B.
    Uses transformers for WSL2-compatible inference.
    Returns structured data about actions, objects, states, and transitions.
    """
    if not video_path.exists():
        return {"error": f"Video file not found: {video_path}"}
    
    print(f"   📹 Analyzing video with Cosmos-Reason1-7B (transformers)...")
    
    # Create sequential analysis prompt for Cosmos-Reason1-7B
    prompt_text = f"""You are analyzing a robot manipulation video. Your task is to understand:

1. INITIAL STATE (start of video):
   - List all objects present (blocks, containers, etc.) with their colors
   - Describe where each object is located (e.g., "green_block on table", "yellow_block in blue_cup", "orange_block on green_block")
   - Note any objects being held by the robot

2. SEQUENCE OF ACTIONS (what happens in the video):
   - Describe the sequence of actions performed (pick up, place, stack, etc.)

3. FINAL STATE (end of video):
   - List where each object ends up (e.g., "green_block in black_bowl", "orange_block on green_block")
   - Describe the final configuration

Instruction: {instruction}

Provide your analysis in JSON format with this exact structure:
{{
  "initial_state": {{
    "objects": [
      {{"name": "green_block", "location": "on table", "color": "green"}},
      {{"name": "blue_cup", "location": "on table", "color": "blue"}}
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
      {{"name": "blue_cup", "location": "on table", "color": "blue"}}
    ],
    "robot_holding": null
  }},
  "object_types": ["block", "container"],
  "predicates": ["on-table", "in", "on", "clear", "holding"]
}}

Focus on:
- Physical objects (blocks, containers/cups/bowls) - DO NOT include surfaces like tables
- Clear spatial relationships (on, in, clear)
- Color information for all objects
- Precise initial and final states"""

    try:
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
            },
        ]
        
        # Process with Qwen2.5-VL processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Use qwen_vl_utils.process_vision_info (returns 2 values: image_inputs, video_inputs)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate with transformers
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.05,
            )
        
        # Decode response - inputs is a dict, access via keys
        input_ids = inputs.get('input_ids')
        if input_ids is None:
            # Try to find input_ids in the dict
            for key, value in inputs.items():
                if 'input' in key.lower() and 'id' in key.lower():
                    input_ids = value
                    break
        
        if input_ids is not None:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
        else:
            # Fallback: just use generated_ids (remove prompt tokens manually)
            generated_ids_trimmed = generated_ids
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract JSON from response
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            response = answer_match.group(1).strip()
        
        # Print the raw response for debugging
        print(f"\n      📝 Cosmos Response (first 500 chars):")
        print(f"      {response[:500]}...")
        if len(response) > 500:
            print(f"      ... (truncated, total length: {len(response)} chars)")
        
        # Remove markdown code blocks if present
        json_str = response
        if "```json" in json_str:
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str, flags=re.MULTILINE)
        elif "```" in json_str:
            json_str = re.sub(r'```\s*', '', json_str)
        
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                print(f"\n      ✅ Successfully parsed JSON analysis")
                print(f"      📊 Analysis keys: {list(analysis.keys())}")
                return analysis
            except json.JSONDecodeError as e:
                print(f"\n      ⚠️  JSON parsing failed: {e}")
                # Try to fix common JSON issues
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                try:
                    analysis = json.loads(json_str)
                    print(f"      ✅ Fixed JSON parsing")
                    return analysis
                except Exception as e2:
                    print(f"      ⚠️  JSON fix failed: {e2}")
        
        # Fallback: return raw response
        print(f"\n      ⚠️  Using raw response as fallback")
        return {
            "raw_response": response,
            "instruction": instruction,
            "video_path": str(video_path)
        }
    
    except Exception as e:
        print(f"      ⚠️  Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "instruction": instruction}


# ... (rest of the functions remain the same - extract_objects_and_predicates, induce_actions_from_analysis, etc.)
# For brevity, I'll copy the key functions that need to work with the new signature

def extract_objects_and_predicates(analysis: Dict) -> Tuple[List[str], List[str]]:
    """Extracts object types and predicates from analysis."""
    types = set()
    predicates = set()
    
    # Extract from object_types field
    if "object_types" in analysis:
        obj_types = analysis["object_types"]
        if isinstance(obj_types, list):
            types.update([str(t).lower() for t in obj_types])
        elif isinstance(obj_types, dict):
            # If it's a dict, extract keys or values
            types.update([str(k).lower() for k in obj_types.keys()])
        elif isinstance(obj_types, str):
            types.add(obj_types.lower())
    
    # Extract from predicates field
    if "predicates" in analysis:
        preds = analysis["predicates"]
        if isinstance(preds, list):
            predicates.update([str(p).lower() for p in preds])
        elif isinstance(preds, dict):
            # If it's a dict, extract keys or values
            predicates.update([str(k).lower() for k in preds.keys()])
        elif isinstance(preds, str):
            predicates.add(preds.lower())
    
    # Infer from initial/goal states
    if "initial_state" in analysis:
        state = analysis["initial_state"]
        if isinstance(state, dict):
            # Look for object mentions
            for key, value in state.items():
                if isinstance(value, (list, dict)):
                    types.add(key.lower())
    
    # Default types if none found
    if not types:
        types = {"block", "container", "robot", "surface"}
    
    # Default predicates if none found
    if not predicates:
        predicates = {"on", "in", "holding", "clear", "on-table"}
    
    return list(types), list(predicates)


def induce_actions_from_analysis(analysis: Dict) -> List[str]:
    """Induces PDDL actions from the analysis."""
    actions = []
    
    if "sequence_of_actions" in analysis:
        action_sequence = analysis["sequence_of_actions"]
        if isinstance(action_sequence, list):
            # Create actions based on sequence
            unique_actions = set()
            for action in action_sequence:
                if isinstance(action, str):
                    unique_actions.add(action.lower())
                elif isinstance(action, dict) and "name" in action:
                    unique_actions.add(action["name"].lower())
            
            # Generate PDDL action templates
            for action_name in unique_actions:
                if "pick" in action_name or "grasp" in action_name:
                    actions.append("""    (:action pick-up
        :parameters (?r - robot ?b - block)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b) (not (on-table ?b)) (not (clear ?b)))
    )""")
                elif "place" in action_name or "put" in action_name:
                    actions.append("""    (:action place
        :parameters (?r - robot ?b - block ?o - object)
        :precondition (and (holding ?r ?b) (clear ?o))
        :effect (and (on ?b ?o) (not (holding ?r ?b)) (clear ?b))
    )""")
                elif "stack" in action_name:
                    actions.append("""    (:action stack
        :parameters (?r - robot ?b1 - block ?b2 - block)
        :precondition (and (holding ?r ?b1) (clear ?b2))
        :effect (and (on ?b1 ?b2) (not (holding ?r ?b1)) (clear ?b1))
    )""")
    
    # Default actions if none found
    if not actions:
        actions = [
            """    (:action pick-up
        :parameters (?r - robot ?b - block)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b) (not (on-table ?b)) (not (clear ?b)))
    )""",
            """    (:action place
        :parameters (?r - robot ?b - block ?o - object)
        :precondition (and (holding ?r ?b) (clear ?o))
        :effect (and (on ?b ?o) (not (holding ?r ?b)) (clear ?b))
    )"""
        ]
    
    return actions


def generate_domain_from_analyses(all_analyses: List[Dict], output_path: Path = Path("domain.pddl")):
    """
    Generates a PDDL domain file by analyzing all video analyses.
    Extracts common types, predicates, and actions from the analyses.
    """
    print(f"\n📝 Generating domain.pddl from video analyses...")
    
    # Collect all types, predicates, and actions from analyses
    all_types = set()
    all_predicates = set()
    all_action_patterns = set()
    
    # Standard types we always need
    all_types.update(["block", "container", "robot"])
    
    # Extract from each analysis
    import re
    for analysis in all_analyses:
        if "error" in analysis:
            continue
        
        # Extract types and predicates
        types, predicates = extract_objects_and_predicates(analysis)
        all_types.update(types)
        all_predicates.update(predicates)
        
        # Extract actions
        actions = induce_actions_from_analysis(analysis)
        for action in actions:
            # Extract action name from PDDL action string
            match = re.search(r'\(:action\s+(\w+)', action)
            if match:
                all_action_patterns.add(action)
    
    # Filter out invalid types
    all_types = {t for t in all_types if t not in ["object", "surface", "table", "objects"] and not t.startswith("{") and not "/" in t}
    
    # Standard predicates we always need (based on what we see in problems)
    standard_predicates = {
        "(on ?x - block ?y - block)",
        "(on-table ?x - block)",
        "(in ?x - block ?c - container)",
        "(clear ?x - object)",
        "(holding ?r - robot ?x - block)",
        "(empty ?r - robot)"
    }
    all_predicates.update(standard_predicates)
    
    # Standard actions we always need
    standard_actions = [
        """    (:action pick-up
        :parameters (?r - robot ?x - block)
        :precondition (and
            (empty ?r)
            (on-table ?x)
            (clear ?x)
        )
        :effect (and
            (holding ?r ?x)
            (not (empty ?r))
            (not (on-table ?x))
            (not (clear ?x))
        )
    )""",
        """    (:action put-down
        :parameters (?r - robot ?x - block)
        :precondition (holding ?r ?x)
        :effect (and
            (empty ?r)
            (on-table ?x)
            (clear ?x)
            (not (holding ?r ?x))
        )
    )""",
        """    (:action stack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and
            (holding ?r ?x)
            (clear ?y)
        )
        :effect (and
            (on ?x ?y)
            (clear ?x)
            (empty ?r)
            (not (holding ?r ?x))
            (not (clear ?y))
        )
    )""",
        """    (:action unstack
        :parameters (?r - robot ?x - block ?y - block)
        :precondition (and
            (empty ?r)
            (on ?x ?y)
            (clear ?x)
        )
        :effect (and
            (holding ?r ?x)
            (clear ?y)
            (not (empty ?r))
            (not (on ?x ?y))
            (not (clear ?x))
        )
    )""",
        """    (:action put-in-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and
            (holding ?r ?x)
            (clear ?c)
        )
        :effect (and
            (in ?x ?c)
            (empty ?r)
            (not (holding ?r ?x))
            (not (clear ?c))
        )
    )""",
        """    (:action take-from-container
        :parameters (?r - robot ?x - block ?c - container)
        :precondition (and
            (empty ?r)
            (in ?x ?c)
        )
        :effect (and
            (holding ?r ?x)
            (clear ?c)
            (not (empty ?r))
            (not (in ?x ?c))
        )
    )"""
    ]
    
    # Combine standard actions with discovered actions
    all_actions = standard_actions + list(all_action_patterns)
    # Remove duplicates (keep first occurrence)
    seen_action_names = set()
    unique_actions = []
    for action in all_actions:
        match = re.search(r'\(:action\s+(\w+)', action)
        if match:
            action_name = match.group(1)
            if action_name not in seen_action_names:
                seen_action_names.add(action_name)
                unique_actions.append(action)
    
    # Format types (ensure proper PDDL format: type - object)
    types_list = []
    for t in sorted(all_types):
        if t not in ["object"]:  # object is implicit base type
            types_list.append(f"{t} - object")
    types_str = "\n        ".join(types_list)
    
    # Format predicates (filter to only valid PDDL predicate format)
    valid_predicates = []
    for pred in all_predicates:
        # Must be a string and start with '(' and contain parameters
        if isinstance(pred, str) and pred.startswith("(") and "?" in pred:
            valid_predicates.append(pred)
        elif isinstance(pred, str) and pred.startswith("("):
            # Might be valid but check format
            if "-" in pred:  # Has type annotation
                valid_predicates.append(pred)
    
    # If no valid predicates found, use standard ones
    if not valid_predicates:
        valid_predicates = list(standard_predicates)
    
    predicates_str = "\n        ".join(sorted(valid_predicates))
    
    # Format actions
    actions_str = "\n\n".join(unique_actions)
    
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
    print(f"✅ Generated domain: {output_path}")
    print(f"   📊 Types: {len(all_types)} ({', '.join(sorted(all_types))})")
    print(f"   📊 Predicates: {len(all_predicates)}")
    print(f"   📊 Actions: {len(unique_actions)}")


def generate_problem_pddl(video_id: str, instruction: str, analysis: Dict, output_dir: Path):
    """Generates a problem.pddl file based on Cosmos analysis of initial and final states."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_file = output_dir / f"problem_{video_id.replace('+', '_').replace('-', '_')}.pddl"
    
    # Initialize objects and predicates
    objects_list = []
    init_predicates = []
    goal_predicates = []
    
    # Track objects we've seen
    seen_objects = set()
    
    # Parse initial_state from analysis
    initial_state = analysis.get("initial_state", {})
    final_state = analysis.get("final_state", {})
    
    # Extract objects from initial_state.objects array
    if isinstance(initial_state, dict) and "objects" in initial_state:
        initial_objects = initial_state["objects"]
        if isinstance(initial_objects, list):
            for obj_data in initial_objects:
                if isinstance(obj_data, dict):
                    obj_name = obj_data.get("name", "")
                    obj_color = obj_data.get("color", "")
                    obj_location = obj_data.get("location", "")
                    
                    # Create color-based name
                    if obj_color and obj_name:
                        # Extract base type from name
                        if "block" in obj_name.lower() or "cube" in obj_name.lower() or "rectangular" in obj_name.lower() or "cylindrical" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_block"
                        elif "cup" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_cup"
                        elif "bowl" in obj_name.lower() or "container" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_bowl"
                        else:
                            # For other objects (straw, mug, etc.), create a simple name
                            base_name = obj_name.lower().replace(" ", "_").replace("-", "_")
                            # Skip if it contains invalid characters like "/"
                            if "/" in base_name or "stacked" in base_name.lower():
                                continue
                            if "straw" in base_name or "mug" in base_name:
                                obj_name_clean = f"{obj_color.lower()}_{base_name}"
                            else:
                                obj_name_clean = f"{obj_color.lower()}_{base_name}"
                    else:
                        obj_name_clean = obj_name.lower().replace(" ", "_").replace("-", "_")
                        # Skip invalid names
                        if "/" in obj_name_clean or "stacked" in obj_name_clean.lower():
                            continue
                    
                    # Skip surfaces/tables
                    if "table" in obj_name_clean or "surface" in obj_name_clean:
                        continue
                    
                    if obj_name_clean and obj_name_clean not in seen_objects:
                        seen_objects.add(obj_name_clean)
                        # Determine type
                        if "block" in obj_name_clean or "cube" in obj_name_clean:
                            objects_list.append(f"{obj_name_clean} - block")
                        elif "cup" in obj_name_clean or "bowl" in obj_name_clean:
                            objects_list.append(f"{obj_name_clean} - container")
                        elif "straw" in obj_name_clean or "mug" in obj_name_clean:
                            # Skip non-manipulatable objects like straw, mug for now
                            # Or treat them as containers if needed
                            pass
                    
                    # Extract initial predicates from location (only for objects we added)
                    if obj_name_clean and obj_name_clean in seen_objects:
                        location_lower = obj_location.lower()
                        if "on table" in location_lower:
                            init_predicates.append(f"(on-table {obj_name_clean})")
                            init_predicates.append(f"(clear {obj_name_clean})")
                        elif "in " in location_lower:
                            # Extract container name from location (e.g., "in white_bowl" -> "white_bowl")
                            container_part = location_lower.split("in ")[1].split()[0]
                            # Create proper container name with color
                            if "_" in container_part:
                                container_name = container_part.replace("_", "_")
                            else:
                                # Extract color and type from container name
                                container_name = container_part
                                # Try to get color from other objects in initial state
                                for other_obj in initial_objects:
                                    if isinstance(other_obj, dict) and other_obj.get("name") == container_part:
                                        container_color = other_obj.get("color", "")
                                        if container_color:
                                            if "cup" in container_part.lower():
                                                container_name = f"{container_color.lower()}_cup"
                                            elif "bowl" in container_part.lower():
                                                container_name = f"{container_color.lower()}_bowl"
                                            else:
                                                container_name = f"{container_color.lower()}_{container_part.lower()}"
                                        break
                            
                            if container_name not in seen_objects:
                                seen_objects.add(container_name)
                                objects_list.append(f"{container_name} - container")
                            init_predicates.append(f"(in {obj_name_clean} {container_name})")
                        elif "on " in location_lower and "table" not in location_lower:
                            # Block is on another block
                            target_block = location_lower.split("on ")[1].split()[0]
                            target_name = f"{target_block}_block" if "block" not in target_block else target_block
                            if target_name not in seen_objects:
                                seen_objects.add(target_name)
                                objects_list.append(f"{target_name} - block")
                            init_predicates.append(f"(on {obj_name_clean} {target_name})")
        
        # Check if robot is holding something initially
        robot_holding = initial_state.get("robot_holding")
        if robot_holding:
            # Robot is holding something - remove on-table predicate for that object
            pass  # Will be handled by checking holding state
    
    # Parse final_state from analysis
    if isinstance(final_state, dict) and "objects" in final_state:
        final_objects = final_state["objects"]
        if isinstance(final_objects, list):
            for obj_data in final_objects:
                if isinstance(obj_data, dict):
                    obj_name = obj_data.get("name", "")
                    obj_color = obj_data.get("color", "")
                    obj_location = obj_data.get("location", "")
                    
                    # Create color-based name (same as initial state)
                    if obj_color and obj_name:
                        if "block" in obj_name.lower() or "cube" in obj_name.lower() or "rectangular" in obj_name.lower() or "cylindrical" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_block"
                        elif "cup" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_cup"
                        elif "bowl" in obj_name.lower() or "container" in obj_name.lower():
                            obj_name_clean = f"{obj_color.lower()}_bowl"
                        else:
                            base_name = obj_name.lower().replace(" ", "_").replace("-", "_")
                            # Skip invalid names
                            if "/" in base_name or "stacked" in base_name.lower():
                                continue
                            obj_name_clean = f"{obj_color.lower()}_{base_name}"
                    else:
                        obj_name_clean = obj_name.lower().replace(" ", "_").replace("-", "_")
                        # Skip invalid names
                        if "/" in obj_name_clean or "stacked" in obj_name_clean.lower():
                            continue
                    
                    # Skip surfaces/tables
                    if "table" in obj_name_clean or "surface" in obj_name_clean or "straw" in obj_name_clean or "mug" in obj_name_clean:
                        continue
                    
                    # Add object if not already added
                    if obj_name_clean and obj_name_clean not in seen_objects:
                        seen_objects.add(obj_name_clean)
                        if "block" in obj_name_clean or "cube" in obj_name_clean:
                            objects_list.append(f"{obj_name_clean} - block")
                        elif "cup" in obj_name_clean or "bowl" in obj_name_clean:
                            objects_list.append(f"{obj_name_clean} - container")
                    
                    # Extract goal predicates from location (only for objects we added)
                    if obj_name_clean and obj_name_clean in seen_objects:
                        location_lower = obj_location.lower()
                        if "in " in location_lower:
                            # Goal: object in container
                            container_part = location_lower.split("in ")[1].split()[0]
                            # Create proper container name
                            container_name = None
                            # Try to get color from other objects in final state
                            for other_obj in final_objects:
                                if isinstance(other_obj, dict):
                                    other_name = other_obj.get("name", "")
                                    if container_part in other_name.lower() or other_name.lower() == container_part:
                                        container_color = other_obj.get("color", "")
                                        if container_color:
                                            if "cup" in other_name.lower():
                                                container_name = f"{container_color.lower()}_cup"
                                            elif "bowl" in other_name.lower():
                                                container_name = f"{container_color.lower()}_bowl"
                                            else:
                                                container_name = f"{container_color.lower()}_bowl"
                                        else:
                                            container_name = f"{container_part}_bowl" if "bowl" in container_part.lower() else f"{container_part}_cup"
                                        break
                            
                            if not container_name:
                                container_name = f"{container_part}_bowl" if "bowl" in container_part.lower() else f"{container_part}_cup"
                            
                            if container_name not in seen_objects:
                                seen_objects.add(container_name)
                                objects_list.append(f"{container_name} - container")
                            goal_predicates.append(f"(in {obj_name_clean} {container_name})")
                        elif "on " in location_lower and "table" not in location_lower:
                            # Goal: object on another object - ensure target is added
                            target_block = location_lower.split("on ")[1].split()[0]
                            # Find the target block in final objects to get its color
                            target_name = None
                            for other_obj in final_objects:
                                if isinstance(other_obj, dict):
                                    other_name = other_obj.get("name", "")
                                    if target_block in other_name.lower() or other_name.lower() == target_block:
                                        other_color = other_obj.get("color", "")
                                        if other_color:
                                            target_name = f"{other_color.lower()}_block"
                                        else:
                                            target_name = f"{target_block}_block"
                                        break
                            if not target_name:
                                target_name = f"{target_block}_block" if "block" not in target_block else target_block
                            if target_name not in seen_objects:
                                seen_objects.add(target_name)
                                objects_list.append(f"{target_name} - block")
                            goal_predicates.append(f"(on {obj_name_clean} {target_name})")
                        elif "on table" in location_lower:
                            goal_predicates.append(f"(on-table {obj_name_clean})")
    
    # Add robot if not present
    if "robot1" not in seen_objects:
        objects_list.append("robot1 - robot")
    
    # Add robot initial state
    init_predicates.insert(0, "(empty robot1)")
    
    # Filter predicates to only include objects that are actually in objects_list
    # Remove predicates that reference objects not in objects_list
    declared_objects = set()
    for obj_decl in objects_list:
        obj_name = obj_decl.split(" - ")[0].strip()
        declared_objects.add(obj_name)
    
    # Filter init predicates - only keep predicates where ALL referenced objects are declared
    filtered_init_predicates = []
    for pred in init_predicates:
        # Extract all object names from predicate
        pred_str = pred
        # Find all object names mentioned in predicate
        referenced_objects = []
        for obj_name in declared_objects:
            # Check if object name appears in predicate (as whole word, not substring)
            import re
            pattern = r'\b' + re.escape(obj_name) + r'\b'
            if re.search(pattern, pred_str):
                referenced_objects.append(obj_name)
        
        # Also check for objects that might be referenced but not in declared_objects
        # Extract words from predicate that look like object names
        words = re.findall(r'\b\w+_\w+\b', pred_str)  # Match underscore-separated names
        for word in words:
            if word not in declared_objects:
                # This predicate references an undeclared object, skip it
                break
        else:
            # All referenced objects are declared, or predicate has no object references
            filtered_init_predicates.append(pred)
    
    # Filter goal predicates
    filtered_goal_predicates = []
    for pred in goal_predicates:
        pred_str = pred
        words = re.findall(r'\b\w+_\w+\b', pred_str)
        for word in words:
            if word not in declared_objects:
                break
        else:
            filtered_goal_predicates.append(pred)
    
    init_predicates = filtered_init_predicates
    goal_predicates = filtered_goal_predicates
    
    # Fallback if no analysis data - use instruction
    if not init_predicates and not goal_predicates:
        # Parse instruction to infer goal
        instruction_lower = instruction.lower()
        colors = []
        color_keywords = {"red", "green", "blue", "yellow", "orange", "black", "white", "brown", "pink", "purple"}
        for color in color_keywords:
            if color in instruction_lower:
                colors.append(color)
        
        block_color = None
        container_color = None
        if "block" in instruction_lower:
            words = instruction_lower.split()
            for i, word in enumerate(words):
                if word == "block" and i > 0:
                    if words[i-1] in color_keywords:
                        block_color = words[i-1]
        if "cup" in instruction_lower or "bowl" in instruction_lower:
            words = instruction_lower.split()
            for i, word in enumerate(words):
                if (word == "cup" or word == "bowl") and i > 0:
                    if words[i-1] in color_keywords:
                        container_color = words[i-1]
        
        if "in" in instruction_lower and ("cup" in instruction_lower or "bowl" in instruction_lower):
            block_name = f"{block_color}_block" if block_color else (f"{colors[0]}_block" if colors else "block1")
            container_name = f"{container_color}_bowl" if container_color else (f"{colors[1]}_bowl" if len(colors) > 1 else "container1")
            if not any(block_name in obj for obj in objects_list):
                objects_list.append(f"{block_name} - block")
            if not any(container_name in obj for obj in objects_list):
                objects_list.append(f"{container_name} - container")
            init_predicates = [f"(on-table {block_name})", f"(clear {block_name})", f"(clear {container_name})"]
            goal_predicates = [f"(in {block_name} {container_name})"]
        elif "on" in instruction_lower and "block" in instruction_lower:
            block1_color = None
            block2_color = None
            words = instruction_lower.split()
            for i, word in enumerate(words):
                if word == "block" and i > 0:
                    if words[i-1] in color_keywords:
                        if block1_color is None:
                            block1_color = words[i-1]
                        else:
                            block2_color = words[i-1]
            
            block1_name = f"{block1_color}_block" if block1_color else (f"{colors[0]}_block" if colors else "block1")
            block2_name = f"{block2_color}_block" if block2_color else (f"{colors[1]}_block" if len(colors) > 1 else "block2")
            if not any(block1_name in obj for obj in objects_list):
                objects_list.append(f"{block1_name} - block")
            if not any(block2_name in obj for obj in objects_list):
                objects_list.append(f"{block2_name} - block")
            init_predicates = [f"(on-table {block1_name})", f"(on-table {block2_name})", f"(clear {block1_name})", f"(clear {block2_name})"]
            goal_predicates = [f"(on {block2_name} {block1_name})"]
    
    # Ensure we have at least basic predicates
    if not init_predicates:
        init_predicates = ["(empty robot1)"]
    if not goal_predicates:
        # Only add fallback goal if we have objects
        if any("block" in obj for obj in objects_list):
            # Get first block from objects_list
            for obj_decl in objects_list:
                if "block" in obj_decl:
                    block_name = obj_decl.split(" - ")[0].strip()
                    goal_predicates = [f"(on-table {block_name})"]
                    break
    
    # Build problem file content
    problem_name = video_id.replace('+', '_').replace('-', '_')
    
    objects_section = "\n        ".join(objects_list) if objects_list else "robot1 - robot"
    init_section = "\n        ".join(init_predicates)
    goal_section = "\n            ".join(goal_predicates) if len(goal_predicates) > 1 else goal_predicates[0] if goal_predicates else "(on-table block1)"
    
    # Format goal section properly
    if len(goal_predicates) > 1:
        goal_section = f"(and\n            {goal_section}\n        )"
    
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
    
    print(f"      ✅ Generated: {problem_file}")


def load_annotations() -> Dict[str, str]:
    """Loads DROID language annotations."""
    annotations_path = Path(ANNOTATIONS_FILE)
    if not annotations_path.exists():
        print(f"   ⚠️  Annotations file not found: {ANNOTATIONS_FILE}")
        return {}
    
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Handle empty file
            if not content:
                print(f"   ⚠️  Annotations file is empty")
                return {}
            
            # Handle Git LFS and merge conflicts
            if content.startswith("version https://git-lfs.github.com"):
                lines = content.split('\n')
                json_start = next((i for i, line in enumerate(lines) if line.strip().startswith('{')), None)
                if json_start is not None:
                    content = '\n'.join(lines[json_start:])
                else:
                    print(f"   ⚠️  Could not find JSON in Git LFS file")
                    return {}
            
            # Handle merge conflicts - take the content after =======
            if '<<<<<<<' in content and '=======' in content:
                # Skip the HEAD section, take the part after =======
                parts = content.split('=======')
                if len(parts) > 1:
                    content = parts[1]
                    # Remove the ending marker if present
                    if '>>>>>>>' in content:
                        content = content.split('>>>>>>>')[0]
            elif '<<<<<<<' in content:
                content = content.split('<<<<<<<')[0]
            
            # Skip if still empty after processing
            if not content.strip():
                print(f"   ⚠️  Annotations file has no valid JSON content")
                return {}
            
            annotations = json.loads(content)
            if isinstance(annotations, dict):
                print(f"   ✅ Loaded {len(annotations)} annotations")
                return annotations
            else:
                print(f"   ⚠️  Annotations is not a dict, got {type(annotations)}")
                return {}
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON decode error: {e}")
        print(f"   💡 Trying to fix JSON...")
        # Try to extract JSON from corrupted content
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find first { and last }
                start = content.find('{')
                end = content.rfind('}')
                if start >= 0 and end > start:
                    content = content[start:end+1]
                    annotations = json.loads(content)
                    print(f"   ✅ Fixed and loaded {len(annotations)} annotations")
                    return annotations
        except:
            pass
        return {}
    except Exception as e:
        print(f"   ⚠️  Error loading annotations: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main function to generate PDDL files from sequential video analysis."""
    print("=" * 70)
    print("🚀 PDDL GENERATION WITH COSMOS-REASON1-7B")
    print("   Sequential Video Understanding for Domain & Problem Generation")
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
    video_dir = Path(VIDEO_DIR)
    if not video_dir.exists():
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return
    
    video_episodes = [d for d in video_dir.iterdir() if d.is_dir()]
    print(f"\n📁 Found {len(video_episodes)} video episodes")
    
    # STEP 1: Analyze all videos first to collect domain information
    print("\n" + "=" * 70)
    print("STEP 1: ANALYZING ALL VIDEOS")
    print("=" * 70)
    
    all_analyses = []
    
    for i, episode_dir in enumerate(video_episodes[:SAMPLE_SIZE], 1):
        episode_id = episode_dir.name
        print(f"\n[{i}/{min(len(video_episodes), SAMPLE_SIZE)}] Analyzing: {episode_id}")
        
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
    
    domain_path = Path("domain.pddl")
    generate_domain_from_analyses(all_analyses, domain_path)
    
    # STEP 3: Generate problem files for each video based on the domain
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING PROBLEM FILES")
    print("=" * 70)
    
    for i, analysis in enumerate(all_analyses, 1):
        if "error" in analysis:
            print(f"\n[{i}/{len(all_analyses)}] ⚠️  Skipping {analysis.get('episode_id', 'unknown')}: analysis failed")
            continue
        
        episode_id = analysis.get("episode_id", "unknown")
        instruction = analysis.get("instruction", "Perform robot manipulation task")
        print(f"\n[{i}/{len(all_analyses)}] Processing: {episode_id}")
        
        # Generate problem file
        generate_problem_pddl(episode_id, instruction, analysis, Path(OUTPUT_PROBLEM_DIR))
    
    print(f"\n✅ Generated {len([a for a in all_analyses if 'error' not in a])} problem files in {OUTPUT_PROBLEM_DIR}/")
    print("\n" + "=" * 70)
    print("✅ PDDL GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

