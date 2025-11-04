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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
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
        # Use 4-bit quantization to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            COSMOS_MODEL_ID,
            quantization_config=quantization_config,
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
    prompt_text = f"""Analyze this sequential video showing a robot manipulation task.

Task Instruction: {instruction}

Provide a detailed sequential analysis in JSON format with:
1. "initial_state": Objects, positions, and relationships at the start
2. "goal_state": Desired final configuration  
3. "sequence_of_actions": List of actions performed in order
4. "object_types": Types of objects (e.g., block, container, robot, surface)
5. "predicates": Relationships between objects (e.g., on, in, holding, clear)
6. "state_transitions": How states change between key frames
7. "key_frames": Frame indices where significant state changes occur

Focus on:
- Physical common sense reasoning
- Spatial relationships
- Temporal sequence of actions
- Object manipulation states
- Sequential reasoning about what happens step by step

Format your response as valid JSON."""

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
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
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
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Extract JSON from response
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            response = answer_match.group(1).strip()
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                return analysis
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = json_match.group(0)
                json_str = json_str.replace("'", '"')
                try:
                    analysis = json.loads(json_str)
                    return analysis
                except:
                    pass
        
        # Fallback: return raw response
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
        types.update([t.lower() for t in analysis["object_types"]])
    
    # Extract from predicates field
    if "predicates" in analysis:
        predicates.update([p.lower() for p in analysis["predicates"]])
    
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


def generate_problem_pddl(video_id: str, instruction: str, analysis: Dict, output_dir: Path):
    """Generates a problem.pddl file for a video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_file = output_dir / f"problem_{video_id.replace('+', '_').replace('-', '_')}.pddl"
    
    # Extract initial and goal states
    initial_state = analysis.get("initial_state", {})
    goal_state = analysis.get("goal_state", {})
    
    # Convert to PDDL format (simplified)
    init_predicates = []
    goal_predicates = []
    
    # Simple conversion - you may need to adjust based on actual analysis format
    if isinstance(initial_state, dict):
        for key, value in initial_state.items():
            if isinstance(value, list):
                for item in value:
                    init_predicates.append(f"({key} {item})")
            elif value:
                init_predicates.append(f"({key} {value})")
    
    if isinstance(goal_state, dict):
        for key, value in goal_state.items():
            if isinstance(value, list):
                for item in value:
                    goal_predicates.append(f"({key} {item})")
            elif value:
                goal_predicates.append(f"({key} {value})")
    
    # Default if empty
    if not init_predicates:
        init_predicates = ["(on-table block1)", "(clear block1)"]
    if not goal_predicates:
        goal_predicates = ["(on block1 block2)"]
    
    problem_pddl = f"""(define (problem {video_id.replace('+', '_').replace('-', '_')})
    (:domain robot-manipulation)
    (:objects
        block1 block2 - block
        robot1 - robot
    )
    (:init
        {' '.join(init_predicates[:5])}  ; Initial state
    )
    (:goal
        (and {' '.join(goal_predicates[:3])})  ; Goal state
    )
)"""
    
    with open(problem_file, 'w') as f:
        f.write(problem_pddl)
    
    print(f"      ✅ Generated: {problem_file}")


def load_annotations() -> Dict[str, str]:
    """Loads DROID language annotations."""
    annotations_path = Path(ANNOTATIONS_FILE)
    if not annotations_path.exists():
        print(f"   ⚠️  Annotations file not found: {ANNOTATIONS_FILE}")
        return {}
    
    try:
        with open(annotations_path, 'r') as f:
            content = f.read()
            # Handle Git LFS and merge conflicts
            if content.startswith("version https://git-lfs.github.com"):
                lines = content.split('\n')
                json_start = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
                content = '\n'.join(lines[json_start:])
            if '<<<<<<<' in content:
                content = content.split('<<<<<<<')[0]
            annotations = json.loads(content)
        print(f"   ✅ Loaded {len(annotations)} annotations")
        return annotations
    except Exception as e:
        print(f"   ⚠️  Error loading annotations: {e}")
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
    
    # Process videos
    all_analyses = []
    all_types = set()
    all_predicates = set()
    all_actions = []
    
    for i, episode_dir in enumerate(video_episodes[:SAMPLE_SIZE], 1):
        episode_id = episode_dir.name
        print(f"\n[{i}/{min(len(video_episodes), SAMPLE_SIZE)}] Processing: {episode_id}")
        
        # Find video file
        video_files = list(episode_dir.glob("*.mp4"))
        if not video_files:
            print(f"   ⚠️  No video file found")
            continue
        
        video_path = video_files[0]
        print(f"   Video: {video_path.name}")
        
        # Get instruction
        instruction = annotations.get(episode_id, "Perform robot manipulation task")
        print(f"   Instruction: {instruction}")
        
        # Analyze sequential video
        analysis = analyze_sequential_video(processor, model, device, video_path, instruction)
        analysis["episode_id"] = episode_id
        analysis["instruction"] = instruction
        all_analyses.append(analysis)
        
        # Extract domain information
        types, predicates = extract_objects_and_predicates(analysis)
        all_types.update(types)
        all_predicates.update(predicates)
        
        actions = induce_actions_from_analysis(analysis)
        all_actions.extend(actions)
        
        # Generate problem file
        generate_problem_pddl(episode_id, instruction, analysis, Path(OUTPUT_PROBLEM_DIR))
    
    # Generate unified domain
    print(f"\n📝 Generating unified domain...")
    types_str = "\n        object"
    for t in sorted(all_types):
        if t != "object":
            types_str += f"\n        {t} - object"
    
    predicates_str = "\n        (on ?o1 - block ?o2 - object)"
    for p in sorted(all_predicates):
        if p not in ["on"]:
            predicates_str += f"\n        ({p} ?o - object)"
    
    actions_str = "\n\n".join(list(dict.fromkeys(all_actions)))  # Remove duplicates
    
    domain_pddl = f"""(define (domain robot-manipulation)
    (:requirements :strips :typing)
    (:types{types_str}
    )
    (:predicates{predicates_str}
    )
{actions_str}
)"""
    
    with open(OUTPUT_DOMAIN_FILE, 'w') as f:
        f.write(domain_pddl)
    
    print(f"✅ Generated domain: {OUTPUT_DOMAIN_FILE}")
    print(f"✅ Generated {len(all_analyses)} problem files in {OUTPUT_PROBLEM_DIR}/")
    print("\n" + "=" * 70)
    print("✅ PDDL GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

