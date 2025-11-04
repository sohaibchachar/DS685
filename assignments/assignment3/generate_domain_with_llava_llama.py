#!/usr/bin/env python3
"""
PDDL Domain Generation using LLaVA-1.6 (VLM) and Llama-3 (LLM)
Based on the provided code structure, adapted for DROID dataset.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
import torch
from PIL import Image
import cv2
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import LlavaNextProcessor

# Model IDs
VLM_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
# Llama-3 requires authentication, so we'll use alternatives
LLM_MODEL_ID = "meta-llama/Llama-3-8B-Instruct"  # Requires auth
LLM_MODEL_ID_FALLBACK = "microsoft/Phi-3-mini-4k-instruct"  # Open alternative

# Configuration
VIDEO_DIR = "raw_videos"
ANNOTATIONS_FILE = "droid_language_annotations.json"
OUTPUT_DOMAIN_FILE = "domain.pddl"
SAMPLE_SIZE = 10  # Number of videos to process
BLOCK_KEYWORDS = ["block", "stack", "place", "put", "pick", "move", "remove"]

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


def load_models():
    """Loads the VLM and LLM with 4-bit quantization."""
    print("🔧 Loading models (this may take a few minutes)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    try:
        # Load VLM (LLaVA-1.6)
        print("   🚀 Loading LLaVA-1.6 (VLM)...")
        vlm_processor = LlavaNextProcessor.from_pretrained(VLM_MODEL_ID)
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("   ✅ LLaVA-1.6 loaded")
    except Exception as e:
        print(f"   ⚠️  LLaVA-1.6 failed to load: {e}")
        print("   💡 Falling back to BLIP...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        vlm_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
        print("   ✅ BLIP loaded (fallback)")
    
    try:
        # Load LLM (Llama 3 - requires authentication)
        print("   🚀 Loading Llama-3-8B-Instruct (LLM)...")
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("   ✅ Llama-3 loaded")
    except Exception as e:
        print(f"   ⚠️  Llama-3 failed to load: {e}")
        print("   💡 Trying alternative LLM (Phi-3-mini)...")
        try:
            # Fallback to Phi-3-mini (open, no auth required)
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID_FALLBACK)
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID_FALLBACK,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # Use eager attention to avoid cache compatibility issues
            )
            print("   ✅ Phi-3-mini loaded (alternative LLM)")
        except Exception as e2:
            print(f"   ⚠️  Phi-3-mini also failed: {e2}")
            print("   💡 Will use default actions (no LLM)")
            llm_tokenizer = None
            llm_model = None
    
    print("   ✅ Models loaded successfully.")
    return {
        "vlm_processor": vlm_processor,
        "vlm_model": vlm_model,
        "llm_tokenizer": llm_tokenizer,
        "llm_model": llm_model,
        "device": device
    }


def load_annotations(base_dir: Path) -> Dict:
    """Load and parse language annotations."""
    annotations_path = base_dir / ANNOTATIONS_FILE
    print(f"📖 Loading annotations from {annotations_path}...")
    
    try:
        with open(annotations_path, 'r') as f:
            content = f.read()
        
        # Handle Git LFS markers if present
        if '<<<<<<< HEAD' in content:
            start_idx = content.find('{')
            end_idx = content.rfind('}', 0, content.find('>>>>>>>'))
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx + 1]
        
        annotations = json.loads(content)
        print(f"   ✅ Loaded {len(annotations)} annotations")
        return annotations
    except Exception as e:
        print(f"   ⚠️  Error loading annotations: {e}")
        return {}


def load_data_samples(base_dir: Path, annotations: Dict) -> tuple[List[Path], List[str], List[str]]:
    """Loads video paths and instructions from DROID dataset."""
    video_dir = base_dir / VIDEO_DIR
    print(f"📁 Loading video samples from {video_dir}...")
    
    video_paths = []
    instructions = []
    episode_ids = []
    
    # Find all video folders (episode IDs)
    if not video_dir.exists():
        print(f"   ⚠️  Video directory not found: {video_dir}")
        return video_paths, instructions, episode_ids
    
    episode_folders = [d for d in video_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(episode_folders)} video episodes")
    
    count = 0
    for episode_folder in episode_folders:
        if count >= SAMPLE_SIZE:
            break
        
        episode_id = episode_folder.name
        
        # Get instruction from annotations
        annot = annotations.get(episode_id, {})
        if isinstance(annot, dict):
            instruction = annot.get('language_instruction1', '')
            if not instruction:
                instruction = annot.get('language_instruction2', '')
            if not instruction:
                instruction = annot.get('language_instruction3', '')
        else:
            instruction = ''
        
        # Check if instruction contains block keywords
        if instruction and any(kw in instruction.lower() for kw in BLOCK_KEYWORDS):
            # Find video file
            video_path = None
            for ext in ['*.mp4', '*.MP4', '*.avi', '*.mov']:
                videos = list(episode_folder.rglob(ext))
                if videos:
                    video_path = videos[0]
                    break
            
            if video_path and video_path.exists():
                video_paths.append(video_path)
                instructions.append(instruction)
                episode_ids.append(episode_id)
                count += 1
                print(f"   [{count}/{SAMPLE_SIZE}] {episode_id}: {instruction[:60]}...")
    
    print(f"   ✅ Found {len(instructions)} sample block tasks.")
    return video_paths, instructions, episode_ids


def get_first_frame(video_path: Path) -> Optional[Image.Image]:
    """Extracts the first frame from a video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"   ⚠️  Error extracting frame from {video_path.name}: {e}")
        return None


def discover_vocabulary_vlm(models: Dict, video_paths: List[Path]) -> tuple[List[str], List[str]]:
    """Uses VLM to discover types and predicates from frames."""
    print("🔍 Discovering vocabulary from video frames (using VLM)...")
    
    frames = []
    for video_path in video_paths:
        frame = get_first_frame(video_path)
        if frame:
            frames.append(frame)
    
    if not frames:
        print("   ⚠️  No frames extracted, using default vocabulary")
        return ["block", "container", "robot", "surface"], [
            "(holding ?r - robot ?b - block)",
            "(on ?o1 - block ?o2 - object)",
            "(clear ?o - block)",
            "(in ?o - block ?c - container)",
            "(on-table ?o - block)",
            "(empty ?c - container)"
        ]
    
    print(f"   📸 Analyzing {len(frames)} frames...")
    
    discovered_types = set(["block", "container", "robot", "surface", "object"])  # Base types
    discovered_predicates = set()
    
    vlm_processor = models["vlm_processor"]
    vlm_model = models["vlm_model"]
    device = models["device"]
    
    # Create prompt text (without <image> placeholder for LLaVA)
    prompt_text = (
        "You are a PDDL domain expert. Analyze this image from a robotics task.\n"
        "1. What are the basic object *types*? (e.g., 'block', 'container', 'robot', 'surface').\n"
        "2. What are the key *predicates* (relationships)? (e.g., 'on', 'clear', 'holding', 'in', 'on-table').\n"
        "Provide your answer as a JSON object with two keys: 'types' and 'predicates'."
    )
    
    for i, frame in enumerate(frames):
        try:
            # Check if this is LLaVA processor
            if isinstance(vlm_processor, LlavaNextProcessor):
                # LLaVA-1.6: Use message format, then apply_chat_template
                # Based on batch_video_to_pddl.py pattern
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"},
                    ]
                }]
                # Apply chat template to convert messages to text
                chat = vlm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                # Process with images and text separately (signature: images, text)
                inputs = vlm_processor(images=frame, text=chat, return_tensors="pt").to(device)
                # Remove image_sizes if present to avoid compatibility issues
                if "image_sizes" in inputs:
                    del inputs["image_sizes"]
            elif hasattr(vlm_processor, 'apply_chat_template'):
                # Other LLaVA-style processors
                inputs = vlm_processor(prompt_text, images=[frame], return_tensors="pt").to(device)
            else:
                # BLIP style
                inputs = vlm_processor(frame, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if hasattr(vlm_model, 'generate'):
                    output_ids = vlm_model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    if isinstance(vlm_processor, LlavaNextProcessor):
                        response = vlm_processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    else:
                        response = vlm_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                else:
                    # BLIP fallback
                    output_ids = vlm_model.generate(**inputs, max_length=200)
                    response = vlm_processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    types_list = data.get('types', [])
                    predicates_list = data.get('predicates', [])
                    
                    discovered_types.update([t.lower().strip() for t in types_list])
                    discovered_predicates.update([p.lower().strip() for p in predicates_list])
                    
                    print(f"      [{i+1}/{len(frames)}] Found {len(types_list)} types, {len(predicates_list)} predicates")
            except Exception as e:
                print(f"      [{i+1}/{len(frames)}] JSON parsing failed: {e}")
                print(f"         Response: {response[:200]}...")
        except Exception as e:
            print(f"      [{i+1}/{len(frames)}] VLM processing failed: {e}")
    
    # Format predicates for PDDL
    formatted_predicates = []
    
    # Standard predicates
    if "on" in discovered_predicates or any("on" in p for p in discovered_predicates):
        formatted_predicates.append("(on ?o1 - block ?o2 - object)")
    if "clear" in discovered_predicates or any("clear" in p for p in discovered_predicates):
        formatted_predicates.append("(clear ?o - block)")
    if "holding" in discovered_predicates or any("holding" in p for p in discovered_predicates):
        formatted_predicates.append("(holding ?r - robot ?b - block)")
    if "in" in discovered_predicates or any("in" in p for p in discovered_predicates):
        formatted_predicates.append("(in ?o - block ?c - container)")
    if "on-table" in discovered_predicates or any("on-table" in p or "ontable" in p for p in discovered_predicates):
        formatted_predicates.append("(on-table ?o - block)")
    if "empty" in discovered_predicates or any("empty" in p for p in discovered_predicates):
        formatted_predicates.append("(empty ?c - container)")
    
    # Default predicates if none found
    if not formatted_predicates:
        formatted_predicates = [
            "(holding ?r - robot ?b - block)",
            "(on ?o1 - block ?o2 - object)",
            "(clear ?o - block)",
            "(in ?o - block ?c - container)",
            "(on-table ?o - block)",
            "(empty ?c - container)"
        ]
    
    # Ensure base types
    final_types = list(discovered_types)
    if "block" not in final_types:
        final_types.append("block")
    if "container" not in final_types:
        final_types.append("container")
    if "robot" not in final_types:
        final_types.append("robot")
    if "surface" not in final_types:
        final_types.append("surface")
    
    return final_types, formatted_predicates


def induce_actions_llm(models: Dict, instructions: List[str]) -> str:
    """Uses LLM to generate PDDL actions from instructions."""
    print("🤖 Inducing actions from instructions (using LLM)...")
    
    if models["llm_model"] is None or models["llm_tokenizer"] is None:
        print("   ⚠️  LLM not available, using default actions")
        return generate_default_actions()
    
    # Create system prompt
    system_prompt = (
        "You are an expert in AI planning and PDDL. "
        "Your task is to generate complete, syntactically correct PDDL `(:action ...)` blocks "
        "based on user's instructions for robot manipulation tasks. "
        "The domain includes types: 'block', 'container', 'robot', 'surface' (all inherit from 'object'). "
        "The predicates are: (holding ?r - robot ?b - block), (on ?o1 - block ?o2 - object), "
        "(clear ?o - block), (in ?o - block ?c - container), (on-table ?o - block), (empty ?c - container)."
    )
    
    # Create user prompt with sample instructions
    unique_instructions = list(set(instructions))[:5]  # Limit to 5 examples
    instruction_list = "\n".join(f"- {inst}" for inst in unique_instructions)
    user_prompt = (
        f"Here are example instructions from robot manipulation tasks:\n{instruction_list}\n\n"
        "Based on these instructions, generate PDDL `(:action ...)` blocks for common robot manipulation actions "
        "such as 'pick-up', 'place', 'put-in-container', 'remove-from-container', and 'stack'. "
        "Each action should have proper :parameters, :precondition, and :effect. "
        "Provide *only* the PDDL action blocks, starting with '(:action ...'."
    )
    
    llm_tokenizer = models["llm_tokenizer"]
    llm_model = models["llm_model"]
    device = models["device"]
    
    try:
        # Create messages for chat template (works for Llama-3, Phi-3, etc.)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Try to use chat template
        if hasattr(llm_tokenizer, 'apply_chat_template'):
            try:
                prompt = llm_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: simple format
                prompt = f"{system_prompt}\n\n{user_prompt}\n\nASSISTANT:"
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nASSISTANT:"
        
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Try generation with proper cache handling
            # For Phi-3, disable cache to avoid DynamicCache compatibility issues
            try:
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=False,  # Disable cache for Phi-3 compatibility
                    pad_token_id=llm_tokenizer.pad_token_id if hasattr(llm_tokenizer, 'pad_token_id') and llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id if hasattr(llm_tokenizer, 'eos_token_id') else None
                )
            except Exception as gen_error:
                # Fallback: try without cache or with different parameters
                print(f"      ⚠️  Generation error: {gen_error}")
                try:
                    # Try without cache and with different parameters
                    outputs = llm_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=llm_tokenizer.pad_token_id if hasattr(llm_tokenizer, 'pad_token_id') and llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id if hasattr(llm_tokenizer, 'eos_token_id') else None,
                        output_attentions=False,
                        output_hidden_states=False
                    )
                except Exception as e2:
                    print(f"      ⚠️  Generation failed again: {e2}")
                    # Last resort: use default actions
                    raise e2
        
        response = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract action blocks from response
        if user_prompt in response:
            action_text = response.split(user_prompt)[-1].strip()
        else:
            action_text = response
        
        # Find all '(:action ...)' blocks
        actions = re.findall(r"\(\s*:action[\s\S]*?\)\s*\)", action_text, re.DOTALL)
        
        if actions:
            return "\n\n".join(actions)
        else:
            print("   ⚠️  No actions found in LLM response, using defaults")
            return generate_default_actions()
            
    except Exception as e:
        print(f"   ⚠️  LLM generation failed: {e}")
        print("   💡 Using default actions")
        return generate_default_actions()


def generate_default_actions() -> str:
    """Generate default PDDL actions if LLM is not available."""
    return """
    (:action pick-up
        :parameters (?r - robot ?b - block)
        :precondition (and (on-table ?b) (clear ?b))
        :effect (and (holding ?r ?b) (not (on-table ?b)) (not (clear ?b)))
    )
    
    (:action place
        :parameters (?r - robot ?b - block ?o - object)
        :precondition (and (holding ?r ?b) (clear ?o))
        :effect (and (on ?b ?o) (not (holding ?r ?b)) (clear ?b))
    )
    
    (:action put-in-container
        :parameters (?r - robot ?b - block ?c - container)
        :precondition (and (holding ?r ?b) (empty ?c))
        :effect (and (in ?b ?c) (not (holding ?r ?b)) (not (empty ?c)))
    )
    
    (:action remove-from-container
        :parameters (?r - robot ?b - block ?c - container)
        :precondition (and (in ?b ?c))
        :effect (and (holding ?r ?b) (not (in ?b ?c)) (empty ?c))
    )
    
    (:action stack
        :parameters (?r - robot ?b1 - block ?b2 - block)
        :precondition (and (holding ?r ?b1) (clear ?b2))
        :effect (and (on ?b1 ?b2) (not (holding ?r ?b1)) (clear ?b1))
    )
"""


def main():
    base_dir = Path(__file__).parent
    
    print("="*70)
    print("🚀 PDDL DOMAIN GENERATION WITH LLaVA-1.6 + Llama-3")
    print("="*70)
    print()
    
    # 1. Load models
    models = load_models()
    print()
    
    # 2. Load annotations
    annotations = load_annotations(base_dir)
    print()
    
    # 3. Load data samples
    video_paths, instructions, episode_ids = load_data_samples(base_dir, annotations)
    print()
    
    if not instructions:
        print("⚠️  No block-related videos found. Check KEYWORDS and video directory.")
        return
    
    # 4. Discover Vocabulary using VLM
    types, predicates = discover_vocabulary_vlm(models, video_paths)
    print(f"   📊 Discovered Types: {types}")
    print(f"   📊 Discovered Predicates: {len(predicates)} predicates")
    print()
    
    # 5. Induce Actions using LLM
    actions_str = induce_actions_llm(models, instructions)
    print(f"   📝 Generated Actions:\n{actions_str[:200]}...")
    print()
    
    # 6. Assemble Domain
    print("📝 Assembling PDDL domain...")
    
    # Format types
    types_set = set(types)
    types_list = ["block", "container", "robot", "surface"]
    types_list = [t for t in types_list if t in types_set] + [t for t in types_set if t not in types_list]
    
    types_str = "\n        ".join(types_list) if len(types_list) > 1 else types_list[0]
    if len(types_list) > 1:
        types_str = f"{types_list[0]} container robot surface - object"
    else:
        types_str = "block container robot surface - object"
    
    # Format predicates
    predicates_str = "\n        ".join(predicates)
    
    domain_pddl = f"""(define (domain robot-manipulation)
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
    
    # 7. Save File
    output_path = base_dir / OUTPUT_DOMAIN_FILE
    with open(output_path, "w") as f:
        f.write(domain_pddl)
    
    print("="*70)
    print("✅ DOMAIN GENERATION COMPLETE!")
    print(f"📄 Saved domain to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

