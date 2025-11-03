"""Batch process videos from raw_videos folder to generate PDDL domain and problem files.

This script processes all videos in raw_videos (one from each lab folder) and generates:
- A unified PDDL domain file (domain.pddl)
- Individual problem files for each video (problem_<episode_id>.pddl)

Uses LLaVA for video understanding with prompt engineering techniques from NVIDIA's guide:
https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/

Usage:
    python batch_video_to_pddl.py --raw-videos-dir raw_videos --llava
    python batch_video_to_pddl.py --raw-videos-dir raw_videos --llava --max-videos 5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def extract_frames(video_path: Path, frames_dir: Path, fps: float = 0.5) -> None:
    """Extract frames from video at specified FPS."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def analyze_video_with_llava(
    frames_dir: Path,
    model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
    max_frames: int = 10,
) -> Dict[str, Any]:
    """Analyze video frames using LLaVA with video understanding prompts.
    
    Based on NVIDIA prompt engineering guide for video understanding:
    - Sequential visual understanding
    - Temporal action detection
    - Structured output extraction
    """
    try:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        import torch
        from PIL import Image
    except Exception as e:
        raise RuntimeError(
            "LLaVA dependencies missing. Install: pip install 'transformers>=4.41' accelerate safetensors sentencepiece pillow torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    try:
        processor = LlavaProcessor.from_pretrained(model_id, token=hf_token, use_fast=False)
    except Exception as e:
        print(f"   ⚠️  Warning: {e}, trying with use_fast=True")
        processor = LlavaProcessor.from_pretrained(model_id, token=hf_token, use_fast=True)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, device_map="auto", token=hf_token
    )

    jpgs = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return {"captions": [], "actions": [], "objects": [], "frames": 0}

    # Video understanding prompts based on NVIDIA guide
    prompts = [
        "What happened in this video? Elaborate on the visual and narrative elements in detail. Highlight all actions performed by the robot.",
        "What objects are visible in this video? List all objects including blocks, containers, and surfaces.",
        "What actions did the robot perform? List all manipulation actions in sequence.",
        "What is the initial state? Describe what objects are where at the beginning.",
        "What is the goal state? Describe what the robot is trying to achieve.",
    ]

    captions: List[tuple[str, str]] = []
    actions: List[str] = []
    objects: List[str] = []
    initial_state: List[str] = []
    goal_state: List[str] = []

    # Analyze first few frames for initial state
    if len(jpgs) >= 2:
        initial_frames = jpgs[:min(3, len(jpgs))]
        for img_path in initial_frames:
            try:
                image = Image.open(img_path).convert("RGB")
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts[3]},  # Initial state prompt
                        {"type": "image"},
                    ]
                }]
                chat = processor.apply_chat_template(message, add_generation_prompt=True)
                inputs = processor(images=image, text=chat, return_tensors="pt")
                # Remove image_sizes if present to avoid compatibility issues
                if "image_sizes" in inputs:
                    del inputs["image_sizes"]
                # Handle tensor movement properly
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype.is_floating_point:
                            processed_inputs[k] = v.to(device, dtype=dtype)
                        else:
                            processed_inputs[k] = v.to(device)
                    else:
                        processed_inputs[k] = v
                with torch.no_grad():
                    output = model.generate(**processed_inputs, max_new_tokens=128, do_sample=False)
                # Decode output properly
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        text = processor.decode(output[0], skip_special_tokens=True)
                    else:
                        text = processor.decode(output, skip_special_tokens=True)
                else:
                    text = processor.decode(output[0], skip_special_tokens=True)
                # Extract just the generated text (remove prompt)
                if isinstance(text, str):
                    if "ASSISTANT:" in text:
                        text = text.split("ASSISTANT:")[-1].strip()
                    elif "assistant:" in text:
                        text = text.split("assistant:")[-1].strip()
                initial_state.append(text)
            except Exception as e:
                print(f"   ⚠️  Warning: Error processing initial state frame {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
                initial_state.append(f"Error: {str(e)}")

    # Analyze middle frames for actions
    if len(jpgs) >= 4:
        middle_frames = jpgs[len(jpgs)//4:len(jpgs)//2]
        for img_path in middle_frames[:3]:
            try:
                image = Image.open(img_path).convert("RGB")
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts[2]},  # Actions prompt
                        {"type": "image"},
                    ]
                }]
                chat = processor.apply_chat_template(message, add_generation_prompt=True)
                inputs = processor(images=image, text=chat, return_tensors="pt")
                # Remove image_sizes if present to avoid compatibility issues
                if "image_sizes" in inputs:
                    del inputs["image_sizes"]
                # Handle tensor movement properly
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype.is_floating_point:
                            processed_inputs[k] = v.to(device, dtype=dtype)
                        else:
                            processed_inputs[k] = v.to(device)
                    else:
                        processed_inputs[k] = v
                with torch.no_grad():
                    output = model.generate(**processed_inputs, max_new_tokens=128, do_sample=False)
                # Decode output properly
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        text = processor.decode(output[0], skip_special_tokens=True)
                    else:
                        text = processor.decode(output, skip_special_tokens=True)
                else:
                    text = processor.decode(output[0], skip_special_tokens=True)
                # Extract just the generated text (remove prompt)
                if isinstance(text, str):
                    if "ASSISTANT:" in text:
                        text = text.split("ASSISTANT:")[-1].strip()
                    elif "assistant:" in text:
                        text = text.split("assistant:")[-1].strip()
                actions.append(text)
            except Exception as e:
                print(f"   ⚠️  Warning: Error processing actions frame {img_path.name}: {e}")
                actions.append(f"Error: {str(e)}")

    # Analyze last frames for goal state
    if len(jpgs) >= 2:
        final_frames = jpgs[-min(3, len(jpgs)):]
        for img_path in final_frames:
            try:
                image = Image.open(img_path).convert("RGB")
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts[4]},  # Goal state prompt
                        {"type": "image"},
                    ]
                }]
                chat = processor.apply_chat_template(message, add_generation_prompt=True)
                inputs = processor(images=image, text=chat, return_tensors="pt")
                # Remove image_sizes if present to avoid compatibility issues
                if "image_sizes" in inputs:
                    del inputs["image_sizes"]
                # Handle tensor movement properly
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype.is_floating_point:
                            processed_inputs[k] = v.to(device, dtype=dtype)
                        else:
                            processed_inputs[k] = v.to(device)
                    else:
                        processed_inputs[k] = v
                with torch.no_grad():
                    output = model.generate(**processed_inputs, max_new_tokens=128, do_sample=False)
                # Decode output properly
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        text = processor.decode(output[0], skip_special_tokens=True)
                    else:
                        text = processor.decode(output, skip_special_tokens=True)
                else:
                    text = processor.decode(output[0], skip_special_tokens=True)
                # Extract just the generated text (remove prompt)
                if isinstance(text, str):
                    if "ASSISTANT:" in text:
                        text = text.split("ASSISTANT:")[-1].strip()
                    elif "assistant:" in text:
                        text = text.split("assistant:")[-1].strip()
                goal_state.append(text)
            except Exception as e:
                print(f"   ⚠️  Warning: Error processing goal state frame {img_path.name}: {e}")
                goal_state.append(f"Error: {str(e)}")

    # Overall video description
    if jpgs:
        # Use middle frame for overall description
        middle_idx = len(jpgs) // 2
        image = Image.open(jpgs[middle_idx]).convert("RGB")
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompts[0]},  # Overall description
                {"type": "image"},
            ]
        }]
        chat = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(images=image, text=chat, return_tensors="pt")
        # Remove image_sizes if present to avoid compatibility issues
        if "image_sizes" in inputs:
            del inputs["image_sizes"]
        # Handle tensor movement properly
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    processed_inputs[k] = v.to(device, dtype=dtype)
                else:
                    processed_inputs[k] = v.to(device)
            else:
                processed_inputs[k] = v
        with torch.no_grad():
            output = model.generate(**processed_inputs, max_new_tokens=256, do_sample=False)
        # Decode output properly - handle both single and batch outputs
        try:
            if isinstance(output, torch.Tensor):
                if output.dim() > 1:
                    overall_desc = processor.decode(output[0], skip_special_tokens=True)
                else:
                    overall_desc = processor.decode(output, skip_special_tokens=True)
            else:
                overall_desc = processor.decode(output[0], skip_special_tokens=True)
            # Extract just the generated text (remove prompt)
            if isinstance(overall_desc, str):
                if "ASSISTANT:" in overall_desc:
                    overall_desc = overall_desc.split("ASSISTANT:")[-1].strip()
                elif "assistant:" in overall_desc:
                    overall_desc = overall_desc.split("assistant:")[-1].strip()
            captions.append(("overall", overall_desc))
        except Exception as e:
            print(f"   ⚠️  Warning: Error decoding output: {e}")
            captions.append(("overall", "Unable to decode description"))

    return {
        "captions": captions,
        "actions": actions,
        "objects": objects,
        "initial_state": initial_state,
        "goal_state": goal_state,
        "frames": len(jpgs),
    }


def extract_objects_and_actions(vlm_output: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Extract objects and actions from VLM output."""
    objects = set()
    actions = set()
    
    # Simple keyword extraction
    keywords = {
        "objects": ["block", "cube", "box", "container", "bowl", "cup", "table", "surface", "object"],
        "actions": ["pick", "place", "stack", "move", "put", "grasp", "release", "open", "close"],
    }
    
    # Extract from captions
    for caption_item in vlm_output.get("captions", []):
        if isinstance(caption_item, tuple) and len(caption_item) >= 2:
            text = caption_item[1].lower()
        elif isinstance(caption_item, str):
            text = caption_item.lower()
        else:
            text = str(caption_item).lower()
        
        for obj_kw in keywords["objects"]:
            if obj_kw in text:
                objects.add(obj_kw)
        for act_kw in keywords["actions"]:
            if act_kw in text:
                actions.add(act_kw)
    
    # Extract from actions list
    for action_text in vlm_output.get("actions", []):
        if isinstance(action_text, str):
            text = action_text.lower()
            for act_kw in keywords["actions"]:
                if act_kw in text:
                    actions.add(act_kw)
    
    # Extract from initial and goal states
    for state_list in [vlm_output.get("initial_state", []), vlm_output.get("goal_state", [])]:
        for state_text in state_list:
            if isinstance(state_text, str):
                text = state_text.lower()
                for obj_kw in keywords["objects"]:
                    if obj_kw in text:
                        objects.add(obj_kw)
                for act_kw in keywords["actions"]:
                    if act_kw in text:
                        actions.add(act_kw)
    
    return sorted(list(objects)), sorted(list(actions))


def write_domain(domain_path: Path, all_actions: set, all_objects: set) -> None:
    """Write comprehensive PDDL domain file based on extracted information."""
    
    # Types based on common robot manipulation domain
    types = """
        block - object
        container - object
        surface - object
        robot - agent
    """
    
    # Predicates based on common manipulation scenarios
    predicates = """
        (holding ?r - robot ?o - block)
        (on ?o1 - block ?o2 - object)
        (clear ?o - block)
        (in ?o - block ?c - container)
        (on-table ?o - block)
        (empty ?c - container)
        (open ?c - container)
        (closed ?c - container)
    """
    
    # Actions based on extracted actions
    actions = []
    
    if "pick" in all_actions or "grasp" in all_actions:
        actions.append("""
    (:action pick
        :parameters (?r - robot ?o - block)
        :precondition (and (clear ?o) (on-table ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)) (not (on-table ?o)))
    )
""")
    
    if "place" in all_actions or "put" in all_actions:
        actions.append("""
    (:action place
        :parameters (?r - robot ?o - block ?dst - surface)
        :precondition (and (holding ?r ?o))
        :effect (and (on-table ?o) (clear ?o) (not (holding ?r ?o)))
    )
""")
    
    if "stack" in all_actions:
        actions.append("""
    (:action stack
        :parameters (?r - robot ?o1 - block ?o2 - block)
        :precondition (and (holding ?r ?o1) (clear ?o2))
        :effect (and (on ?o1 ?o2) (clear ?o1) (not (holding ?r ?o1)) (not (clear ?o2)))
    )
""")
    
    if "put" in all_actions or "in" in [kw.lower() for kw in all_objects]:
        actions.append("""
    (:action put-in
        :parameters (?r - robot ?o - block ?c - container)
        :precondition (and (holding ?r ?o) (open ?c))
        :effect (and (in ?o ?c) (not (holding ?r ?o)) (clear ?o))
    )
""")
    
    if "open" in all_actions:
        actions.append("""
    (:action open-container
        :parameters (?r - robot ?c - container)
        :precondition (and (closed ?c))
        :effect (and (open ?c) (not (closed ?c)))
    )
""")
    
    if "close" in all_actions:
        actions.append("""
    (:action close-container
        :parameters (?r - robot ?c - container)
        :precondition (and (open ?c))
        :effect (and (closed ?c) (not (open ?c)))
    )
""")
    
    # Default actions if none extracted
    if not actions:
        actions = ["""
    (:action pick
        :parameters (?r - robot ?o - block)
        :precondition (and (clear ?o) (on-table ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)) (not (on-table ?o)))
    )
    (:action place
        :parameters (?r - robot ?o - block ?dst - surface)
        :precondition (and (holding ?r ?o))
        :effect (and (on-table ?o) (clear ?o) (not (holding ?r ?o)))
    )
    (:action stack
        :parameters (?r - robot ?o1 - block ?o2 - block)
        :precondition (and (holding ?r ?o1) (clear ?o2))
        :effect (and (on ?o1 ?o2) (clear ?o1) (not (holding ?r ?o1)) (not (clear ?o2)))
    )
"""]
    
    content = f"""(define (domain robot-manipulation)
    (:requirements :strips :typing)
    (:types
{types}
    )
    (:predicates
{predicates}
    )
{"".join(actions)}
)
"""
    domain_path.write_text(content)


def write_problem(problem_path: Path, episode_id: str, vlm_output: Dict[str, Any]) -> None:
    """Write PDDL problem file based on VLM analysis."""
    
    objects, actions = extract_objects_and_actions(vlm_output)
    
    # Generate objects based on VLM output
    num_blocks = max(2, len([o for o in objects if "block" in o or "cube" in o]))
    block_objects = [f"block{i}" for i in range(1, num_blocks + 1)]
    
    containers = []
    if "container" in objects or "bowl" in objects or "cup" in objects:
        containers = ["bowl1", "cup1"]
    
    # Initial state (from VLM analysis)
    init = []
    if vlm_output.get("initial_state"):
        # Simple heuristics based on common patterns
        init.extend([f"(on-table {block})" for block in block_objects[:2]])
        init.extend([f"(clear {block})" for block in block_objects])
        if containers:
            init.append("(open bowl1)")
            init.append("(empty bowl1)")
    else:
        # Default initial state
        init = [
            "(on-table block1)",
            "(on-table block2)",
            "(clear block1)",
            "(clear block2)",
        ]
    
    # Goal state (from VLM analysis)
    goal = ""
    if vlm_output.get("goal_state"):
        # Try to infer goal from VLM output
        goal_text = " ".join(vlm_output["goal_state"]).lower()
        if "stack" in goal_text or "stack" in actions:
            goal = "(on block1 block2)"
        elif "in" in goal_text or "bowl" in goal_text or "cup" in goal_text:
            goal = "(in block1 bowl1)"
        else:
            goal = "(on block1 block2)"
    else:
        # Default goal
        goal = "(on block1 block2)"
    
    # Build comment section with VLM analysis
    comments = []
    if vlm_output.get("captions"):
        comments.append("; VLM Analysis:")
        for frame, caption in vlm_output["captions"][:3]:
            comments.append(f";   {frame}: {caption[:100]}...")
    if vlm_output.get("actions"):
        comments.append(f"; Extracted actions: {vlm_output['actions']}")
    if vlm_output.get("initial_state"):
        comments.append(f"; Initial state hints: {vlm_output['initial_state'][:2]}")
    if vlm_output.get("goal_state"):
        comments.append(f"; Goal state hints: {vlm_output['goal_state'][:2]}")
    
    comment_section = "\n    ".join(comments) if comments else "; No VLM analysis available"
    
    objects_list = " ".join(block_objects) + " - block"
    if containers:
        objects_list += "\n        " + " ".join(containers) + " - container"
    objects_list += "\n        table1 - surface\n        robot1 - robot"
    
    content = f"""(define (problem {episode_id.replace('+', '_').replace('-', '_')})
    (:domain robot-manipulation)
    (:objects
        {objects_list}
    )
    (:init
        {' '.join(init)}
    )
    (:goal
        {goal}
    )
    {comment_section}
)
"""
    problem_path.write_text(content)


def validate_pddl(domain_path: Path, problem_path: Path) -> bool:
    """Validate PDDL files using Unified Planning library."""
    try:
        from unified_planning.io import PDDLReader
        reader = PDDLReader()
        reader.parse_problem(str(domain_path), str(problem_path))
        return True
    except ImportError:
        print("⚠️  Unified Planning not installed. Skipping validation.")
        print("   Install: pip install unified-planning")
        return False
    except Exception as e:
        print(f"❌ PDDL validation error: {e}")
        return False


def find_video_in_folder(folder_path: Path) -> Optional[Path]:
    """Find a non-stereo MP4 video in the folder."""
    mp4_dir = folder_path / "recordings" / "MP4"
    if not mp4_dir.exists():
        return None
    
    # Prefer non-stereo MP4 files
    videos = list(mp4_dir.glob("*.mp4"))
    non_stereo = [v for v in videos if "-stereo" not in v.name]
    if non_stereo:
        return non_stereo[0]
    elif videos:
        return videos[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos from raw_videos to generate PDDL files"
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Directory containing episode folders (default: raw_videos)",
    )
    parser.add_argument(
        "--llava",
        action="store_true",
        help="Use LLaVA for video understanding (default: True)",
    )
    parser.add_argument(
        "--llava-model",
        type=str,
        default="llava-hf/llava-v1.6-vicuna-7b-hf",
        help="LLaVA model ID",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum frames to analyze per video (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Frame extraction rate (default: 0.5)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate PDDL files using Unified Planning",
    )
    
    args = parser.parse_args()
    
    if not args.raw_videos_dir.exists():
        print(f"❌ Raw videos directory not found: {args.raw_videos_dir}")
        sys.exit(1)
    
    # Find all episode folders
    episode_folders = [d for d in args.raw_videos_dir.iterdir() if d.is_dir()]
    
    if not episode_folders:
        print(f"❌ No episode folders found in {args.raw_videos_dir}")
        sys.exit(1)
    
    print(f"📁 Found {len(episode_folders)} episode folders")
    
    # Limit number of videos if specified
    if args.max_videos:
        episode_folders = episode_folders[:args.max_videos]
    
    all_actions = set()
    all_objects = set()
    processed_videos = []
    
    # Process each video
    for i, episode_folder in enumerate(episode_folders, 1):
        episode_id = episode_folder.name
        print(f"\n[{i}/{len(episode_folders)}] Processing: {episode_id}")
        
        # Find video file
        video_path = find_video_in_folder(episode_folder)
        if not video_path:
            print(f"   ⚠️  No video found in {episode_folder}")
            continue
        
        print(f"   📹 Video: {video_path.name}")
        
        # Extract frames
        frames_dir = Path("frames") / episode_id
        print(f"   🎬 Extracting frames...")
        try:
            extract_frames(video_path, frames_dir, fps=args.fps)
            frame_count = len(list(frames_dir.glob("*.jpg")))
            print(f"   ✅ Extracted {frame_count} frames")
        except Exception as e:
            print(f"   ❌ Error extracting frames: {e}")
            continue
        
        # Analyze with VLM
        if args.llava:
            print(f"   🤖 Analyzing with LLaVA...")
            try:
                vlm_output = analyze_video_with_llava(
                    frames_dir,
                    model_id=args.llava_model,
                    max_frames=args.max_frames,
                )
                objects, actions = extract_objects_and_actions(vlm_output)
                all_actions.update(actions)
                all_objects.update(objects)
                print(f"   ✅ Analysis complete: {len(actions)} actions, {len(objects)} objects")
            except Exception as e:
                print(f"   ❌ Error analyzing video: {e}")
                continue
        
        # Generate problem file
        problem_file = Path(f"problem_{episode_id.replace('+', '_').replace('-', '_')}.pddl")
        write_problem(problem_file, episode_id, vlm_output)
        print(f"   ✅ Generated {problem_file}")
        
        processed_videos.append((episode_id, video_path, problem_file))
    
    # Generate unified domain file
    print(f"\n📝 Generating unified domain file...")
    domain_path = Path("domain.pddl")
    write_domain(domain_path, all_actions, all_objects)
    print(f"✅ Generated {domain_path}")
    
    # Validate PDDL files if requested
    if args.validate:
        print(f"\n🔍 Validating PDDL files...")
        for episode_id, _, problem_file in processed_videos:
            if validate_pddl(domain_path, problem_file):
                print(f"   ✅ {problem_file.name} is valid")
            else:
                print(f"   ❌ {problem_file.name} has errors")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Processing complete!")
    print(f"{'='*70}")
    print(f"📊 Processed: {len(processed_videos)} videos")
    print(f"📁 Domain file: {domain_path}")
    print(f"📄 Problem files: {len(processed_videos)}")
    print(f"🎯 Actions extracted: {sorted(all_actions)}")
    print(f"🎯 Objects extracted: {sorted(all_objects)}")


if __name__ == "__main__":
    main()

