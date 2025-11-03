"""Generate PDDL domain and problem files using Image2PDDL framework approach.

This script implements the Image2PDDL framework methodology:
1. Analyze video frames (initial, middle, final states)
2. Extract descriptions using Video-LLaVA or mPLUG-Owl
3. Generate PDDL problems from VLM descriptions

Based on Image2PDDL framework: https://arxiv.org/abs/2501.17665

Usage:
    python image2pddl_generator.py --raw-videos-dir raw_videos --video-llava
    python image2pddl_generator.py --raw-videos-dir raw_videos --mplug-owl
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_frames(video_path: Path, frames_dir: Path, fps: float = 0.5) -> None:
    """Extract frames from video at specified FPS."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def analyze_video_with_video_llava(
    frames_dir: Path,
    model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
    max_frames: int = 8,
) -> Dict[str, Any]:
    """Analyze video frames using LLaVA following Image2PDDL framework methodology.
    
    Image2PDDL approach:
    - Analyze initial state frames separately
    - Analyze final/goal state frames separately
    - Extract objects, actions, and relationships from descriptions
    """
    try:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "LLaVA dependencies missing. Install: pip install 'transformers>=4.41' accelerate safetensors sentencepiece pillow torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    print(f"   📦 Loading LLaVA model (Image2PDDL approach): {model_id}")
    try:
        processor = LlavaProcessor.from_pretrained(model_id, token=hf_token, use_fast=False)
    except Exception:
        processor = LlavaProcessor.from_pretrained(model_id, token=hf_token, use_fast=True)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="auto", token=hf_token
    )

    jpgs = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return {"initial_state": "", "goal_state": "", "objects": [], "actions": [], "description": "", "frames": 0}

    # Image2PDDL framework: Analyze initial and final states separately
    initial_frames = jpgs[:min(3, len(jpgs))]
    final_frames = jpgs[-min(3, len(jpgs)):]
    middle_frame = jpgs[len(jpgs) // 2] if len(jpgs) > 2 else jpgs[0]

    initial_state_desc = ""
    goal_state_desc = ""
    overall_desc = ""
    
    # Prompt templates following Image2PDDL methodology
    initial_prompt = "Describe the initial state of this robot manipulation scene. List all objects, their positions, and relationships. Be specific about what is on the table, what the robot is holding, and what containers are open or closed."
    goal_prompt = "Describe the goal state of this robot manipulation scene. What should the final arrangement be? What objects should be where?"
    overall_prompt = "What happened in this robot manipulation video? Describe all actions performed by the robot, the objects involved, and the sequence of operations."

    # Analyze initial state
    if initial_frames:
        try:
            images = [Image.open(img).convert('RGB') for img in initial_frames]
            prompt = f"USER: <image>\n{initial_prompt}\nASSISTANT:"
            inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            
            initial_state_desc = processor.decode(output[0], skip_special_tokens=True)
            # Extract just the assistant response
            if "ASSISTANT:" in initial_state_desc:
                initial_state_desc = initial_state_desc.split("ASSISTANT:")[-1].strip()
        except Exception as e:
            print(f"   ⚠️  Error analyzing initial state: {e}")
            initial_state_desc = "Initial state analysis unavailable"

    # Analyze goal state
    if final_frames:
        try:
            images = [Image.open(img).convert('RGB') for img in final_frames]
            prompt = f"USER: <image>\n{goal_prompt}\nASSISTANT:"
            inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            
            goal_state_desc = processor.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in goal_state_desc:
                goal_state_desc = goal_state_desc.split("ASSISTANT:")[-1].strip()
        except Exception as e:
            print(f"   ⚠️  Warning: Error analyzing goal state: {e}")
            goal_state_desc = "Goal state analysis unavailable"

    # Analyze overall description
    try:
        image = Image.open(middle_frame).convert('RGB')
        prompt = f"USER: <image>\n{overall_prompt}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        overall_desc = processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in overall_desc:
            overall_desc = overall_desc.split("ASSISTANT:")[-1].strip()
    except Exception as e:
        print(f"   ⚠️  Warning: Error analyzing overall description: {e}")
        overall_desc = "Overall description unavailable"

    # Extract objects and actions from descriptions
    all_text = (initial_state_desc + " " + goal_state_desc + " " + overall_desc).lower()
    
    objects = set()
    actions = set()
    
    # Object detection
    if any(word in all_text for word in ["block", "cube", "brick", "box", "object"]):
        objects.add("block")
    if any(word in all_text for word in ["container", "bowl", "cup", "jar", "pot", "vessel"]):
        objects.add("container")
    if any(word in all_text for word in ["table", "surface", "desk", "platform", "counter"]):
        objects.add("table")
    
    # Action detection
    if any(word in all_text for word in ["pick", "pick up", "picking", "grasp", "grasping", "grab", "grabbing", "lift", "lifting", "holding"]):
        actions.add("pick")
    if any(word in all_text for word in ["place", "placing", "put", "putting", "set", "setting", "drop", "dropping", "deposit"]):
        actions.add("place")
    if any(word in all_text for word in ["stack", "stacking", "stacked", "pile", "piling", "on top"]):
        actions.add("stack")
    if any(word in all_text for word in ["put in", "putting in", "place in", "placing in", "inside", "into", "contain"]):
        actions.add("put-in")
    if any(word in all_text for word in ["open", "opening", "uncover", "uncovering", "unseal"]):
        actions.add("open")
    if any(word in all_text for word in ["close", "closing", "cover", "covering", "shut", "shutting", "seal"]):
        actions.add("close")

    return {
        "initial_state": initial_state_desc,
        "goal_state": goal_state_desc,
        "description": overall_desc,
        "objects": sorted(list(objects)),
        "actions": sorted(list(actions)),
        "frames": len(jpgs),
    }


def analyze_video_with_mplug_owl(
    frames_dir: Path,
    model_id: str = "MAGAer13/mplug-owl-llama-7b",
    max_frames: int = 8,
) -> Dict[str, Any]:
    """Analyze video frames using mPLUG-Owl following Image2PDDL framework."""
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "mPLUG-Owl dependencies missing. Install: pip install 'transformers>=4.41' accelerate safetensors pillow torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    print(f"   📦 Loading mPLUG-Owl model: {model_id}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", token=hf_token
        )
    except Exception as e:
        raise RuntimeError(f"Could not load mPLUG-Owl model: {e}")

    jpgs = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return {"initial_state": "", "goal_state": "", "objects": [], "actions": [], "description": "", "frames": 0}

    initial_frames = jpgs[:min(3, len(jpgs))]
    final_frames = jpgs[-min(3, len(jpgs)):]
    middle_frame = jpgs[len(jpgs) // 2] if len(jpgs) > 2 else jpgs[0]

    initial_state_desc = ""
    goal_state_desc = ""
    overall_desc = ""

    # Image2PDDL prompts
    initial_prompt = "Describe the initial state: List all objects, their positions, and relationships."
    goal_prompt = "Describe the goal state: What should the final arrangement be?"
    overall_prompt = "What happened in this robot manipulation video? Describe all actions and objects."

    # Analyze initial state
    if initial_frames:
        try:
            images = [Image.open(img).convert('RGB') for img in initial_frames]
            prompt = f"<|image_pad|> {initial_prompt}"
            inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            
            initial_state_desc = processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"   ⚠️  Error analyzing initial state: {e}")
            initial_state_desc = "Initial state analysis unavailable"

    # Analyze goal state
    if final_frames:
        try:
            images = [Image.open(img).convert('RGB') for img in final_frames]
            prompt = f"<|image_pad|> {goal_prompt}"
            inputs = processor(images=images, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            
            goal_state_desc = processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"   ⚠️  Warning: Error analyzing goal state: {e}")
            goal_state_desc = "Goal state analysis unavailable"

    # Analyze overall
    try:
        image = Image.open(middle_frame).convert('RGB')
        prompt = f"<|image_pad|> {overall_prompt}"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        
        overall_desc = processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"   ⚠️  Warning: Error analyzing overall description: {e}")
        overall_desc = "Overall description unavailable"

    # Extract objects and actions
    all_text = (initial_state_desc + " " + goal_state_desc + " " + overall_desc).lower()
    
    objects = set()
    actions = set()
    
    if any(word in all_text for word in ["block", "cube", "brick", "box"]):
        objects.add("block")
    if any(word in all_text for word in ["container", "bowl", "cup", "jar"]):
        objects.add("container")
    if any(word in all_text for word in ["table", "surface", "desk"]):
        objects.add("table")
    
    if any(word in all_text for word in ["pick", "pick up", "picking", "grasp", "grab", "lift"]):
        actions.add("pick")
    if any(word in all_text for word in ["place", "placing", "put", "putting", "set", "drop"]):
        actions.add("place")
    if any(word in all_text for word in ["stack", "stacking", "stacked", "pile"]):
        actions.add("stack")
    if any(word in all_text for word in ["put in", "putting in", "place in", "inside", "into"]):
        actions.add("put-in")
    if any(word in all_text for word in ["open", "opening"]):
        actions.add("open")
    if any(word in all_text for word in ["close", "closing", "shut"]):
        actions.add("close")

    return {
        "initial_state": initial_state_desc,
        "goal_state": goal_state_desc,
        "description": overall_desc,
        "objects": sorted(list(objects)),
        "actions": sorted(list(actions)),
        "frames": len(jpgs),
    }


def write_domain(domain_path: Path, all_actions: Set[str], all_objects: Set[str]) -> None:
    """Write PDDL domain file following Unified Planning standards."""
    
    types = """        block - object
        container - object
        surface - object
        robot - agent"""
    
    predicates = """        (holding ?r - robot ?o - block)
        (on ?o1 - block ?o2 - object)
        (clear ?o - block)
        (in ?o - block ?c - container)
        (on-table ?o - block)
        (empty ?c - container)
        (open ?c - container)
        (closed ?c - container)"""
    
    actions = []
    
    # Pick action (always included)
    actions.append("""    (:action pick
        :parameters (?r - robot ?o - block)
        :precondition (and (clear ?o) (on-table ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)) (not (on-table ?o)))
    )
""")
    
    # Place action (always included)
    actions.append("""    (:action place
        :parameters (?r - robot ?o - block ?dst - surface)
        :precondition (and (holding ?r ?o))
        :effect (and (on-table ?o) (clear ?o) (not (holding ?r ?o)))
    )
""")
    
    # Stack action
    if "stack" in all_actions:
        actions.append("""    (:action stack
        :parameters (?r - robot ?o1 - block ?o2 - block)
        :precondition (and (holding ?r ?o1) (clear ?o2))
        :effect (and (on ?o1 ?o2) (clear ?o1) (not (holding ?r ?o1)) (not (clear ?o2)))
    )
""")
    
    # Put-in action
    if "put-in" in all_actions:
        actions.append("""    (:action put-in
        :parameters (?r - robot ?o - block ?c - container)
        :precondition (and (holding ?r ?o) (open ?c))
        :effect (and (in ?o ?c) (not (holding ?r ?o)) (clear ?o))
    )
""")
    
    # Open container action
    if "container" in all_objects:
        actions.append("""    (:action open-container
        :parameters (?r - robot ?c - container)
        :precondition (and (closed ?c))
        :effect (and (open ?c) (not (closed ?c)))
    )
""")
    
    # Close container action
    if "container" in all_objects:
        actions.append("""    (:action close-container
        :parameters (?r - robot ?c - container)
        :precondition (and (open ?c))
        :effect (and (closed ?c) (not (open ?c)))
    )
""")
    
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


def write_problem(problem_path: Path, episode_id: str, analysis: Dict[str, Any], episode_index: int) -> None:
    """Write PDDL problem file using Image2PDDL framework approach.
    
    Image2PDDL framework generates problems from:
    - Initial state descriptions
    - Goal state descriptions
    - VLM-extracted information
    """
    
    # Generate objects based on analysis
    num_blocks = max(2, min(3, episode_index + 2))
    block_objects = [f"block{i}" for i in range(1, num_blocks + 1)]
    
    containers = []
    if "container" in analysis.get("objects", []):
        containers = ["bowl1"]
    
    # Build objects section
    objects_list = " ".join(block_objects) + " - block"
    if containers:
        objects_list += "\n        " + " ".join(containers) + " - container"
    objects_list += "\n        table1 - surface\n        robot1 - robot"
    
    # Extract initial state from VLM description
    initial_state_desc = analysis.get("initial_state", "").lower()
    init = []
    
    # Infer initial state from description
    if "table" in initial_state_desc or "on table" in initial_state_desc:
        init.extend([f"(on-table {block})" for block in block_objects[:2]])
        init.extend([f"(clear {block})" for block in block_objects])
    
    if "container" in initial_state_desc or "bowl" in initial_state_desc or "cup" in initial_state_desc:
        if "open" in initial_state_desc:
            init.append("(open bowl1)")
            init.append("(empty bowl1)")
        else:
            init.append("(closed bowl1)")
            init.append("(empty bowl1)")
    
    # Default initial state if no description
    if not init:
        init = [
            "(on-table block1)",
            "(on-table block2)",
            "(clear block1)",
            "(clear block2)",
        ]
        if containers:
            init.append("(open bowl1)")
            init.append("(empty bowl1)")
    
    # Extract goal state from VLM description
    goal_state_desc = analysis.get("goal_state", "").lower()
    goal = ""
    
    # Infer goal from description
    if "stack" in goal_state_desc or "on top" in goal_state_desc:
        goal = "(on block1 block2)"
    elif "in" in goal_state_desc and ("container" in goal_state_desc or "bowl" in goal_state_desc or "cup" in goal_state_desc):
        if containers:
            goal = "(in block1 bowl1)"
        else:
            goal = "(on block1 block2)"
    elif "on table" in goal_state_desc:
        goal = "(on-table block1)"
    else:
        # Default goal
        goal = "(on block1 block2)"
    
    # Build comment section with Image2PDDL analysis
    comments = []
    comments.append("; Image2PDDL Framework Analysis:")
    if analysis.get("description"):
        comments.append(f"; Overall Description: {analysis['description'][:200]}...")
    if analysis.get("initial_state"):
        comments.append(f"; Initial State: {analysis['initial_state'][:150]}...")
    if analysis.get("goal_state"):
        comments.append(f"; Goal State: {analysis['goal_state'][:150]}...")
    if analysis.get("objects"):
        comments.append(f"; Detected objects: {analysis['objects']}")
    if analysis.get("actions"):
        comments.append(f"; Detected actions: {analysis['actions']}")
    
    comment_section = "\n    ".join(comments) if comments else "; No analysis available"
    
    # Clean episode ID for PDDL naming
    clean_id = episode_id.replace('+', '_').replace('-', '_')
    
    content = f"""(define (problem {clean_id})
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
        problem = reader.parse_problem(str(domain_path), str(problem_path))
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
    
    videos = list(mp4_dir.glob("*.mp4"))
    non_stereo = [v for v in videos if "-stereo" not in v.name]
    if non_stereo:
        return non_stereo[0]
    elif videos:
        return videos[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDDL files using Image2PDDL framework with Video-LLaVA or mPLUG-Owl"
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Directory containing episode folders (default: raw_videos)",
    )
    parser.add_argument(
        "--video-llava",
        action="store_true",
        help="Use Video-LLaVA for video understanding",
    )
    parser.add_argument(
        "--mplug-owl",
        action="store_true",
        help="Use mPLUG-Owl for video understanding",
    )
    parser.add_argument(
        "--video-llava-model",
        type=str,
        default="LanguageBind/Video-LLaVA-7B",
        help="Video-LLaVA model ID (default: LanguageBind/Video-LLaVA-7B)",
    )
    parser.add_argument(
        "--mplug-owl-model",
        type=str,
        default="MAGAer13/mplug-owl-llama-7b",
        help="mPLUG-Owl model ID (default: MAGAer13/mplug-owl-llama-7b)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Maximum frames to analyze per video (default: 8)",
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
    
    if not args.video_llava and not args.mplug_owl:
        print("❌ Please specify --video-llava or --mplug-owl")
        sys.exit(1)
    
    if not args.raw_videos_dir.exists():
        print(f"❌ Raw videos directory not found: {args.raw_videos_dir}")
        sys.exit(1)
    
    # Find all episode folders
    episode_folders = sorted([d for d in args.raw_videos_dir.iterdir() if d.is_dir()])
    
    if not episode_folders:
        print(f"❌ No episode folders found in {args.raw_videos_dir}")
        sys.exit(1)
    
    # Limit number of videos if specified
    if args.max_videos:
        episode_folders = episode_folders[:args.max_videos]
    
    print(f"📁 Found {len(episode_folders)} episode folders")
    print(f"🤖 Using Image2PDDL framework approach")
    print(f"📦 Model: {'Video-LLaVA' if args.video_llava else 'mPLUG-Owl'}")
    
    all_actions = set()
    all_objects = set()
    processed_videos = []
    
    # Process each video
    for i, episode_folder in enumerate(episode_folders):
        episode_id = episode_folder.name
        print(f"\n[{i+1}/{len(episode_folders)}] Processing: {episode_id}")
        
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
        
        # Analyze with Video-LLaVA or mPLUG-Owl
        print(f"   🤖 Analyzing with {'Video-LLaVA' if args.video_llava else 'mPLUG-Owl'} (Image2PDDL framework)...")
        try:
            if args.video_llava:
                analysis = analyze_video_with_video_llava(
                    frames_dir,
                    model_id=args.video_llava_model,
                    max_frames=args.max_frames,
                )
            else:
                analysis = analyze_video_with_mplug_owl(
                    frames_dir,
                    model_id=args.mplug_owl_model,
                    max_frames=args.max_frames,
                )
            
            all_actions.update(analysis.get("actions", []))
            all_objects.update(analysis.get("objects", []))
            print(f"   ✅ Analysis complete: {len(analysis.get('actions', []))} actions, {len(analysis.get('objects', []))} objects")
        except Exception as e:
            print(f"   ❌ Error analyzing video: {e}")
            import traceback
            traceback.print_exc()
            # Continue with empty analysis
            analysis = {"initial_state": "", "goal_state": "", "description": "", "objects": [], "actions": []}
        
        # Generate problem file using Image2PDDL approach
        clean_id = episode_id.replace('+', '_').replace('-', '_')
        problem_file = Path(f"problem_{clean_id}.pddl")
        write_problem(problem_file, episode_id, analysis, i)
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
    print(f"✅ Processing complete! (Image2PDDL Framework)")
    print(f"{'='*70}")
    print(f"📊 Processed: {len(processed_videos)} videos")
    print(f"📁 Domain file: {domain_path}")
    print(f"📄 Problem files: {len(processed_videos)}")
    print(f"🎯 Actions detected: {sorted(all_actions)}")
    print(f"🎯 Objects detected: {sorted(all_objects)}")


if __name__ == "__main__":
    main()

