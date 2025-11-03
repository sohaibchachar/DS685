"""Generate PDDL domain and problem files from DROID videos using CLIP or BLIP.

This script processes videos from raw_videos folder and generates:
- A unified PDDL domain file (domain.pddl) following Unified Planning standards
- Individual problem files for each video (problem_<episode_id>.pddl)

Uses CLIP or BLIP for visual understanding, following NVIDIA prompt engineering best practices.

Usage:
    python generate_pddl_from_videos.py --raw-videos-dir raw_videos --openclip
    python generate_pddl_from_videos.py --raw-videos-dir raw_videos --blip
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def extract_frames(video_path: Path, frames_dir: Path, fps: float = 0.5) -> None:
    """Extract frames from video at specified FPS."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(frames_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def analyze_frames_with_openclip(frames_dir: Path, max_frames: int = 10) -> Dict[str, Any]:
    """Analyze frames using OpenCLIP for object and action detection."""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "OpenCLIP dependencies missing. Install: pip install open-clip-torch torch pillow"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Define prompts for robot manipulation tasks
    prompts = [
        "What is the robot doing with what color of block if there are multiple blocks"
    ]

    # Tokenize prompts
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Analyze frames
    jpgs = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return {"detections": [], "objects": [], "actions": [], "frames": 0}

    detections = []
    detected_objects = set()
    detected_actions = set()

    for img_path in jpgs:
        try:
            image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.T
                probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
            
            # Get top 3 detections
            top_indices = probs.argsort()[-3:][::-1]
            frame_detections = []
            for idx in top_indices:
                if probs[idx] > 0.1:  # Threshold
                    prompt = prompts[idx]
                    frame_detections.append((prompt, float(probs[idx])))
                    
                    # Extract objects and actions
                    prompt_lower = prompt.lower()
                    if "block" in prompt_lower:
                        detected_objects.add("block")
                    if "container" in prompt_lower or "bowl" in prompt_lower or "cup" in prompt_lower:
                        detected_objects.add("container")
                    if "table" in prompt_lower:
                        detected_objects.add("table")
                    if "pick" in prompt_lower or "picking" in prompt_lower:
                        detected_actions.add("pick")
                    if "place" in prompt_lower or "placing" in prompt_lower:
                        detected_actions.add("place")
                    if "stack" in prompt_lower or "stacking" in prompt_lower:
                        detected_actions.add("stack")
                    if "in" in prompt_lower and ("container" in prompt_lower or "bowl" in prompt_lower or "cup" in prompt_lower):
                        detected_actions.add("put-in")
            
            if frame_detections:
                detections.append((img_path.name, frame_detections))
        except Exception as e:
            print(f"   ⚠️  Warning: Error processing {img_path.name}: {e}")
            continue

    return {
        "detections": detections,
        "objects": sorted(list(detected_objects)),
        "actions": sorted(list(detected_actions)),
        "frames": len(jpgs),
    }


def analyze_frames_with_blip(frames_dir: Path, max_frames: int = 10) -> Dict[str, Any]:
    """Analyze frames using BLIP for captioning and question-answering."""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
        import torch
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "BLIP dependencies missing. Install: pip install transformers torch pillow"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load BLIP models
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = caption_model.to(device)
    caption_model.eval()
    
    # Also use BLIP for question-answering if available
    try:
        qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        qa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        qa_model = qa_model.to(device)
        qa_model.eval()
        use_qa = True
    except Exception:
        use_qa = False
        qa_processor = None
        qa_model = None

    jpgs = sorted(frames_dir.glob("*.jpg"))[:max_frames]
    if not jpgs:
        return {"captions": [], "objects": [], "actions": [], "frames": 0}

    captions = []
    detected_objects = set()
    detected_actions = set()
    
    # Questions for better object/action detection
    questions = [
        "What objects are in the image?",
        "Is there a block or cube?",
        "Is there a container or bowl?",
        "Is there a table or surface?",
        "What is the robot doing?",
        "Is the robot picking up something?",
        "Is the robot placing something?",
        "Is the robot stacking blocks?",
        "Is the robot putting something in a bowl?",
        "What different colored blocks are present?",
        "Are there any actions involving different colored blocks?",
    ]

    for img_path in jpgs:
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Generate caption
            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = caption_model.generate(**inputs, max_length=100)
            caption = processor.decode(output[0], skip_special_tokens=True)
            captions.append((img_path.name, caption))
            
            # Question-answering for better extraction
            if use_qa and qa_model:
                try:
                    for question in questions[:4]:  # Use first 4 questions
                        qa_inputs = qa_processor(image, question, return_tensors="pt").to(device)
                        with torch.no_grad():
                            qa_output = qa_model.generate(**qa_inputs, max_length=20)
                        answer = qa_processor.decode(qa_output[0], skip_special_tokens=True)
                        
                        # Combine caption and answers for analysis
                        caption = caption + " " + answer.lower()
                except Exception:
                    pass  # Continue with caption only
            
            # Extract objects and actions from caption
            caption_lower = caption.lower()
            
            # Object detection
            if any(word in caption_lower for word in ["block", "cube", "brick", "box", "object"]):
                detected_objects.add("block")
            if any(word in caption_lower for word in ["container", "bowl", "cup", "jar", "pot", "vessel"]):
                detected_objects.add("container")
            if any(word in caption_lower for word in ["table", "surface", "desk", "platform", "counter"]):
                detected_objects.add("table")
            
            # Action detection
            if any(word in caption_lower for word in ["pick", "pick up", "picking", "grasp", "grasping", "grab", "grabbing", "lift", "lifting", "holding"]):
                detected_actions.add("pick")
            if any(word in caption_lower for word in ["place", "placing", "put", "putting", "set", "setting", "drop", "dropping", "deposit"]):
                detected_actions.add("place")
            if any(word in caption_lower for word in ["stack", "stacking", "stacked", "pile", "piling", "on top"]):
                detected_actions.add("stack")
            if any(word in caption_lower for word in ["put in", "putting in", "place in", "placing in", "inside", "into", "contain"]):
                detected_actions.add("put-in")
            if any(word in caption_lower for word in ["open", "opening", "uncover", "uncovering", "unseal"]):
                detected_actions.add("open")
            if any(word in caption_lower for word in ["close", "closing", "cover", "covering", "shut", "shutting", "seal"]):
                detected_actions.add("close")
        except Exception as e:
            print(f"   ⚠️  Warning: Error processing {img_path.name}: {e}")
            continue

    return {
        "captions": captions,
        "objects": sorted(list(detected_objects)),
        "actions": sorted(list(detected_actions)),
        "frames": len(jpgs),
    }


def write_domain(domain_path: Path, all_actions: Set[str], all_objects: Set[str]) -> None:
    """Write PDDL domain file following Unified Planning standards."""
    
    # Types - standard blocks world types (Unified Planning format)
    types = """        block - object
        container - object
        surface - object
        robot - agent"""
    
    # Predicates - comprehensive manipulation predicates
    predicates = """        (holding ?r - robot ?o - block)
        (on ?o1 - block ?o2 - object)
        (clear ?o - block)
        (in ?o - block ?c - container)
        (on-table ?o - block)
        (empty ?c - container)
        (open ?c - container)
        (closed ?c - container)"""
    
    # Build actions based on detected actions
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
    """Write PDDL problem file following Unified Planning standards."""
    
    # Generate problem-specific objects
    num_blocks = max(2, min(3, episode_index + 2))  # Vary number of blocks per problem
    block_objects = [f"block{i}" for i in range(1, num_blocks + 1)]
    
    containers = []
    if "container" in analysis.get("objects", []):
        containers = ["bowl1"]
    
    # Build objects section
    objects_list = " ".join(block_objects) + " - block"
    if containers:
        objects_list += "\n        " + " ".join(containers) + " - container"
    objects_list += "\n        table1 - surface\n        robot1 - robot"
    
    # Initial state - vary per problem
    init = []
    if episode_index == 0:
        # Problem 1: All blocks on table
        init.extend([f"(on-table {block})" for block in block_objects])
        init.extend([f"(clear {block})" for block in block_objects])
    elif episode_index == 1:
        # Problem 2: One block on table, one clear
        init.append("(on-table block1)")
        init.append("(on-table block2)")
        init.append("(clear block1)")
        init.append("(clear block2)")
        if containers:
            init.append("(open bowl1)")
            init.append("(empty bowl1)")
    elif episode_index == 2:
        # Problem 3: Blocks stacked or in container
        init.append("(on-table block1)")
        init.append("(on-table block2)")
        init.append("(clear block1)")
        init.append("(clear block2)")
        if containers:
            init.append("(closed bowl1)")
            init.append("(empty bowl1)")
    else:
        # Default initial state
        init.extend([f"(on-table {block})" for block in block_objects])
        init.extend([f"(clear {block})" for block in block_objects])
        if containers:
            init.append("(open bowl1)")
            init.append("(empty bowl1)")
    
    # Goal state - vary per problem
    if episode_index == 0:
        # Goal: Stack blocks
        goal = "(on block1 block2)"
    elif episode_index == 1:
        # Goal: Put block in container
        if containers:
            goal = "(in block1 bowl1)"
        else:
            goal = "(on block1 block2)"
    elif episode_index == 2:
        # Goal: Stack or place on table
        goal = "(on block1 block2)"
    else:
        # Default goal
        goal = "(on block1 block2)"
    
    # Build comment section with analysis
    comments = []
    if analysis.get("detections"):
        comments.append("; CLIP Analysis:")
        for frame, detections in analysis["detections"][:3]:
            comments.append(f";   {frame}: {detections}")
    elif analysis.get("captions"):
        comments.append("; BLIP Captions:")
        for frame, caption in analysis["captions"][:3]:
            comments.append(f";   {frame}: {caption}")
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
        description="Generate PDDL files from DROID videos using CLIP or BLIP"
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=Path("raw_videos"),
        help="Directory containing episode folders (default: raw_videos)",
    )
    parser.add_argument(
        "--openclip",
        action="store_true",
        help="Use OpenCLIP for visual analysis",
    )
    parser.add_argument(
        "--blip",
        action="store_true",
        help="Use BLIP for visual analysis",
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
        "--validate",
        action="store_true",
        help="Validate PDDL files using Unified Planning",
    )
    
    args = parser.parse_args()
    
    if not args.openclip and not args.blip:
        print("❌ Please specify --openclip or --blip")
        sys.exit(1)
    
    if not args.raw_videos_dir.exists():
        print(f"❌ Raw videos directory not found: {args.raw_videos_dir}")
        sys.exit(1)
    
    # Find all episode folders
    episode_folders = sorted([d for d in args.raw_videos_dir.iterdir() if d.is_dir()])
    
    if not episode_folders:
        print(f"❌ No episode folders found in {args.raw_videos_dir}")
        sys.exit(1)
    
    print(f"📁 Found {len(episode_folders)} episode folders")
    
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
        
        # Analyze with CLIP or BLIP
        print(f"   🤖 Analyzing with {'OpenCLIP' if args.openclip else 'BLIP'}...")
        try:
            if args.openclip:
                analysis = analyze_frames_with_openclip(frames_dir, max_frames=args.max_frames)
            else:
                analysis = analyze_frames_with_blip(frames_dir, max_frames=args.max_frames)
            
            all_actions.update(analysis.get("actions", []))
            all_objects.update(analysis.get("objects", []))
            print(f"   ✅ Analysis complete: {len(analysis.get('actions', []))} actions, {len(analysis.get('objects', []))} objects")
        except Exception as e:
            print(f"   ❌ Error analyzing video: {e}")
            # Continue with empty analysis
            analysis = {"objects": [], "actions": [], "detections": [], "captions": []}
        
        # Generate problem file
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
    print(f"✅ Processing complete!")
    print(f"{'='*70}")
    print(f"📊 Processed: {len(processed_videos)} videos")
    print(f"📁 Domain file: {domain_path}")
    print(f"📄 Problem files: {len(processed_videos)}")
    print(f"🎯 Actions detected: {sorted(all_actions)}")
    print(f"🎯 Objects detected: {sorted(all_objects)}")


if __name__ == "__main__":
    main()

