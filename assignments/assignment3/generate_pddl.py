#!/usr/bin/env python3
"""
Generate PDDL domain and problem files from robot manipulation videos.

This script uses Visual Language Models (CLIP + BLIP) to analyze videos
and extract PDDL specifications as per assignment requirements.

Usage:
    python generate_pddl.py
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip


class PDDLGenerator:
    """Generate PDDL files from video frames using VLM models."""
    
    def __init__(self, raw_videos_dir: str = "raw_videos"):
        self.raw_videos_dir = Path(raw_videos_dir)
        self.frames_dir = Path("frames")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Initializing VLM models on {self.device}...")
        
        # Initialize BLIP for image captioning and scene understanding
        print("   Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
        # Initialize CLIP for object detection and understanding
        print("   Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        print("✅ Models loaded successfully!\n")
    
    def extract_frames_from_video(self, video_path: Path, output_dir: Path, fps: float = 0.5) -> int:
        """Extract frames from video using ffmpeg."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps={fps}",
            str(output_dir / "%04d.jpg"),
            "-loglevel", "error"
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        frame_count = len(list(output_dir.glob("*.jpg")))
        return frame_count
    
    def analyze_frame_with_blip(self, image_path: Path, question: str = None) -> str:
        """Analyze a single frame using BLIP."""
        image = Image.open(image_path).convert("RGB")
        
        if question:
            # Visual Question Answering
            inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
        else:
            # Image Captioning
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=100)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def analyze_objects_with_clip(self, image_path: Path) -> Dict[str, float]:
        """Detect objects in frame using CLIP."""
        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Object categories relevant to block manipulation
        text_prompts = [
            "a red block on a table",
            "a blue block on a table",
            "a green block on a table",
            "a yellow block on a table",
            "blocks stacked on each other",
            "a block in a bowl",
            "a block in a container",
            "an empty table",
            "a robot arm holding a block",
            "a robot arm moving",
        ]
        
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        results = {prompt: similarity[0, i].item() for i, prompt in enumerate(text_prompts)}
        return results
    
    def analyze_video(self, frames_dir: Path) -> Dict:
        """Analyze all frames of a video to extract PDDL information."""
        frames = sorted(frames_dir.glob("*.jpg"))
        
        if not frames:
            return {"objects": [], "actions": [], "initial_state": "", "goal_state": ""}
        
        print(f"      Analyzing {len(frames)} frames...")
        
        # Analyze first frame for initial state
        initial_caption = self.analyze_frame_with_blip(
            frames[0], 
            "What objects are on the table and where are they positioned?"
        )
        
        # Analyze middle frames for actions
        actions = []
        if len(frames) > 2:
            mid_idx = len(frames) // 2
            action_caption = self.analyze_frame_with_blip(
                frames[mid_idx],
                "What is the robot doing?"
            )
            actions.append(action_caption)
        
        # Analyze last frame for goal state
        goal_caption = self.analyze_frame_with_blip(
            frames[-1],
            "What is the final arrangement of objects on the table?"
        )
        
        # Use CLIP to detect objects across frames
        all_objects = set()
        for frame in [frames[0], frames[-1]]:
            clip_results = self.analyze_objects_with_clip(frame)
            # Add objects with high confidence (>0.15)
            for obj, score in clip_results.items():
                if score > 0.15:
                    all_objects.add(obj)
        
        return {
            "objects": list(all_objects),
            "actions": actions,
            "initial_state": initial_caption,
            "goal_state": goal_caption,
            "frames": len(frames)
        }
    
    def generate_domain_pddl(self, output_path: str = "domain.pddl"):
        """Generate the PDDL domain file."""
        domain_content = """(define (domain robot-manipulation)
    (:requirements :strips :typing)
    
    (:types
        block - object
        container - object
        location - object
        robot - agent
    )
    
    (:predicates
        (on ?obj - block ?target - object)
        (in ?obj - block ?container - container)
        (clear ?obj - object)
        (holding ?r - robot ?obj - block)
        (empty ?r - robot)
        (on-table ?obj - block)
    )
    
    (:action pick-up
        :parameters (?r - robot ?obj - block)
        :precondition (and 
            (empty ?r)
            (clear ?obj)
            (on-table ?obj)
        )
        :effect (and 
            (holding ?r ?obj)
            (not (empty ?r))
            (not (on-table ?obj))
            (not (clear ?obj))
        )
    )
    
    (:action put-down
        :parameters (?r - robot ?obj - block)
        :precondition (holding ?r ?obj)
        :effect (and 
            (empty ?r)
            (on-table ?obj)
            (clear ?obj)
            (not (holding ?r ?obj))
        )
    )
    
    (:action stack
        :parameters (?r - robot ?obj - block ?target - block)
        :precondition (and 
            (holding ?r ?obj)
            (clear ?target)
        )
        :effect (and 
            (empty ?r)
            (on ?obj ?target)
            (clear ?obj)
            (not (holding ?r ?obj))
            (not (clear ?target))
        )
    )
    
    (:action unstack
        :parameters (?r - robot ?obj - block ?target - block)
        :precondition (and 
            (empty ?r)
            (on ?obj ?target)
            (clear ?obj)
        )
        :effect (and 
            (holding ?r ?obj)
            (clear ?target)
            (not (empty ?r))
            (not (on ?obj ?target))
            (not (clear ?obj))
        )
    )
    
    (:action put-in-container
        :parameters (?r - robot ?obj - block ?container - container)
        :precondition (and 
            (holding ?r ?obj)
            (clear ?container)
        )
        :effect (and 
            (empty ?r)
            (in ?obj ?container)
            (not (holding ?r ?obj))
        )
    )
)
"""
        with open(output_path, 'w') as f:
            f.write(domain_content)
        print(f"✅ Generated {output_path}")
    
    def generate_problem_pddl(self, episode_id: str, analysis: Dict, output_path: str):
        """Generate a PDDL problem file for a specific video."""
        
        # Extract information from VLM analysis
        initial_desc = analysis.get("initial_state", "")
        goal_desc = analysis.get("goal_state", "")
        
        # Determine number of blocks based on descriptions
        num_blocks = 2  # Default
        if "three" in initial_desc.lower() or "3" in initial_desc:
            num_blocks = 3
        elif "four" in initial_desc.lower() or "4" in initial_desc:
            num_blocks = 4
        
        # Create block objects
        blocks = [f"block{i+1}" for i in range(num_blocks)]
        
        # Infer initial state from description
        init_predicates = ["(empty robot1)"]
        
        # Simple heuristic: assume blocks start on table
        for block in blocks:
            init_predicates.append(f"(on-table {block})")
            init_predicates.append(f"(clear {block})")
        
        # Check if stacking is mentioned in initial state
        if "stack" in initial_desc.lower() or "on top" in initial_desc.lower():
            # Modify for stacked configuration
            init_predicates = [
                "(empty robot1)",
                "(on-table block1)",
                "(on block2 block1)",
                "(clear block2)",
            ]
            if num_blocks > 2:
                init_predicates.append("(on-table block3)")
                init_predicates.append("(clear block3)")
        
        # Infer goal state from description
        goal_predicates = []
        
        if "stack" in goal_desc.lower() or "on top" in goal_desc.lower() or "pile" in goal_desc.lower():
            # Goal is to stack blocks
            if num_blocks == 2:
                goal_predicates.append("(on block2 block1)")
            elif num_blocks == 3:
                goal_predicates.extend([
                    "(on block2 block1)",
                    "(on block3 block2)"
                ])
        elif "bowl" in goal_desc.lower() or "container" in goal_desc.lower():
            # Goal involves container
            goal_predicates.append("(in block1 bowl1)")
        elif "separate" in goal_desc.lower() or "table" in goal_desc.lower():
            # Goal is blocks on table
            for block in blocks[:2]:
                goal_predicates.append(f"(on-table {block})")
        else:
            # Default goal: stack blocks
            goal_predicates.append("(on block2 block1)")
        
        # If no goal predicates inferred, use default
        if not goal_predicates:
            goal_predicates.append("(on block2 block1)")
        
        # Generate problem file content
        problem_name = episode_id.replace('+', '_').replace('-', '_')
        
        objects_section = "\n        ".join([f"{block} - block" for block in blocks])
        objects_section += "\n        robot1 - robot"
        
        # Check if container is needed
        if any("bowl" in p or "container" in p for p in goal_predicates):
            objects_section += "\n        bowl1 - container"
            if "(in block1 bowl1)" in goal_predicates:
                # Add container initial state
                init_predicates.append("(clear bowl1)")
        
        init_section = "\n        ".join(init_predicates)
        goal_section = "\n        ".join(goal_predicates)
        
        problem_content = f""";; Problem generated from video: {episode_id}
;; Initial state description: {initial_desc}
;; Goal state description: {goal_desc}

(define (problem {problem_name})
    (:domain robot-manipulation)
    
    (:objects
        {objects_section}
    )
    
    (:init
        {init_section}
    )
    
    (:goal
        (and
            {goal_section}
        )
    )
)
"""
        
        with open(output_path, 'w') as f:
            f.write(problem_content)
        
        print(f"      ✅ Generated {output_path}")
    
    def find_video_in_folder(self, folder_path: Path) -> Optional[Path]:
        """Find a video file in the episode folder."""
        mp4_dir = folder_path / "recordings" / "MP4"
        if not mp4_dir.exists():
            return None
        
        videos = list(mp4_dir.glob("*.mp4"))
        # Prefer non-stereo videos
        non_stereo = [v for v in videos if "-stereo" not in v.name]
        if non_stereo:
            return non_stereo[0]
        elif videos:
            return videos[0]
        return None
    
    def process_all_videos(self):
        """Process all videos and generate PDDL files."""
        # Find all episode folders
        episode_folders = sorted([d for d in self.raw_videos_dir.iterdir() if d.is_dir()])
        
        if not episode_folders:
            print(f"❌ No episode folders found in {self.raw_videos_dir}")
            return
        
        print(f"📁 Found {len(episode_folders)} episode folders\n")
        print("="*70)
        
        # Generate domain file first
        print("\n📝 Step 1: Generating domain.pddl...")
        self.generate_domain_pddl()
        
        print("\n📹 Step 2: Processing videos and generating problem files...\n")
        print("="*70)
        
        # Process each video
        for i, episode_folder in enumerate(episode_folders, 1):
            episode_id = episode_folder.name
            print(f"\n[{i}/{len(episode_folders)}] Processing: {episode_id}")
            
            # Find video file
            video_path = self.find_video_in_folder(episode_folder)
            if not video_path:
                print(f"   ⚠️  No video found, skipping...")
                continue
            
            print(f"   📹 Video: {video_path.name}")
            
            # Extract frames (if not already extracted)
            frames_output_dir = self.frames_dir / episode_id
            if not frames_output_dir.exists() or len(list(frames_output_dir.glob("*.jpg"))) == 0:
                print(f"   🎬 Extracting frames...")
                try:
                    frame_count = self.extract_frames_from_video(video_path, frames_output_dir, fps=0.5)
                    print(f"      ✅ Extracted {frame_count} frames")
                except Exception as e:
                    print(f"      ❌ Error extracting frames: {e}")
                    continue
            else:
                frame_count = len(list(frames_output_dir.glob("*.jpg")))
                print(f"   ✅ Using existing {frame_count} frames")
            
            # Analyze video with VLMs
            print(f"   🤖 Analyzing video with BLIP + CLIP...")
            try:
                analysis = self.analyze_video(frames_output_dir)
                print(f"      ✅ Analysis complete")
                print(f"      📊 Initial: {analysis['initial_state'][:60]}...")
                print(f"      📊 Goal: {analysis['goal_state'][:60]}...")
            except Exception as e:
                print(f"      ❌ Error analyzing video: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Generate problem file
            problem_filename = f"problem_{episode_id.replace('+', '_').replace('-', '_')}.pddl"
            print(f"   📄 Generating {problem_filename}...")
            try:
                self.generate_problem_pddl(episode_id, analysis, problem_filename)
            except Exception as e:
                print(f"      ❌ Error generating problem file: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*70)
        print("✅ PDDL generation complete!")
        print("="*70)
        print(f"📁 Generated files:")
        print(f"   - domain.pddl")
        problem_files = sorted(Path(".").glob("problem_*.pddl"))
        for pf in problem_files:
            print(f"   - {pf.name}")
        print("="*70)


def main():
    """Main entry point."""
    print("="*70)
    print("PDDL Generation from Robot Manipulation Videos")
    print("Using BLIP + CLIP Visual Language Models")
    print("="*70)
    
    generator = PDDLGenerator(raw_videos_dir="raw_videos")
    generator.process_all_videos()
    
    print("\n✅ Done! You can now use these files with a PDDL planner.")


if __name__ == "__main__":
    main()


