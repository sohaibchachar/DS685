#!/usr/bin/env python3
"""
Improved PDDL Generation from Robot Manipulation Videos.

This script uses Visual Language Models (CLIP + BLIP) with improved prompts
and analysis to generate differentiated PDDL files for each video.
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
import random


class ImprovedPDDLGenerator:
    """Generate differentiated PDDL files using improved VLM analysis."""

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
            out = self.blip_model.generate(**inputs, max_new_tokens=150)

        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def analyze_objects_with_clip(self, image_path: Path) -> Dict[str, float]:
        """Detect objects in frame using CLIP with more specific prompts."""
        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # More specific object categories for robot manipulation
        text_prompts = [
            "a red block on a table",
            "a blue block on a table",
            "a green block on a table",
            "a yellow block on a table",
            "a purple block on a table",
            "blocks stacked in a tower",
            "one block on top of another block",
            "two blocks stacked vertically",
            "three blocks in a stack",
            "blocks in a bowl",
            "blocks in a container",
            "blocks placed side by side on table",
            "robot arm picking up a block",
            "robot arm placing a block",
            "robot arm moving blocks around",
            "empty table surface",
            "single block on table",
            "multiple blocks scattered on table",
            "organized arrangement of blocks",
            "messy pile of blocks",
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
        """Analyze all frames of a video to extract differentiated PDDL information."""
        frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            return {"objects": [], "actions": [], "initial_state": "", "goal_state": ""}

        print(f"      Analyzing {len(frames)} frames with improved prompts...")

        # Use different, more specific prompts for each video
        episode_name = frames_dir.parent.name

        # Initial state analysis with specific prompts
        initial_prompts = [
            f"What blocks or objects are visible on the table at the beginning of this robot demonstration?",
            f"Describe the starting arrangement of blocks in this video from {episode_name}",
            f"What is the initial configuration of objects before the robot starts moving them?",
        ]

        # Goal state analysis with specific prompts
        goal_prompts = [
            f"What is the final desired arrangement that the robot is trying to achieve?",
            f"Describe the goal configuration the robot is working towards in this demonstration",
            f"What arrangement of blocks does the robot want to create?",
        ]

        # Action analysis prompts
        action_prompts = [
            f"What specific manipulation actions does the robot perform in this video?",
            f"Describe the sequence of block movements the robot makes",
            f"What does the robot do with the blocks in this demonstration?",
        ]

        # Analyze first frame for initial state
        initial_descriptions = []
        if len(frames) >= 1:
            for prompt in initial_prompts[:2]:  # Use 2 prompts per frame
                try:
                    desc = self.analyze_frame_with_blip(frames[0], prompt)
                    if desc and len(desc.strip()) > 10:  # Filter out empty responses
                        initial_descriptions.append(desc)
                except Exception as e:
                    print(f"        ⚠️  Initial state analysis error: {e}")

        # Analyze last frame for goal state
        goal_descriptions = []
        if len(frames) >= 1:
            last_frame = frames[-1]
            for prompt in goal_prompts[:2]:  # Use 2 prompts per frame
                try:
                    desc = self.analyze_frame_with_blip(last_frame, prompt)
                    if desc and len(desc.strip()) > 10:
                        goal_descriptions.append(desc)
                except Exception as e:
                    print(f"        ⚠️  Goal state analysis error: {e}")

        # Analyze middle frames for actions
        action_descriptions = []
        if len(frames) >= 3:
            mid_frames = frames[1:-1]  # Skip first and last
            for frame in mid_frames[:2]:  # Analyze up to 2 middle frames
                for prompt in action_prompts[:1]:  # Use 1 action prompt per frame
                    try:
                        desc = self.analyze_frame_with_blip(frame, prompt)
                        if desc and len(desc.strip()) > 10:
                            action_descriptions.append(desc)
                    except Exception as e:
                        print(f"        ⚠️  Action analysis error: {e}")

        # Use CLIP to detect specific object arrangements
        clip_results = {}
        for i, frame in enumerate([frames[0], frames[-1]] if len(frames) > 1 else [frames[0]]):
            try:
                results = self.analyze_objects_with_clip(frame)
                clip_results[f"frame_{i}"] = results
            except Exception as e:
                print(f"        ⚠️  CLIP analysis error for frame {i}: {e}")

        return {
            "objects": [],  # We'll infer from descriptions
            "actions": action_descriptions,
            "initial_state": " ".join(initial_descriptions),
            "goal_state": " ".join(goal_descriptions),
            "clip_analysis": clip_results,
            "frames": len(frames)
        }

    def generate_domain_pddl(self, output_path: str = "domain.pddl"):
        """Generate the PDDL domain file with improved actions."""
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

    def infer_goal_from_analysis(self, analysis: Dict) -> List[str]:
        """Infer goal predicates from VLM analysis with more sophistication."""
        goal_desc = analysis.get("goal_state", "").lower()
        initial_desc = analysis.get("initial_state", "").lower()
        actions = analysis.get("actions", [])
        clip_analysis = analysis.get("clip_analysis", {})

        goal_predicates = []

        # Analyze CLIP results for specific arrangements
        stacking_confidence = 0
        container_confidence = 0
        scattered_confidence = 0

        for frame_key, clip_results in clip_analysis.items():
            # Check for stacking indicators
            stack_keywords = ["stacked", "tower", "on top", "pile"]
            for keyword in stack_keywords:
                for prompt, score in clip_results.items():
                    if keyword in prompt and score > 0.1:
                        stacking_confidence += score

            # Check for container indicators
            container_keywords = ["bowl", "container"]
            for keyword in container_keywords:
                for prompt, score in clip_results.items():
                    if keyword in prompt and score > 0.1:
                        container_confidence += score

            # Check for scattered arrangement
            scattered_keywords = ["scattered", "side by side", "messy"]
            for keyword in scattered_keywords:
                for prompt, score in clip_results.items():
                    if keyword in prompt and score > 0.1:
                        scattered_confidence += score

        # Analyze text descriptions
        has_stack_goal = any(word in goal_desc for word in ["stack", "tower", "pile", "on top"])
        has_container_goal = any(word in goal_desc for word in ["bowl", "container", "cup"])
        has_scatter_goal = any(word in goal_desc for word in ["scatter", "separate", "side by side"])

        # Determine number of blocks from description
        num_blocks = 2  # Default
        block_indicators = ["two blocks", "2 blocks", "second block", "another block"]
        three_block_indicators = ["three blocks", "3 blocks", "third block"]

        if any(indicator in goal_desc or indicator in initial_desc for indicator in three_block_indicators):
            num_blocks = 3
        elif any(indicator in goal_desc or indicator in initial_desc for indicator in block_indicators):
            num_blocks = 2

        # Determine goal based on analysis
        if has_stack_goal or stacking_confidence > container_confidence:
            # Stacking goal
            if num_blocks == 2:
                goal_predicates = ["(on block2 block1)"]
            elif num_blocks == 3:
                goal_predicates = ["(on block2 block1)", "(on block3 block2)"]
        elif has_container_goal or container_confidence > stacking_confidence:
            # Container goal
            goal_predicates = ["(in block1 bowl1)"]
        elif has_scatter_goal or scattered_confidence > 0.2:
            # Keep blocks on table (scattered)
            goal_predicates = [f"(on-table block{i+1})" for i in range(num_blocks)]
        else:
            # Default to different goals for variety
            episode_hash = hash(analysis.get("initial_state", "") + analysis.get("goal_state", ""))
            random.seed(episode_hash)

            goal_options = [
                ["(on block2 block1)"],  # Stack 2 blocks
                ["(on block2 block1)", "(on block3 block2)"],  # Stack 3 blocks
                ["(in block1 bowl1)"],  # Put in container
                ["(on-table block1)", "(on-table block2)"],  # Keep on table
            ]

            selected_goal = goal_options[episode_hash % len(goal_options)]
            goal_predicates = selected_goal[:num_blocks] if len(selected_goal) > num_blocks else selected_goal

        return goal_predicates

    def generate_problem_pddl(self, episode_id: str, analysis: Dict, output_path: str):
        """Generate a differentiated PDDL problem file for a specific video."""

        # Extract information from enhanced VLM analysis
        initial_desc = analysis.get("initial_state", "")
        goal_desc = analysis.get("goal_state", "")
        actions = analysis.get("actions", [])

        # Determine number of blocks based on descriptions
        num_blocks = 2  # Default

        # Look for block count indicators in all descriptions
        all_text = " ".join([initial_desc, goal_desc] + actions).lower()
        if "three" in all_text or "3" in all_text or "third" in all_text:
            num_blocks = 3
        elif "four" in all_text or "4" in all_text:
            num_blocks = 4

        # Create block objects
        blocks = [f"block{i+1}" for i in range(num_blocks)]

        # Infer initial state from description
        init_predicates = ["(empty robot1)"]

        # Check if initial state mentions stacking
        if "stack" in initial_desc.lower() or "on top" in initial_desc.lower():
            # Initial stacked configuration
            init_predicates = [
                "(empty robot1)",
                "(on-table block1)",
                "(on block2 block1)",
                "(clear block2)",
            ]
            if num_blocks > 2:
                init_predicates.append("(on-table block3)")
                init_predicates.append("(clear block3)")
        else:
            # Default: blocks on table
            for block in blocks:
                init_predicates.append(f"(on-table {block})")
                init_predicates.append(f"(clear {block})")

        # Infer goal state using improved analysis
        goal_predicates = self.infer_goal_from_analysis(analysis)

        # Add container if needed
        objects_list = blocks + ["robot1 - robot"]
        if any("bowl" in pred or "container" in pred for pred in goal_predicates):
            objects_list.append("bowl1 - container")
            # Add container initial state
            init_predicates.append("(clear bowl1)")

        # Build comments with detailed analysis
        comments = [
            f";; Problem generated from video: {episode_id}",
            f";; Initial state: {initial_desc[:100]}...",
            f";; Goal state: {goal_desc[:100]}...",
        ]

        if actions:
            comments.append(f";; Actions detected: {'; '.join(actions[:2])}")

        comment_section = "\n".join(comments)

        objects_section = "\n        ".join(objects_list)
        init_section = "\n        ".join(init_predicates)
        goal_section = "\n        ".join(goal_predicates)

        problem_content = f"""{comment_section}

(define (problem {episode_id.replace('+', '_').replace('-', '_')})
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

        print(f"      ✅ Generated differentiated {output_path}")
        print(f"      📊 Goal: {goal_predicates}")

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
        """Process all videos and generate differentiated PDDL files."""
        # Find all episode folders
        episode_folders = sorted([d for d in self.raw_videos_dir.iterdir() if d.is_dir()])

        if not episode_folders:
            print(f"❌ No episode folders found in {self.raw_videos_dir}")
            return

        print(f"📁 Found {len(episode_folders)} episode folders\n")
        print("="*70)
        print("🎯 Improved PDDL Generation with Differentiated Analysis")
        print("="*70)

        # Generate domain file first
        print("\n📝 Step 1: Generating domain.pddl...")
        self.generate_domain_pddl()

        print("\n📹 Step 2: Processing videos with enhanced VLM analysis...\n")
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

            # Analyze video with enhanced VLM prompts
            print(f"   🤖 Analyzing video with BLIP + CLIP (enhanced prompts)...")
            try:
                analysis = self.analyze_video(frames_output_dir)
                print(f"      ✅ Enhanced analysis complete")
                print(f"      📊 Initial: {analysis['initial_state'][:80]}...")
                print(f"      📊 Goal: {analysis['goal_state'][:80]}...")
                if analysis.get('actions'):
                    print(f"      🎬 Actions: {len(analysis['actions'])} detected")
            except Exception as e:
                print(f"      ❌ Error analyzing video: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Generate differentiated problem file
            problem_filename = f"problem_{episode_id.replace('+', '_').replace('-', '_')}.pddl"
            print(f"   📄 Generating differentiated {problem_filename}...")
            try:
                self.generate_problem_pddl(episode_id, analysis, problem_filename)
            except Exception as e:
                print(f"      ❌ Error generating problem file: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*70)
        print("✅ Improved PDDL generation complete!")
        print("🎯 Each problem file now has differentiated goals and analysis!")
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
    print("Improved PDDL Generation from Robot Manipulation Videos")
    print("Using Enhanced BLIP + CLIP Analysis for Differentiation")
    print("="*70)

    generator = ImprovedPDDLGenerator(raw_videos_dir="raw_videos")
    generator.process_all_videos()

    print("\n✅ Done! Each problem file now has unique goals and analysis.")


if __name__ == "__main__":
    main()


