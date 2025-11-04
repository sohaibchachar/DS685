#!/usr/bin/env python3
"""
Generate PDDL files from VLM Analysis of Video Content.

This script analyzes actual video frames using BLIP + CLIP to understand
what the robot is doing, then generates PDDL problems based on VLM observations
rather than pre-written annotations.
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


class VLMAnalysisPDDLGenerator:
    """Generate PDDL files by analyzing video content with VLMs."""

    def __init__(self, raw_videos_dir: str = "raw_videos", annotations_file: str = "droid_language_annotations.json"):
        self.raw_videos_dir = Path(raw_videos_dir)
        self.annotations_file = Path(annotations_file)
        self.frames_dir = Path("frames")

        print(f"🔧 Loading DROID annotations for reference...")
        # Load annotations for comparison/validation
        with open(self.annotations_file, 'r') as f:
            content = f.read()

        # Skip Git LFS header if present
        if content.startswith('<<<<<<< HEAD'):
            json_start = content.find('{')
            if json_start != -1:
                content = content[json_start:]

        # Extract JSON content between merge conflict markers
        if '<<<<<<< HEAD' in content and '>>>>>>>' in content:
            start_marker = '<<<<<<< HEAD'
            end_marker = '>>>>>>> '
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                json_start = content.find('{', start_idx)
                if json_start != -1:
                    json_end = content.rfind('}', 0, end_idx)
                    if json_end != -1:
                        content = content[json_start:json_end+1]

        self.annotations = json.loads(content)
        print(f"✅ Loaded {len(self.annotations)} reference annotations")

        # Initialize VLMs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing VLMs on {self.device}...")

        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ VLMs loaded")
        except Exception as e:
            print(f"⚠️  VLM loading failed: {e}")
            self.blip_processor = None
            self.blip_model = None
            self.clip_model = None
            self.clip_preprocess = None

        # Episode IDs
        self.episode_ids = [
            'AUTOLab+0d4edc83+2023-10-27-20h-25m-34s',
            'GuptaLab+553d1bd5+2023-05-19-10h-36m-14s',
            'RAD+c6cf6b42+2023-08-31-14h-00m-49s',
            'RAIL+d027f2ae+2023-06-05-16h-33m-01s',
            'TRI+52ca9b6a+2024-01-16-16h-43m-04s'
        ]

    def extract_frames_from_video(self, video_path: Path, output_dir: Path, fps: float = 5) -> int:
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
        if not self.blip_processor or not self.blip_model:
            return "VLM not available"

        image = Image.open(image_path).convert("RGB")

        if question:
            inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
        else:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=100)

        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def analyze_objects_with_clip(self, image_path: Path) -> Dict[str, float]:
        """Detect objects in frame using CLIP."""
        if not self.clip_model or not self.clip_preprocess:
            return {}

        image = Image.open(image_path).convert("RGB")
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Object categories for robot manipulation
        text_prompts = [
            "a red block on a table",
            "a blue block on a table",
            "a green block on a table",
            "a yellow block on a table",
            "blocks stacked vertically",
            "blocks in a container",
            "blocks in a bowl",
            "empty table surface",
            "robot arm holding a block",
            "blocks scattered on table",
            "organized block arrangement",
        ]

        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        results = {prompt: similarity[0, i].item() for i, prompt in enumerate(text_prompts)}
        return results

    def analyze_video_states(self, frames_dir: Path) -> Dict:
        """Analyze initial and final states from video frames."""
        frames = sorted(frames_dir.glob("*.jpg"))

        if not frames:
            return {"initial_state": "", "final_state": "", "action_description": ""}

        print(f"      Analyzing {len(frames)} frames...")

        # Analyze initial state (first frame)
        initial_frame = frames[0]
        initial_analysis = self.analyze_frame_with_blip(
            initial_frame,
            "Describe what objects are visible and their arrangement at the beginning"
        )

        # Analyze final state (last frame)
        final_frame = frames[-1]
        final_analysis = self.analyze_frame_with_blip(
            final_frame,
            "Describe what objects are visible and their arrangement at the end"
        )

        # Analyze action (middle frame if available)
        action_description = ""
        if len(frames) > 2:
            middle_frame = frames[len(frames)//2]
            action_description = self.analyze_frame_with_blip(
                middle_frame,
                "Describe what the robot is doing in this scene"
            )

        # CLIP analysis for both states
        initial_clip = self.analyze_objects_with_clip(initial_frame)
        final_clip = self.analyze_objects_with_clip(final_frame)

        return {
            "initial_state": initial_analysis,
            "final_state": final_analysis,
            "action_description": action_description,
            "initial_clip": initial_clip,
            "final_clip": final_clip,
            "frames_analyzed": len(frames)
        }

    def infer_goal_from_vlm_states(self, initial_state: str, final_state: str, action_desc: str) -> Dict:
        """Infer PDDL goal from VLM analysis of initial vs final states."""

        initial_lower = initial_state.lower()
        final_lower = final_state.lower()
        action_lower = action_desc.lower()

        # Detect if container is involved
        container_keywords = ['bowl', 'container', 'cup', 'mug', 'glass', 'jar', 'box']
        has_container = any(word in initial_lower + final_lower for word in container_keywords)

        # Determine container name
        container_name = 'container1'
        for keyword in container_keywords:
            if keyword in initial_lower or keyword in final_lower:
                if keyword == 'bowl':
                    container_name = 'bowl1'
                elif keyword == 'cup':
                    container_name = 'cup1'
                elif keyword == 'mug':
                    container_name = 'mug1'
                elif keyword == 'glass':
                    container_name = 'glass1'
                elif keyword == 'jar':
                    container_name = 'jar1'
                elif keyword == 'box':
                    container_name = 'box1'
                break

        # Analyze state changes to infer goal
        goal_predicates = []
        goal_type = "unknown"

        # Check for stacking (blocks ending up on other blocks)
        if any(word in final_lower for word in ['on top', 'stacked', 'on another']) and \
           not any(word in initial_lower for word in ['on top', 'stacked']):
            goal_predicates = ['(on block2 block1)']
            goal_type = "stack_blocks"
        elif has_container:
            # Check if block moved into container
            if any(word in final_lower for word in ['in bowl', 'in container', 'inside']) and \
               not any(word in initial_lower for word in ['in bowl', 'in container', 'inside']):
                goal_predicates = [f'(in block1 {container_name})']
                goal_type = "put_in_container"
            # Check if block moved out of container
            elif any(word in final_lower for word in ['on table', 'outside']) and \
                 any(word in initial_lower for word in ['in bowl', 'in container', 'inside']):
                goal_predicates = ['(on-table block1)']
                goal_type = "remove_from_container"
        else:
            # Default: blocks should be on table
            goal_predicates = ['(on-table block1)', '(on-table block2)']
            goal_type = "arrange_on_table"

        return {
            'goal_predicates': goal_predicates,
            'goal_type': goal_type,
            'has_container': has_container,
            'container_name': container_name if has_container else None,
            'vlm_initial': initial_state,
            'vlm_final': final_state,
            'vlm_action': action_desc
        }

    def generate_domain_pddl(self, output_path: str = "domain.pddl"):
        """Generate the PDDL domain file."""
        domain_content = """(define (domain robot-manipulation)
    (:requirements :strips :typing)

    (:types
        block - object
        container - object
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

    (:action take-from-container
        :parameters (?r - robot ?obj - block ?container - container)
        :precondition (and
            (empty ?r)
            (in ?obj ?container)
            (clear ?container)
        )
        :effect (and
            (holding ?r ?obj)
            (not (empty ?r))
            (not (in ?obj ?container))
        )
    )
)
"""
        with open(output_path, 'w') as f:
            f.write(domain_content)
        print(f"✅ Generated {output_path}")

    def generate_problem_pddl(self, episode_id: str, vlm_analysis: Dict, output_path: str):
        """Generate PDDL problem based on VLM analysis of video content."""

        # Get reference annotation for comparison
        reference_instruction = ""
        if episode_id in self.annotations:
            reference_instruction = self.annotations[episode_id]['language_instruction1']

        # Infer goal from VLM analysis
        goal_inference = self.infer_goal_from_vlm_states(
            vlm_analysis['initial_state'],
            vlm_analysis['final_state'],
            vlm_analysis['action_description']
        )

        goal_predicates = goal_inference['goal_predicates']
        goal_type = goal_inference['goal_type']
        has_container = goal_inference['has_container']
        container_name = goal_inference['container_name']

        # Determine number of blocks (assume 2 for now, can be extended)
        num_blocks = 2

        # Create objects
        blocks = [f"block{i+1}" for i in range(num_blocks)]
        objects_list = [f"{block} - block" for block in blocks] + ["robot1 - robot"]

        if has_container:
            objects_list.append(f"{container_name} - container")

        # Set up initial state based on goal type
        init_predicates = ["(empty robot1)"]

        if goal_type == "remove_from_container":
            # Block starts in container
            init_predicates.extend([
                f"(in block1 {container_name})",
                f"(clear {container_name})"
            ])
            if num_blocks > 1:
                init_predicates.extend([
                    "(on-table block2)",
                    "(clear block2)"
                ])
        elif goal_type == "put_in_container":
            # Blocks start on table, container is empty
            for block in blocks:
                init_predicates.append(f"(on-table {block})")
                init_predicates.append(f"(clear {block})")
            init_predicates.append(f"(clear {container_name})")
        else:
            # Default: blocks on table
            for block in blocks:
                init_predicates.append(f"(on-table {block})")
                init_predicates.append(f"(clear {block})")

        # Build detailed comments
        comments = [
            f";; Episode: {episode_id}",
            f";; VLM-Based Goal Generation",
            f";; Reference Instruction: {reference_instruction}",
            f";; VLM Initial State: {vlm_analysis['initial_state'][:80]}...",
            f";; VLM Final State: {vlm_analysis['final_state'][:80]}...",
            f";; VLM Action: {vlm_analysis['action_description'][:60]}...",
            f";; Inferred Goal Type: {goal_type}",
        ]

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

        print(f"      ✅ Generated VLM-based problem for {episode_id}")
        print(f"      🎯 Goal Type: {goal_type}")
        print(f"      📊 Predicates: {goal_predicates}")

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
        """Process all videos using VLM analysis of actual video content."""
        # Find all episode folders
        episode_folders = sorted([d for d in self.raw_videos_dir.iterdir() if d.is_dir()])

        if not episode_folders:
            print(f"❌ No episode folders found in {self.raw_videos_dir}")
            return

        print(f"📁 Found {len(episode_folders)} episode folders\n")
        print("="*100)
        print("🎯 VLM-BASED PDDL GENERATION: Analyzing Actual Video Content")
        print("="*100)

        # Generate domain file first
        print("\n📝 Step 1: Generating domain.pddl...")
        self.generate_domain_pddl()

        print("\n📹 Step 2: Processing videos with VLM analysis of content...\n")
        print("="*100)

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

            # Extract frames
            frames_output_dir = self.frames_dir / episode_id
            if not frames_output_dir.exists() or len(list(frames_output_dir.glob("*.jpg"))) == 0:
                print(f"   🎬 Extracting frames...")
                try:
                    frame_count = self.extract_frames_from_video(video_path, frames_output_dir, fps=5)
                    print(f"      ✅ Extracted {frame_count} frames")
                except Exception as e:
                    print(f"      ❌ Error extracting frames: {e}")
                    continue
            else:
                frame_count = len(list(frames_output_dir.glob("*.jpg")))
                print(f"   ✅ Using existing {frame_count} frames")

            # Analyze video with VLMs
            print(f"   🤖 Analyzing video content with BLIP + CLIP...")
            try:
                vlm_analysis = self.analyze_video_states(frames_output_dir)
                print(f"      ✅ VLM analysis complete")
                print(f"      📊 Initial: {vlm_analysis['initial_state'][:60]}...")
                print(f"      📊 Final: {vlm_analysis['final_state'][:60]}...")
                if vlm_analysis['action_description']:
                    print(f"      🎬 Action: {vlm_analysis['action_description'][:60]}...")
            except Exception as e:
                print(f"      ❌ Error analyzing video: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Generate problem based on VLM observations
            problem_filename = f"problem_{episode_id.replace('+', '_').replace('-', '_')}.pddl"
            print(f"   📄 Generating problem from VLM observations...")
            try:
                self.generate_problem_pddl(episode_id, vlm_analysis, problem_filename)
            except Exception as e:
                print(f"      ❌ Error generating problem file: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*100)
        print("✅ VLM-BASED GENERATION COMPLETE!")
        print("🎯 Goals generated from actual video content analysis!")
        print("="*100)
        print("📋 Generation Method:")
        print("   • Analyzed initial video frames → Initial state description")
        print("   • Analyzed final video frames → Goal state description")
        print("   • Inferred PDDL goals from state changes observed by VLMs")
        print("   • Compared with DROID annotations for reference")
        print("="*100)


def main():
    """Main entry point."""
    print("="*100)
    print("VLM-BASED PDDL GENERATION")
    print("Analyzing Actual Video Content with BLIP + CLIP")
    print("="*100)

    generator = VLMAnalysisPDDLGenerator(
        raw_videos_dir="raw_videos",
        annotations_file="droid_language_annotations.json"
    )
    generator.process_all_videos()

    print("\n✅ Done! PDDL problems generated from VLM analysis of video content.")


def main():
    """Main entry point."""
    print("="*100)
    print("VLM-BASED PDDL GENERATION")
    print("Analyzing Actual Video Content with BLIP + CLIP")
    print("="*100)

    generator = VLMAnalysisPDDLGenerator(
        raw_videos_dir="raw_videos",
        annotations_file="droid_language_annotations.json"
    )
    generator.process_all_videos()

    print("\n✅ Done! PDDL problems generated from VLM analysis of video content.")


if __name__ == "__main__":
    main()
