#!/usr/bin/env python3
"""
Multimodal PDDL Generator
Uses both Vision Language Models (BLIP/CLIP) and text annotations 
to generate domain.pddl and problem.pddl files for robot manipulation tasks.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)
import clip
from huggingface_hub import snapshot_download


class MultimodalPDDLGenerator:
    """Generate PDDL files using both visual and textual understanding."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.raw_videos_dir = base_dir / "raw_videos"
        self.frames_dir = base_dir / "frames"
        self.annotations_path = base_dir / "droid_language_annotations.json"
        
        # Initialize models
        print("🔧 Loading VLM models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_mode = "mplug"
        self.mplug_processor: Optional[AutoProcessor] = None
        self.mplug_model: Optional[AutoModelForCausalLM] = None
        self.blip_processor: Optional[BlipProcessor] = None
        self.blip_model: Optional[BlipForConditionalGeneration] = None
        self.max_caption_frames = 12  # limit for expensive captioning models
        self.mplug_prompt = (
            "Describe this frame in detail, focusing on robot actions, block colors, containers, and their "
            "spatial relationships. Mention if a block is inside a container or stacked."
        )

        # Prepare local storage for large VLM checkpoints
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Official checkpoint hosted by MAGAer13 (open access)
        mplug_model_name = "MAGAer13/mplug-owl-llama-7b"
        self.mplug_local_dir = self.models_dir / "mplug-owl-llama-7b"
        try:
            if not self.mplug_local_dir.exists() or not any(self.mplug_local_dir.iterdir()):
                print("   ⬇️  Downloading mPLUG-Owl weights (one-time setup)...")
                snapshot_download(
                    repo_id=mplug_model_name,
                    local_dir=str(self.mplug_local_dir),
                    local_dir_use_symlinks=False,
                    token=os.environ.get("HF_TOKEN"),
                )

            processor_kwargs = {"trust_remote_code": True}
            model_kwargs = {"trust_remote_code": True}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"

            self.mplug_processor = AutoProcessor.from_pretrained(
                str(self.mplug_local_dir),
                **processor_kwargs,
            )
            self.mplug_model = AutoModelForCausalLM.from_pretrained(
                str(self.mplug_local_dir),
                **model_kwargs,
            )
            if hasattr(self.mplug_model, "eval"):
                self.mplug_model.eval()
            print("   ✅ Loaded mPLUG-Owl for captioning")
        except Exception as e:
            print(f"   ⚠️  Failed to load mPLUG-Owl ({e}); falling back to BLIP captioning.")
            self.caption_mode = "blip"
            self._load_blip_captioner()

        # CLIP for object detection and verification
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"   ✅ CLIP loaded on {self.device}")
        
        # Load annotations
        self.annotations = self.load_annotations()
        
    def load_annotations(self) -> Dict:
        """Load and parse language annotations."""
        print("📖 Loading language annotations...")
        
        try:
            with open(self.annotations_path, 'r') as f:
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
    
    def extract_frames(self, video_path: Path, output_dir: Path, fps: int = 10) -> List[Path]:
        """Extract frames from video."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if frames already exist
        existing_frames = sorted(output_dir.glob("*.jpg"))
        if len(existing_frames) >= 5:
            print(f"   ✅ Using {len(existing_frames)} existing frames")
            return existing_frames
        
        # Extract frames using ffmpeg
        output_pattern = output_dir / "%04d.jpg"
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}',
            '-q:v', '2',
            str(output_pattern),
            '-y'
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        frames = sorted(output_dir.glob("*.jpg"))
        print(f"   ✅ Extracted {len(frames)} frames")
        return frames
    
    def _load_blip_captioner(self) -> None:
        """Lazy-load BLIP captioning model (fallback)."""
        if self.blip_processor is not None and self.blip_model is not None:
            return
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_model.eval()
        print("   ✅ Loaded BLIP captioner")

    def caption_image_with_blip(self, image_path: Path) -> str:
        """Generate caption for an image using BLIP."""
        self._load_blip_captioner()
        image = Image.open(image_path).convert('RGB')
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        out = self.blip_model.generate(**inputs, max_length=50)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def caption_image_with_mplug(self, image_path: Path) -> str:
        """Generate caption for an image using mPLUG-Owl."""
        if self.mplug_processor is None or self.mplug_model is None:
            raise RuntimeError("mPLUG-Owl is not available")

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.mplug_prompt},
                    {"type": "image"},
                ],
            }
        ]

        prompt = self.mplug_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.mplug_processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )

        # Move tensors to the first device the model resides on
        target_device = next(self.mplug_model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        pad_token_id = None
        if hasattr(self.mplug_processor, "tokenizer") and hasattr(self.mplug_processor.tokenizer, "eos_token_id"):
            pad_token_id = self.mplug_processor.tokenizer.eos_token_id
        elif hasattr(self.mplug_processor, "eos_token_id"):
            pad_token_id = self.mplug_processor.eos_token_id

        generation_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            generated_ids = self.mplug_model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_text = self.mplug_processor.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True,
        )[0]

        return generated_text.strip()

    def caption_image(self, image_path: Path) -> str:
        """Generate caption for an image using the configured captioner."""
        if self.caption_mode == "mplug" and self.mplug_processor is not None and self.mplug_model is not None:
            try:
                return self.caption_image_with_mplug(image_path)
            except Exception as exc:
                print(f"   ⚠️  mPLUG captioning failed ({exc}); falling back to BLIP.")
                self.caption_mode = "blip"

        return self.caption_image_with_blip(image_path)
    
    def detect_objects_with_clip(self, image_path: Path, candidate_objects: List[str]) -> Dict[str, float]:
        """Detect objects in image using CLIP."""
        image = Image.open(image_path).convert('RGB')
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Create text prompts
        text_prompts = [f"a photo of a {obj}" for obj in candidate_objects]
        text_inputs = clip.tokenize(text_prompts).to(self.device)
        
        # Get similarities
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Return as dict
        results = {}
        for obj, score in zip(candidate_objects, similarity[0].cpu().numpy()):
            results[obj] = float(score)
        
        return results
    
    def analyze_video_multimodal(self, episode_id: str, video_path: Path) -> Dict:
        """Analyze video using both vision and text."""
        print(f"   🔍 Multimodal analysis for {episode_id}")
        
        # Extract frames
        frames_output = self.frames_dir / episode_id
        frames = self.extract_frames(video_path, frames_output, fps=10)
        
        if len(frames) < 2:
            print(f"   ⚠️  Not enough frames extracted")
            return {}
        
        # Get text annotation (use language_instruction1 as primary)
        annot = self.annotations.get(episode_id, {})
        if isinstance(annot, dict):
            text_instruction = annot.get('language_instruction1', '')
            if not text_instruction:
                text_instruction = annot.get('language_instruction2', '')
            if not text_instruction:
                text_instruction = annot.get('language_instruction3', '')
        else:
            text_instruction = ''
        print(f"   📝 Instruction: '{text_instruction}'")
        
        # Select frames for captioning (all frames, downsampled if necessary)
        total_frames = len(frames)
        if total_frames <= self.max_caption_frames:
            selected_indices = list(range(total_frames))
        else:
            step = (total_frames - 1) / (self.max_caption_frames - 1)
            selected_indices = []
            for i in range(self.max_caption_frames):
                idx = int(round(i * step))
                idx = min(idx, total_frames - 1)
                selected_indices.append(idx)
            # Remove duplicates while preserving order
            seen_indices = set()
            selected_indices = [i for i in selected_indices if not (i in seen_indices or seen_indices.add(i))]

        selected_frames = [frames[i] for i in selected_indices]
        print(f"   🖼️  Captioning {len(selected_frames)} frame(s) with {self.caption_mode.upper()}...")

        all_captions = []
        for idx, frame_path in enumerate(selected_frames):
            position = "initial" if idx == 0 else "final" if idx == len(selected_frames) - 1 else f"frame_{idx+1}"
            caption = self.caption_image(frame_path)
            all_captions.append({
                "frame": position,
                "caption": caption,
                "path": frame_path.name,
                "index": selected_indices[idx],
            })
            print(f"      [{position}] {caption}")

        initial_frame = frames[0]
        final_frame = frames[-1]
        initial_caption = all_captions[0]["caption"]
        final_caption = all_captions[-1]["caption"]
        middle_captions = [entry["caption"] for entry in all_captions[1:-1]]
        
        # Detect objects using CLIP
        candidate_objects = [
            "red block", "green block", "blue block", "orange block", "yellow block",
            "bowl", "cup", "container", "mug", "jar",
            "robot arm", "robot gripper", "table"
        ]
        
        print(f"   🎯 Detecting objects with CLIP...")
        initial_objects = self.detect_objects_with_clip(initial_frame, candidate_objects)
        final_objects = self.detect_objects_with_clip(final_frame, candidate_objects)
        
        # Get top detected objects
        initial_top = sorted(initial_objects.items(), key=lambda x: x[1], reverse=True)[:5]
        final_top = sorted(final_objects.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"      Initial objects: {[f'{obj}({score:.2f})' for obj, score in initial_top]}")
        print(f"      Final objects: {[f'{obj}({score:.2f})' for obj, score in final_top]}")
        
        return {
            'text_instruction': text_instruction,
            'initial_caption': initial_caption,
            'final_caption': final_caption,
            'middle_captions': middle_captions,
            'initial_objects': initial_objects,
            'final_objects': final_objects,
            'frames_analyzed': len(frames),
            'captioned_frames': len(selected_frames),
            'captions_detail': all_captions,
        }
    
    def infer_pddl_from_multimodal(self, analysis: Dict) -> Dict:
        """Infer PDDL problem from multimodal analysis."""
        text = analysis['text_instruction'].lower()
        initial_cap = analysis['initial_caption'].lower()
        final_cap = analysis['final_caption'].lower()
        initial_obj = analysis['initial_objects']
        final_obj = analysis['final_objects']
        
        print(f"   🧠 Inferring PDDL structure...")
        
        # Detect colors/blocks from vision (consider both initial & final frames)
        color_blocks = set()
        for color in ['red', 'green', 'blue', 'orange', 'yellow']:
            color_key = f"{color} block"
            if initial_obj.get(color_key, 0) > 0.12 or final_obj.get(color_key, 0) > 0.12:
                color_blocks.add(color)
        detected_block_count = len(color_blocks)
        
        # Detect containers from both vision and text
        has_container = False
        container_type = 'container1'
        
        container_keywords = {
            'bowl': ('bowl', 0.15),
            'cup': ('cup', 0.15),
            'mug': ('mug', 0.12),
            'jar': ('jar', 0.12),
            'container': ('container', 0.10)
        }
        
        # Check CLIP scores
        for name, (key, threshold) in container_keywords.items():
            if initial_obj.get(key, 0) > threshold or final_obj.get(key, 0) > threshold:
                has_container = True
                container_type = f"{name}1"
                print(f"      🥣 Detected {name} (visual)")
                break
        
        # Check text
        if not has_container:
            for name in container_keywords.keys():
                if name in text:
                    has_container = True
                    container_type = f"{name}1"
                    print(f"      🥣 Detected {name} (text)")
                    break
        
        # Determine the best estimate of how many distinct blocks we have
        if has_container:
            # Container tasks typically involve a single manipulated block
            num_blocks = max(1, detected_block_count)
        else:
            num_blocks = max(2, detected_block_count)

        # Refine block count using textual hints (e.g. "three blocks")
        if any(token in text for token in ['three blocks', '3 blocks', 'three-block', 'three block']):
            num_blocks = max(num_blocks, 3)
        elif any(token in text for token in ['four blocks', '4 blocks', 'four-block', 'four block']):
            num_blocks = max(num_blocks, 4)

        # Cap overly large estimates but keep at least 2 for stacking
        if num_blocks > 5:
            num_blocks = 5
        
        # Infer task type from text and vision
        task_type = None
        goal_predicates = []
        initial_predicates = []
        
        # Task detection using both modalities - check remove first (most specific)
        if any(word in text for word in ['remove', 'take out', 'take from']) and has_container:
            # Check if goal is to put on table
            if 'table' in text or 'on the table' in text:
                task_type = 'remove_from_container'
                goal_predicates = ['(on-table block1)']
                initial_predicates = [f'(in block1 {container_type})']
                print(f"      🎯 Task: Remove from container to table")
        
        elif any(word in text for word in ['put', 'place', 'drop']) and has_container:
            if any(word in text for word in ['into', 'in', 'inside', 'in the']):
                task_type = 'put_in_container'
                goal_predicates = [f'(in block1 {container_type})']
                print(f"      🎯 Task: Put in container")
        
        elif any(word in text for word in ['stack', 'pile', 'on top']):
            # Check if it's multi-block stacking
            if any(word in text for word in ['three', '3', 'tower']):
                task_type = 'stack_three'
                num_blocks = 3
                goal_predicates = ['(on block2 block1)', '(on block3 block2)']
                print(f"      🎯 Task: Stack three blocks")
            else:
                task_type = 'stack_two'
                goal_predicates = ['(on block2 block1)']
                print(f"      🎯 Task: Stack two blocks")
        
        elif 'on' in text and 'block' in text:
            # Likely a stacking task mentioned as "put X on Y"
            task_type = 'stack_two'
            goal_predicates = ['(on block2 block1)']
            print(f"      🎯 Task: Stack two blocks (on-inference)")
        
        # If no task detected from text, use vision changes
        if not task_type:
            # Compare initial vs final captions
            if 'stack' in final_cap and 'stack' not in initial_cap:
                task_type = 'stack_two'
                goal_predicates = ['(on block2 block1)']
                print(f"      🎯 Task: Stack (inferred from vision)")
            elif has_container and initial_obj.get(container_type.split('1')[0], 0) < final_obj.get(container_type.split('1')[0], 0):
                task_type = 'put_in_container'
                goal_predicates = [f'(in block1 {container_type})']
                print(f"      🎯 Task: Container (inferred from vision)")
            else:
                # Default: stacking
                task_type = 'stack_two'
                goal_predicates = ['(on block2 block1)']
                print(f"      🎯 Task: Stack (default)")
        
        # Generate initial state if not specified
        if not initial_predicates:
            initial_predicates = ['(empty robot1)']
            for i in range(1, num_blocks + 1):
                initial_predicates.append(f'(on-table block{i})')
                initial_predicates.append(f'(clear block{i})')
            if has_container:
                initial_predicates.append(f'(clear {container_type})')
        else:
            # For remove task
            init_preds = ['(empty robot1)'] + initial_predicates
            if num_blocks > 1:
                for i in range(2, num_blocks + 1):
                    init_preds.append(f'(on-table block{i})')
                    init_preds.append(f'(clear block{i})')
            init_preds.append(f'(clear {container_type})')
            initial_predicates = init_preds
        
        return {
            'task_type': task_type,
            'num_blocks': num_blocks,
            'has_container': has_container,
            'container_type': container_type,
            'goal_predicates': goal_predicates,
            'initial_predicates': initial_predicates,
            'detected_colors': color_blocks
        }
    
    def generate_domain_pddl(self) -> str:
        """Generate domain.pddl file."""
        domain = """(define (domain robot-manipulation)
  (:requirements :strips :typing)
  
  (:types
    block container robot - object
  )
  
  (:predicates
    (on ?x - block ?y - block)          ; block x is on block y
    (on-table ?x - block)               ; block x is on the table
    (in ?x - block ?c - container)      ; block x is in container c
    (clear ?x - object)                 ; object x has nothing on top
    (holding ?r - robot ?x - block)     ; robot r is holding block x
    (empty ?r - robot)                  ; robot r is not holding anything
  )
  
  (:action pick-up
    :parameters (?r - robot ?x - block)
    :precondition (and 
      (empty ?r)
      (on-table ?x)
      (clear ?x)
    )
    :effect (and 
      (holding ?r ?x)
      (not (on-table ?x))
      (not (clear ?x))
      (not (empty ?r))
    )
  )
  
  (:action put-down
    :parameters (?r - robot ?x - block)
    :precondition (holding ?r ?x)
    :effect (and 
      (on-table ?x)
      (clear ?x)
      (empty ?r)
      (not (holding ?r ?x))
    )
  )
  
  (:action stack
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
  )
  
  (:action unstack
    :parameters (?r - robot ?x - block ?y - block)
    :precondition (and 
      (empty ?r)
      (on ?x ?y)
      (clear ?x)
    )
    :effect (and 
      (holding ?r ?x)
      (clear ?y)
      (not (on ?x ?y))
      (not (clear ?x))
      (not (empty ?r))
    )
  )
  
  (:action put-in-container
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
  )
  
  (:action take-from-container
    :parameters (?r - robot ?x - block ?c - container)
    :precondition (and 
      (empty ?r)
      (in ?x ?c)
    )
    :effect (and 
      (holding ?r ?x)
      (clear ?c)
      (not (in ?x ?c))
      (not (empty ?r))
    )
  )
)
"""
        return domain
    
    def generate_problem_pddl(self, episode_id: str, pddl_config: Dict) -> str:
        """Generate problem.pddl file."""
        # Clean episode ID for problem name
        problem_name = episode_id.replace('+', '_').replace('-', '_')
        
        # Build objects list
        objects = ['robot1 - robot']
        for i in range(1, pddl_config['num_blocks'] + 1):
            objects.append(f'block{i} - block')
        if pddl_config['has_container']:
            objects.append(f"{pddl_config['container_type']} - container")
        
        # Build problem
        problem = f"""(define (problem {problem_name})
  (:domain robot-manipulation)
  
  (:objects
    {' '.join(objects)}
  )
  
  (:init
    {' '.join(pddl_config['initial_predicates'])}
  )
  
  (:goal
    (and {' '.join(pddl_config['goal_predicates'])})
  )
)
"""
        return problem
    
    def process_all_videos(self):
        """Process all videos and generate PDDL files."""
        print("\n" + "=" * 70)
        print("🚀 MULTIMODAL PDDL GENERATION")
        print("=" * 70)
        
        # Generate domain file
        print("\n📝 Generating domain.pddl...")
        domain_content = self.generate_domain_pddl()
        domain_path = self.base_dir / "domain.pddl"
        domain_path.write_text(domain_content)
        print(f"   ✅ Domain file saved: {domain_path}")
        
        # Process each video
        video_dirs = sorted([d for d in self.raw_videos_dir.iterdir() if d.is_dir()])
        
        print(f"\n📹 Processing {len(video_dirs)} videos...\n")
        
        for idx, video_dir in enumerate(video_dirs, 1):
            episode_id = video_dir.name
            print(f"[{idx}/{len(video_dirs)}] {episode_id}")
            print("-" * 70)
            
            # Find video file (in recordings/MP4/ subdirectory)
            video_files = list(video_dir.glob("recordings/MP4/*.mp4"))
            if not video_files:
                print(f"   ⚠️  No video found, skipping")
                continue
            
            video_path = video_files[0]
            print(f"   📹 Video: {video_path.name}")
            
            # Multimodal analysis
            analysis = self.analyze_video_multimodal(episode_id, video_path)
            
            if not analysis:
                print(f"   ⚠️  Analysis failed, skipping")
                continue
            
            # Infer PDDL structure
            pddl_config = self.infer_pddl_from_multimodal(analysis)
            
            # Generate problem file
            problem_content = self.generate_problem_pddl(episode_id, pddl_config)
            
            # Save problem file
            problem_filename = f"problem_{episode_id.split('+')[0]}_{episode_id.split('+')[1]}_{episode_id.split('+')[2].replace('-', '_')}.pddl"
            problem_path = self.base_dir / problem_filename
            problem_path.write_text(problem_content)
            
            print(f"   ✅ Problem file saved: {problem_path.name}")
            print(f"   📊 Task: {pddl_config['task_type']}")
            print(f"   🎯 Goals: {pddl_config['goal_predicates']}")
            print()
        
        print("=" * 70)
        print("✨ PDDL generation complete!")
        print("=" * 70)


def main():
    """Main entry point."""
    base_dir = Path("/workspaces/eng-ai-agents/assignments/assignment3")
    
    generator = MultimodalPDDLGenerator(base_dir)
    generator.process_all_videos()
    
    print("\n🔍 Validating generated PDDL files...")
    try:
        from unified_planning.io import PDDLReader
        reader = PDDLReader()
        
        problem_files = sorted(base_dir.glob("problem_*.pddl"))
        valid_count = 0
        
        for problem_file in problem_files:
            try:
                problem = reader.parse_problem(str(base_dir / "domain.pddl"), str(problem_file))
                print(f"   ✅ {problem_file.name}")
                valid_count += 1
            except Exception as e:
                print(f"   ❌ {problem_file.name}: {str(e)[:60]}...")
        
        print(f"\n✅ {valid_count}/{len(problem_files)} problems validated!")
        
    except ImportError:
        print("   ⚠️  unified-planning not available, skipping validation")


if __name__ == "__main__":
    main()

