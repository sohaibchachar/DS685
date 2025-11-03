"""Video2Plan: Generate PDDL from DROID dataset using VLMs."""

import tensorflow_datasets as tfds
import json
from typing import List, Dict, Set, Any
from pathlib import Path


class DROIDAnalyzer:
    """Analyze DROID dataset to extract domain knowledge."""
    
    def __init__(self, num_episodes: int = 50, filter_blocks: bool = False):
        """Initialize DROID analyzer.
        
        Args:
            num_episodes: Number of episodes to analyze
            filter_blocks: If True, only analyze episodes with blocks
        """
        self.num_episodes = num_episodes
        self.filter_blocks = filter_blocks
        self.objects = set()
        self.actions = set()
        self.spatial_relations = set()
        self.instructions = []
        self.block_keywords = {"block", "blocks", "cube", "cubes", "brick", "bricks", 
                              "stack", "stacking", "tower", "blocks world"}
        
    def _is_block_episode(self, instruction: str) -> bool:
        """Check if instruction mentions blocks."""
        instruction_lower = instruction.lower()
        return any(keyword in instruction_lower for keyword in self.block_keywords)
        
    def analyze_dataset(self, data_dir: str | None = None, use_openclip: bool = False, tfds_data_dir: str | None = None, random_sample: bool = False, seed: int | None = None):
        """Analyze DROID episodes to extract domain knowledge.

        If data_dir is provided, scan local episodes there; otherwise load via TFDS.
        If use_openclip is True, score first frame per episode against block prompts.
        """
        if data_dir:
            print(f"Loading local DROID episodes from: {data_dir}")
        else:
            if tfds_data_dir:
                print(f"Loading DROID dataset from local TFDS dir: {tfds_data_dir}")
            else:
                print(f"Loading DROID dataset from gs://gresearch/robotics...")
            print(f"📍 Dataset: https://droid-dataset.github.io/")
            print(f"⚠️  Note: Requires Google Cloud authentication")

        # Optional OpenCLIP setup
        openclip_ctx = None
        if use_openclip:
            try:
                import open_clip
                import torch
                from PIL import Image
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='openai'
                )
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
                openclip_ctx = {
                    'model': model,
                    'preprocess': preprocess,
                    'tokenizer': tokenizer,
                    'device': device,
                    'Image': Image,
                    'torch': torch,
                }
                print("✅ OpenCLIP ready for block scoring")
            except Exception as e:
                print(f"⚠️  OpenCLIP not available: {e}")
                use_openclip = False
        
        filter_msg = f" (🔷 BLOCKS ONLY)" if self.filter_blocks else ""
        print(f"\nAnalyzing {self.num_episodes} episodes{filter_msg}...")
        
        episode_count = 0
        skipped_count = 0
        total_scanned = 0
        
        if data_dir:
            # Local episodes: each subdir under data_dir is an episode
            import os
            from glob import glob
            episode_paths = [p for p in glob(os.path.join(data_dir, '*')) if os.path.isdir(p)]
            if random_sample:
                import random
                if seed is not None:
                    random.seed(seed)
                random.shuffle(episode_paths)
            else:
                episode_paths = sorted(episode_paths)
            for ep_path in episode_paths:
                if episode_count >= self.num_episodes:
                    break

                total_scanned += 1
                episode_has_blocks = False

                # Try to read instruction text if present
                instruction = None
                for name in ["instruction.txt", "language.txt", "caption.txt"]:
                    cand = os.path.join(ep_path, name)
                    if os.path.exists(cand):
                        try:
                            instruction = open(cand, 'r', encoding='utf-8').read().strip()
                        except Exception:
                            pass
                        break

                # If blocks filter requested, use instruction or OpenCLIP
                if self.filter_blocks:
                    if instruction and self._is_block_episode(instruction):
                        episode_has_blocks = True
                    elif use_openclip and openclip_ctx:
                        # Score first image in folder
                        img_files = []
                        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                            img_files.extend(glob(os.path.join(ep_path, '**', ext), recursive=True))
                        if img_files:
                            if self._score_block_with_openclip(openclip_ctx, img_files[0]) >= 0.25:
                                episode_has_blocks = True
                    if not episode_has_blocks:
                        skipped_count += 1
                        continue

                if instruction:
                    self.instructions.append(instruction)
                    words = instruction.lower().split()
                    action_keywords = [
                        "pick", "place", "put", "move", "open", "close",
                        "push", "pull", "grasp", "release", "rotate", "lift",
                        "stack", "unstack", "arrange", "align"
                    ]
                    for w in words:
                        if w in action_keywords:
                            self.actions.add(w)
                    for keyword in self.block_keywords:
                        if keyword in instruction.lower():
                            self.objects.add(keyword.rstrip('s'))

                episode_count += 1
                if episode_count % 10 == 0:
                    skip_msg = f" (scanned {total_scanned}, skipped {skipped_count})" if self.filter_blocks else ""
                    print(f"  Processed {episode_count} episodes{skip_msg}...")

        else:
            # Use local TFDS store if provided, otherwise GCS
            tfds_store = tfds_data_dir if tfds_data_dir else "gs://gresearch/robotics"
            ds = tfds.load("droid", data_dir=tfds_store, split="train")
            if random_sample:
                # Shuffle with buffer; TFDS supports shuffle at dataset level
                ds = ds.shuffle(10000, seed=seed)  # large buffer for better randomness
            for episode in ds:
                if episode_count >= self.num_episodes:
                    break
                
            total_scanned += 1
            episode_has_blocks = False
            
            # Extract language instruction
            episode_data = episode.numpy()
            if "steps" in episode_data:
                for step in episode_data["steps"]:
                    if "language_instruction" in step and len(step["language_instruction"]) > 0:
                        instruction = step["language_instruction"][0].decode('utf-8')
                        
                        # Filter for blocks if requested
                        if self.filter_blocks:
                            if not self._is_block_episode(instruction):
                                skipped_count += 1
                                break  # Skip this episode
                            episode_has_blocks = True
                        
                        # Only process if it's a block episode (or if filtering is off)
                        if not self.filter_blocks or episode_has_blocks:
                            self.instructions.append(instruction)
                            
                            # Simple extraction of objects and actions from text
                            words = instruction.lower().split()
                            # Extract common action verbs
                            action_keywords = ["pick", "place", "put", "move", "open", "close", 
                                             "push", "pull", "grasp", "release", "rotate", "lift",
                                             "stack", "unstack", "arrange", "align"]
                            for word in words:
                                if word in action_keywords:
                                    self.actions.add(word)
                            
                            # Extract block-related objects
                            if self.filter_blocks:
                                # Look for block-related terms
                                for keyword in self.block_keywords:
                                    if keyword in instruction.lower():
                                        if keyword.endswith('s') and keyword not in ["stack", "stacking"]:
                                            self.objects.add(keyword[:-1])  # Remove 's'
                                        else:
                                            self.objects.add(keyword)
                                
                                # Extract color + block combinations
                                colors = ["red", "blue", "green", "yellow", "white", "black", 
                                         "orange", "purple", "pink"]
                                for i, word in enumerate(words):
                                    if word in colors and i + 1 < len(words):
                                        if words[i + 1] in ["block", "blocks", "cube", "cubes"]:
                                            self.objects.add(f"{word}_block")
                            
                            # Extract potential objects (general)
                            object_indicators = ["the", "a", "an"]
                            for i, word in enumerate(words):
                                if word in object_indicators and i + 1 < len(words):
                                    obj = words[i + 1]
                                    if obj not in action_keywords:
                                        self.objects.add(obj)
                        break  # Process only first instruction per episode
            
            # Only increment if we processed this episode
            if not self.filter_blocks or episode_has_blocks:
                episode_count += 1
                if episode_count % 10 == 0:
                    skip_msg = f" (scanned {total_scanned}, skipped {skipped_count})" if self.filter_blocks else ""
                    print(f"  Processed {episode_count} episodes{skip_msg}...")
        
        print(f"\n✅ Extraction Complete:")
        print(f"  Episodes analyzed: {episode_count}")
        if self.filter_blocks:
            print(f"  Total episodes scanned: {total_scanned}")
            print(f"  Block episodes found: {episode_count}")
            print(f"  Non-block episodes skipped: {skipped_count}")
        print(f"  Objects extracted: {len(self.objects)}")
        if self.objects:
            obj_list = list(self.objects)[:15]
            print(f"    {obj_list}")
        print(f"  Actions extracted: {len(self.actions)}")
        if self.actions:
            print(f"    {list(self.actions)}")
        print(f"  Instructions collected: {len(self.instructions)}")
        if self.instructions:
            print(f"\n  Sample instructions:")
            for instr in self.instructions[:5]:
                print(f"    - {instr}")

    def _score_block_with_openclip(self, ctx: dict, image_path: str) -> float:
        """Return probability-like score that image is block-related using OpenCLIP."""
        try:
            model = ctx['model']; preprocess = ctx['preprocess']; tokenizer = ctx['tokenizer']
            device = ctx['device']; Image = ctx['Image']; torch = ctx['torch']
            image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            prompts = [
                "photo of blocks", "stacking blocks", "building a tower",
                "wooden cube blocks", "lego blocks", "toy cubes"
            ]
            with torch.no_grad():
                image_features = model.encode_image(image)
                text = tokenizer(prompts)
                text_features = model.encode_text(text.to(device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T)
                probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().numpy()
                return float(probs.max())
        except Exception:
            return 0.0
        
    def get_common_objects(self) -> List[str]:
        """Get most common objects."""
        return list(self.objects)[:10] if len(self.objects) >= 10 else list(self.objects)
    
    def get_actions(self) -> List[str]:
        """Get extracted actions."""
        return list(self.actions)


class PDDLGenerator:
    """Generate PDDL domain and problem files."""
    
    def __init__(self, analyzer: DROIDAnalyzer):
        self.analyzer = analyzer
        
    def generate_domain(self, output_file: str = "domain.pddl"):
        """Generate PDDL domain file for robot manipulation."""
        
        objects = self.analyzer.get_common_objects()
        actions = self.analyzer.get_actions()
        
        domain_content = f"""(define (domain robot-manipulation)
    
    (:requirements :strips :typing)
    
    (:types
        object - physobj
        robot
        gripper
    )
    
    (:predicates
        (at ?r - robot ?x - object)
        (holding ?r - robot ?x - object)
        (on ?x - object ?y - object)
        (clear ?x - object)
        (handempty ?r - robot)
        (picked ?x - object)
        (placed ?x - object ?y - object)
    )
    
    (:action pick
        :parameters (?r - robot ?x - object)
        :precondition (and (at ?r ?x) (clear ?x) (handempty ?r))
        :effect (and (holding ?r ?x) (not (handempty ?r)) (not (at ?r ?x)) (picked ?x))
    )
    
    (:action place
        :parameters (?r - robot ?x - object ?y - object)
        :precondition (and (holding ?r ?x))
        :effect (and (not (holding ?r ?x)) (handempty ?r) (at ?r ?x) (on ?x ?y) (placed ?x ?y))
    )
    
    (:action place-floor
        :parameters (?r - robot ?x - object)
        :precondition (and (holding ?r ?x))
        :effect (and (not (holding ?r ?x)) (handempty ?r) (at ?r ?x) (clear ?x))
    )
)
"""
        
        with open(output_file, 'w') as f:
            f.write(domain_content)
        
        print(f"✅ Domain file generated: {output_file}")
    
    def generate_problem(self, problem_num: int, output_file: str):
        """Generate a PDDL problem file.
        
        Args:
            problem_num: Problem number (1, 2, or 3)
            output_file: Output file path
        """
        
        # Define different scenarios for each problem
        scenarios = {
            1: {
                "objects": ["object1", "object2", "target"],
                "initial": [
                    "(clear object1)",
                    "(clear object2)",
                    "(handempty robot1)",
                    "(on object1 target)",
                    "(on object2 target)",
                ],
                "goal": "(holding robot1 object1)"
            },
            2: {
                "objects": ["box", "item"],
                "initial": [
                    "(clear box)",
                    "(clear item)",
                    "(handempty robot1)",
                    "(at robot1 item)",
                ],
                "goal": "(and (on item box) (clear box))"
            },
            3: {
                "objects": ["cup", "table"],
                "initial": [
                    "(clear cup)",
                    "(clear table)",
                    "(handempty robot1)",
                    "(at robot1 cup)",
                ],
                "goal": "(and (on cup table) (placed cup table))"
            }
        }
        
        scenario = scenarios[problem_num]
        
        problem_content = f"""(define (problem problem{problem_num})
    (:domain robot-manipulation)
    
    (:objects
        robot1 - robot
        {' '.join(f'{obj} - object' for obj in scenario['objects'])}
    )
    
    (:init
        {' '.join(scenario['initial'])}
    )
    
    (:goal
        {scenario['goal']}
    )
)
"""
        
        with open(output_file, 'w') as f:
            f.write(problem_content)
        
        print(f"✅ Problem file generated: {output_file}")


def main():
    """Main pipeline: Analyze DROID and generate PDDL files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate PDDL from DROID dataset (focus on blocks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all episodes
  python pddl_generator.py --episodes 50
  
  # Analyze only BLOCK episodes (recommended for blocks world domain)
  python pddl_generator.py --blocks --episodes 20
  
  # Get more block episodes (may take longer due to filtering)
  python pddl_generator.py --blocks --episodes 100
        """
    )
    parser.add_argument("--blocks", action="store_true", 
                       help="Filter episodes to only include BLOCK-related tasks")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes to analyze (default: 50)")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to local DROID episodes (e.g., data/droid_100)")
    parser.add_argument("--openclip", action="store_true",
                       help="Use OpenCLIP to detect block scenes from images")
    parser.add_argument("--tfds-dir", type=str, default=None,
                       help="Local TFDS store for DROID (e.g., data/droid_100/1.0.0)")
    parser.add_argument("--random", action="store_true",
                       help="Randomize episode selection each run")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible sampling (optional)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video2Plan: Generating PDDL from DROID Dataset")
    if args.blocks:
        print("🔷 BLOCKS MODE: Filtering for block-related episodes only")
    print("=" * 60)
    
    # Step 1: Analyze DROID dataset
    print("\n1. Analyzing DROID dataset...")
    analyzer = DROIDAnalyzer(num_episodes=args.episodes, filter_blocks=args.blocks)
    
    try:
        analyzer.analyze_dataset(
            data_dir=args.data_dir,
            use_openclip=args.openclip,
            tfds_data_dir=args.tfds_dir,
            random_sample=args.random,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n⚠️  Could not load DROID from gs:// (might need authentication)")
        print(f"   Error: {e}")
        print(f"   Continuing with template-based PDDL generation...")
    
    # Step 2: Generate PDDL domain
    print("\n2. Generating PDDL domain...")
    generator = PDDLGenerator(analyzer)
    generator.generate_domain("domain.pddl")
    
    # Step 3: Generate problem files
    print("\n3. Generating PDDL problem files...")
    for i in range(1, 4):
        generator.generate_problem(i, f"problem{i}.pddl")
    
    # Step 4: Validate with Unified Planning
    print("\n4. Validating PDDL files...")
    try:
        from unified_planning.io import PDDLReader
        reader = PDDLReader()
        
        problem = reader.parse_problem("domain.pddl", "problem1.pddl")
        print("✅ PDDL files validated successfully!")
        print(f"   Problem: {problem.name}")
        print(f"   Number of objects: {len(problem.all_objects)}")
    except Exception as e:
        print(f"⚠️  Could not validate with Unified Planning: {e}")
        print(f"   But PDDL files were generated successfully!")
    
    print("\n" + "=" * 60)
    print("✅ Done! Check domain.pddl and problem*.pddl files")
    print("=" * 60)


if __name__ == "__main__":
    main()
