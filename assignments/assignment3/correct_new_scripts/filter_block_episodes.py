#!/usr/bin/env python3
"""
Script to filter episode IDs where only blocks are involved based on instructions.
Matches specific instruction patterns related to block manipulation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set

ANNOTATIONS_FILE = "droid_language_annotations.json"
OUTPUT_FILE = "correct_new_scripts/block_episode_ids_filtered.json"

# Patterns to match for block-related instructions
PATTERNS = [
    r"put\s+the\s+\w+\s+block\s+inside\s+the\s+\w+\s+(?:bowl|cup)",
    r"put\s+the\s+\w+\s+block\s+in\s+the\s+\w+\s+(?:bowl|cup)",
    r"put\s+the\s+\w+\s+block\s+into\s+the\s+\w+\s+(?:bowl|cup)",
    r"pick\s+up\s+the\s+block",
    r"pick\s+up\s+the\s+\w+\s+block",
    r"remove\s+the\s+\w+\s+block\s+from\s+the\s+\w+\s+and\s+put\s+it\s+on\s+the\s+table",
    r"remove\s+the\s+\w+\s+block\s+from\s+the\s+\w+\s+and\s+place\s+it\s+on\s+the\s+table",
]

# Additional keywords that indicate block manipulation
BLOCK_KEYWORDS = ["block", "blocks"]
CONTAINER_KEYWORDS = ["bowl", "cup", "container"]


def load_annotations() -> Dict:
    """Load DROID language annotations."""
    annotations_path = Path(ANNOTATIONS_FILE)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Handle Git LFS and merge conflicts
        if content.startswith("version https://git-lfs.github.com"):
            lines = content.split('\n')
            json_start = next((i for i, line in enumerate(lines) if line.strip().startswith('{')), None)
            if json_start is not None:
                content = '\n'.join(lines[json_start:])
        
        # Handle merge conflicts
        if '<<<<<<<' in content and '=======' in content:
            parts = content.split('=======')
            if len(parts) > 1:
                content = parts[1]
                if '>>>>>>>' in content:
                    content = content.split('>>>>>>>')[0]
        elif '<<<<<<<' in content:
            content = content.split('<<<<<<<')[0]
        
        return json.loads(content)


def matches_pattern(instruction: str, patterns: List[str]) -> bool:
    """Check if instruction matches any of the patterns."""
    instruction_lower = instruction.lower()
    
    # Check against regex patterns
    for pattern in patterns:
        if re.search(pattern, instruction_lower):
            return True
    
    return False


def is_block_related(instruction: str) -> bool:
    """
    Check if instruction is related to block manipulation.
    Must contain block keywords and be about manipulation (put, pick, remove, etc.).
    """
    instruction_lower = instruction.lower()
    
    # Must contain block keyword
    if not any(keyword in instruction_lower for keyword in BLOCK_KEYWORDS):
        return False
    
    # Must contain manipulation verbs
    manipulation_verbs = [
        "put", "pick", "remove", "place", "move", "stack", "unstack",
        "inside", "in", "into", "on", "from"
    ]
    
    if not any(verb in instruction_lower for verb in manipulation_verbs):
        return False
    
    # Check if it matches our specific patterns
    if matches_pattern(instruction, PATTERNS):
        return True
    
    # Additional check: block + container manipulation
    has_block = any(keyword in instruction_lower for keyword in BLOCK_KEYWORDS)
    has_container = any(keyword in instruction_lower for keyword in CONTAINER_KEYWORDS)
    
    # Block manipulation with container
    if has_block and has_container:
        return True
    
    # Simple block manipulation (pick up, move block, etc.)
    if has_block and ("pick" in instruction_lower or "move" in instruction_lower):
        return True
    
    return False


def filter_block_episodes(annotations: Dict) -> List[str]:
    """
    Filter episode IDs where instructions are block-related.
    """
    filtered_episodes = []
    
    for episode_id, episode_data in annotations.items():
        if not isinstance(episode_data, dict):
            continue
        
        # Check all instruction fields
        instructions = []
        for key in ['language_instruction1', 'language_instruction2', 'language_instruction3']:
            if key in episode_data:
                instruction = episode_data[key]
                if isinstance(instruction, str):
                    instructions.append(instruction)
        
        # If any instruction matches, include this episode
        for instruction in instructions:
            if is_block_related(instruction):
                filtered_episodes.append(episode_id)
                break  # Only add once per episode
    
    return filtered_episodes


def main():
    """Main function."""
    print("=" * 70)
    print("FILTERING BLOCK-RELATED EPISODES")
    print("=" * 70)
    
    # Load annotations
    print("\n📖 Loading annotations...")
    try:
        annotations = load_annotations()
        print(f"   ✅ Loaded {len(annotations)} episodes")
    except Exception as e:
        print(f"   ❌ Error loading annotations: {e}")
        return
    
    # Filter episodes
    print("\n🔍 Filtering block-related episodes...")
    filtered_episodes = filter_block_episodes(annotations)
    print(f"   ✅ Found {len(filtered_episodes)} block-related episodes")
    
    # Show some examples
    print("\n📋 Sample filtered episodes (first 10):")
    for i, episode_id in enumerate(filtered_episodes[:10], 1):
        episode_data = annotations.get(episode_id, {})
        instructions = []
        for key in ['language_instruction1', 'language_instruction2', 'language_instruction3']:
            if key in episode_data:
                instructions.append(episode_data[key])
        print(f"   {i}. {episode_id}")
        if instructions:
            print(f"      Instruction: {instructions[0]}")
    
    # Save filtered episode IDs
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(filtered_episodes, f, indent=2)
    
    print(f"\n✅ Saved {len(filtered_episodes)} episode IDs to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

