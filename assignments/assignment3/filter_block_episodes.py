"""Filter DROID episodes for block-related tasks from language annotations.

This script parses droid_language_annotations.json and finds episodes that
mention block-related keywords in their instructions.

Usage:
    python filter_block_episodes.py
    python filter_block_episodes.py --output block_episode_ids.json --keywords block cube brick stack
"""

import argparse
import json
from pathlib import Path


# Define keywords to search for (be specific to avoid false positives like "unblock")
BLOCK_KEYWORDS = [
    "block",
    "blocks",
    "cube",
    "cubes",
    "brick",
    "bricks",
    "stack",
    "stacking",
    "tower",
    "blocks world",
]


def find_block_episodes(
    annotations_file: Path,
    keywords: list[str] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Find episode IDs with block-related instructions.
    
    Args:
        annotations_file: Path to droid_language_annotations.json
        keywords: List of keywords to search for (defaults to BLOCK_KEYWORDS)
        verbose: Print progress messages
        
    Returns:
        List of episode IDs that contain block-related instructions
    """
    if keywords is None:
        keywords = BLOCK_KEYWORDS
    
    if verbose:
        print(f"Loading annotations from {annotations_file}...")
        print(f"Searching for keywords: {', '.join(keywords)}")
    
    with open(annotations_file, "r", encoding="utf-8") as f:
        all_annotations = json.load(f)
    
    block_episode_ids = []
    total_episodes = len(all_annotations)
    
    for episode_id, episode_data in all_annotations.items():
        # Check all 3 language instructions
        found_match = False
        for key in ["language_instruction1", "language_instruction2", "language_instruction3"]:
            if key not in episode_data:
                continue
                
            instruction = episode_data[key]
            if not instruction:
                continue
                
            instruction_lower = instruction.lower()
            
            # Check if any keyword is in this instruction
            for keyword in keywords:
                if keyword.lower() in instruction_lower:
                    block_episode_ids.append(episode_id)
                    found_match = True
                    break
            
            if found_match:
                break  # Found a match, no need to check other instructions for this episode
    
    if verbose:
        print(f"\n✅ Filtering Complete:")
        print(f"  Total episodes: {total_episodes:,}")
        print(f"  Block episodes found: {len(block_episode_ids):,}")
        print(f"  Filter rate: {100 * len(block_episode_ids) / total_episodes:.2f}%")
        
        if block_episode_ids:
            print(f"\n  Sample episode IDs:")
            for ep_id in block_episode_ids[:5]:
                # Show sample instruction
                sample_instr = ""
                for key in ["language_instruction1", "language_instruction2", "language_instruction3"]:
                    if key in all_annotations[ep_id]:
                        sample_instr = all_annotations[ep_id][key]
                        break
                print(f"    - {ep_id}: \"{sample_instr[:60]}...\"")
    
    return block_episode_ids


def main():
    parser = argparse.ArgumentParser(
        description="Filter DROID episodes for block-related tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: filter with standard block keywords
  python filter_block_episodes.py
  
  # Custom output file and keywords
  python filter_block_episodes.py --output my_blocks.json --keywords block cube stack
        """,
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="droid_language_annotations.json",
        help="Path to droid_language_annotations.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="block_episode_ids.json",
        help="Output JSON file for filtered episode IDs",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Custom keywords to search for (space-separated)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    annotations_file = Path(args.annotations)
    if not annotations_file.exists():
        print(f"❌ Error: Annotations file not found: {annotations_file}")
        print(f"   Expected location: {annotations_file.absolute()}")
        sys.exit(1)
    
    # Find block episodes
    block_ids = find_block_episodes(
        annotations_file,
        keywords=args.keywords,
        verbose=not args.quiet,
    )
    
    # Save filtered list
    output_file = Path(args.output)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(block_ids, f, indent=2)
    
    print(f"\n💾 Saved {len(block_ids)} episode IDs to: {output_file}")
    print(f"\n📋 Next steps:")
    print(f"   1. Download videos for these episodes:")
    print(f"      python download_droid_episodes.py --episode-ids {output_file}")
    print(f"   2. Or process directly from TFDS:")
    print(f"      python process_droid_episodes.py --episode-ids {output_file}")


if __name__ == "__main__":
    import sys
    main()













