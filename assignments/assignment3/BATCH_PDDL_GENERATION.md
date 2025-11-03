# Batch PDDL Generation from DROID Videos

This document describes the automated PDDL generation system for DROID robot manipulation videos.

## Overview

The `batch_video_to_pddl.py` script processes all videos in the `raw_videos` folder and generates:
1. **One unified PDDL domain file** (`domain.pddl`) - shared across all problems
2. **Multiple PDDL problem files** (one per video) - `problem_<episode_id>.pddl`

## Features

Based on [NVIDIA's VLM Prompt Engineering Guide](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/):

- **Video Understanding**: Uses LLaVA for sequential visual understanding
- **Temporal Analysis**: Extracts actions over time (not just single frames)
- **Structured Extraction**: Infers objects, actions, initial states, and goal states
- **PDDL Validation**: Optional validation using Unified Planning library

## Usage

### Basic Usage

```bash
# Process all videos in raw_videos folder
python batch_video_to_pddl.py --raw-videos-dir raw_videos --llava

# Process limited number of videos (for testing)
python batch_video_to_pddl.py --raw-videos-dir raw_videos --llava --max-videos 3

# With validation
python batch_video_to_pddl.py --raw-videos-dir raw_videos --llava --validate
```

### Options

- `--raw-videos-dir`: Directory containing episode folders (default: `raw_videos`)
- `--llava`: Use LLaVA for video understanding (required)
- `--llava-model`: LLaVA model ID (default: `llava-hf/llava-v1.6-vicuna-7b-hf`)
- `--max-frames`: Maximum frames to analyze per video (default: 10)
- `--fps`: Frame extraction rate (default: 0.5)
- `--max-videos`: Limit number of videos to process
- `--validate`: Validate PDDL files using Unified Planning

## PDDL Domain Structure

The generated domain includes:

### Types
- `block` - Manipulatable objects (blocks, cubes)
- `container` - Containers (bowls, cups, boxes)
- `surface` - Surfaces (tables, platforms)
- `robot` - Robot agent

### Predicates
- `(holding ?r - robot ?o - block)` - Robot is holding an object
- `(on ?o1 - block ?o2 - object)` - Object is on another object
- `(clear ?o - block)` - Object has nothing on top
- `(in ?o - block ?c - container)` - Object is in a container
- `(on-table ?o - block)` - Object is on the table
- `(empty ?c - container)` - Container is empty
- `(open ?c - container)` - Container is open
- `(closed ?c - container)` - Container is closed

### Actions
Dynamically generated based on extracted actions:
- `pick` - Pick up a block from the table
- `place` - Place a block on a surface
- `stack` - Stack one block on another
- `put-in` - Put a block in a container
- `open-container` - Open a container
- `close-container` - Close a container

## PDDL Problem Structure

Each problem file includes:

### Objects
- Blocks (e.g., `block1`, `block2`)
- Containers (if detected: `bowl1`, `cup1`)
- Surfaces (`table1`)
- Robot (`robot1`)

### Initial State
Extracted from VLM analysis of early video frames:
- Objects on table
- Clear objects
- Container states

### Goal State
Extracted from VLM analysis of final video frames:
- Desired object arrangements
- Stacking goals
- Container goals

### Comments
Each problem file includes VLM analysis comments:
- Overall video description
- Extracted actions
- Initial state hints
- Goal state hints

## VLM Prompt Engineering

Based on NVIDIA's guide, the script uses specific prompts for video understanding:

1. **Overall Description**: "What happened in this video? Elaborate on the visual and narrative elements in detail. Highlight all actions performed by the robot."

2. **Object Detection**: "What objects are visible in this video? List all objects including blocks, containers, and surfaces."

3. **Action Detection**: "What actions did the robot perform? List all manipulation actions in sequence."

4. **Initial State**: "What is the initial state? Describe what objects are where at the beginning."

5. **Goal State**: "What is the goal state? Describe what the robot is trying to achieve."

## Requirements

```bash
pip install transformers accelerate safetensors sentencepiece pillow torch
pip install unified-planning  # Optional, for validation
```

## Output

After processing, you'll have:

```
domain.pddl                          # Unified domain file
problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl
problem_RAD_c6cf6b42_2023_08_31_14h_00m_49s.pddl
problem_TRI_52ca9b6a_2024_01_16_16h_43m_04s.pddl
problem_GuptaLab_553d1bd5_2023_05_19_10h_36m_14s.pddl
problem_AUTOLab_0d4edc83_2023_10_27_20h_25m_34s.pddl
```

## Validation

Use Unified Planning to validate PDDL files:

```python
from unified_planning.io import PDDLReader

reader = PDDLReader()
problem = reader.parse_problem("domain.pddl", "problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl")
print("✅ PDDL files are valid!")
```

## Notes

- VLM processing can be slow (especially on CPU). Consider using `--max-videos` for testing.
- The domain file is generated after analyzing all videos to capture all possible actions.
- Each video's problem file is generated independently based on its VLM analysis.
- The script automatically selects non-stereo MP4 files when available.

## References

- [NVIDIA VLM Prompt Engineering Guide](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
- [Unified Planning Library](https://github.com/aiplan4eu/unified-planning)
- [Unified Planning PDDL Usage Example](https://colab.research.google.com/github/aiplan4eu/unified-planning/blob/master/docs/notebooks/io/01-pddl-usage-example.ipynb)

