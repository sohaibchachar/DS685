# PDDL Generation from Robot Manipulation Videos

## Overview

This assignment involves automatically extracting PDDL (Planning Domain Definition Language) specifications from videos of robot demonstrations. The solution uses **Visual Language Models (VLMs)** to analyze robot manipulation videos and generate PDDL domain and problem files.

## Generated Files

### Domain File
- **`domain.pddl`**: A unified PDDL domain file defining the robot manipulation domain with:
  - **Types**: block, container, location, robot
  - **Predicates**: on, in, clear, holding, empty, on-table
  - **Actions**: pick-up, put-down, stack, unstack, put-in-container

### Problem Files (5 total - one per video)
1. `problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl`
2. `problem_AUTOLab_0d4edc83_2023_10_27_20h_25m_34s.pddl`
3. `problem_GuptaLab_553d1bd5_2023_05_19_10h_36m_14s.pddl`
4. `problem_RAD_c6cf6b42_2023_08_31_14h_00m_49s.pddl`
5. `problem_TRI_52ca9b6a_2024_01_16_16h_43m_04s.pddl`

Each problem file includes:
- Object declarations (blocks, robot, containers if applicable)
- Initial state predicates
- Goal state predicates
- Comments with VLM analysis descriptions

## Technology Stack

### Visual Language Models Used
1. **BLIP (Salesforce/blip-image-captioning-large)**
   - Used for image captioning and visual question answering
   - Analyzes video frames to extract scene descriptions
   - Answers specific questions about initial states, actions, and goals

2. **CLIP (OpenAI ViT-B/32)**
   - Used for object detection and scene understanding
   - Classifies images against text prompts
   - Identifies objects, blocks, containers, and robot actions

### Key Libraries
- `transformers`: For loading and running BLIP model
- `clip` (OpenAI): For object detection and classification
- `torch`: Deep learning framework
- `unified-planning`: For PDDL validation
- `ffmpeg`: For video frame extraction
- `PIL`: For image processing

## Implementation Approach

### 1. Video Frame Extraction
```bash
ffmpeg -i video.mp4 -vf fps=0.5 frames/%04d.jpg
```
Extracts frames at 0.5 FPS from each video for analysis.

### 2. VLM Analysis Pipeline

#### Initial State Analysis (First Frames)
- Question: "What objects are on the table and where are they positioned?"
- Uses BLIP VQA to understand the starting configuration

#### Action Detection (Middle Frames)
- Question: "What is the robot doing?"
- Extracts manipulation actions performed

#### Goal State Analysis (Final Frames)
- Question: "What is the final arrangement of objects on the table?"
- Determines the desired end state

#### Object Detection (All Frames)
- Uses CLIP with text prompts like:
  - "a red block on a table"
  - "blocks stacked on each other"
  - "a robot arm holding a block"
- Detects presence of objects with confidence scores

### 3. PDDL Generation

#### Domain Generation
- Standard block manipulation domain
- Actions follow classical planning conventions (STRIPS)
- Includes standard predicates for block world problems

#### Problem Generation
- Infers number of blocks from VLM descriptions
- Creates initial state predicates based on scene analysis
- Generates goal predicates from goal state descriptions
- Uses heuristics to determine stacking vs. container goals

### 4. Validation
All generated PDDL files are validated using the Unified Planning library to ensure:
- Syntactic correctness
- Type consistency
- Well-formed predicates and actions

## Usage

### Running the Generation Script

```bash
# Activate virtual environment
source /workspaces/eng-ai-agents/.venv/bin/activate

# Run the PDDL generator
cd /workspaces/eng-ai-agents/assignments/assignment3
python generate_pddl.py
```

### Output

```
======================================================================
PDDL Generation from Robot Manipulation Videos
Using BLIP + CLIP Visual Language Models
======================================================================

✅ Models loaded successfully!
📁 Found 5 episode folders

📝 Step 1: Generating domain.pddl...
✅ Generated domain.pddl

📹 Step 2: Processing videos and generating problem files...

[1/5] Processing: RAIL+d027f2ae+2023-06-05-16h-33m-01s
   ✅ Analysis complete
   ✅ Generated problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl

... (continues for all 5 videos)

✅ PDDL generation complete!
```

## PDDL Domain Structure

### Types Hierarchy
```
object
├── block        (manipulatable objects)
├── container    (bowls, cups, boxes)
└── location     (surfaces, positions)

agent
└── robot        (robotic manipulator)
```

### Core Predicates
- `(on ?obj - block ?target - object)` - Object is on another object
- `(in ?obj - block ?container - container)` - Object is in a container
- `(clear ?obj - object)` - Object has nothing on top
- `(holding ?r - robot ?obj - block)` - Robot is holding an object
- `(empty ?r - robot)` - Robot's gripper is empty
- `(on-table ?obj - block)` - Object is directly on the table

### Actions

#### pick-up
Picks up a block from the table.
- **Parameters**: robot, block
- **Precondition**: robot empty, block clear, block on table
- **Effect**: robot holding block, block not on table

#### put-down
Places a held block on the table.
- **Parameters**: robot, block
- **Precondition**: robot holding block
- **Effect**: robot empty, block on table, block clear

#### stack
Stacks one block on another.
- **Parameters**: robot, source block, target block
- **Precondition**: robot holding source, target clear
- **Effect**: source on target, source clear, robot empty

#### unstack
Removes a block from another block.
- **Parameters**: robot, source block, target block
- **Precondition**: robot empty, source on target, source clear
- **Effect**: robot holding source, target clear

#### put-in-container
Places a block into a container.
- **Parameters**: robot, block, container
- **Precondition**: robot holding block, container clear
- **Effect**: robot empty, block in container

## Example Problem File

```pddl
;; Problem generated from video: RAIL+d027f2ae+2023-06-05-16h-33m-01s
(define (problem RAIL_d027f2ae_2023_06_05_16h_33m_01s)
    (:domain robot-manipulation)
    
    (:objects
        block1 - block
        block2 - block
        robot1 - robot
    )
    
    (:init
        (empty robot1)
        (on-table block1)
        (clear block1)
        (on-table block2)
        (clear block2)
    )
    
    (:goal
        (and
            (on block2 block1)
        )
    )
)
```

## Validation Results

All 5 problem files have been validated against the domain file:

```
✅ problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl: VALID
✅ problem_AUTOLab_0d4edc83_2023_10_27_20h_25m_34s.pddl: VALID
✅ problem_GuptaLab_553d1bd5_2023_05_19_10h_36m_14s.pddl: VALID
✅ problem_RAD_c6cf6b42_2023_08_31_14h_00m_49s.pddl: VALID
✅ problem_TRI_52ca9b6a_2024_01_16_16h_43m_04s.pddl: VALID
```

## Using with PDDL Planners

These files can be used with any PDDL planner that supports STRIPS and typing. Examples:

### Fast-Forward (FF) Planner
```bash
ff -o domain.pddl -f problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl
```

### Fast Downward
```bash
fast-downward.py domain.pddl problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl --search "astar(lmcut())"
```

### Unified Planning Library
```python
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

reader = PDDLReader()
problem = reader.parse_problem('domain.pddl', 'problem_RAIL_d027f2ae_2023_06_05_16h_33m_01s.pddl')

with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    print(result.plan)
```

## Files Structure

```
assignment3/
├── raw_videos/                    # Input videos (5 folders)
│   ├── RAIL+d027f2ae+...
│   ├── AUTOLab+0d4edc83+...
│   ├── GuptaLab+553d1bd5+...
│   ├── RAD+c6cf6b42+...
│   └── TRI+52ca9b6a+...
├── frames/                        # Extracted video frames
├── generate_pddl.py              # Main generation script
├── domain.pddl                   # Generated domain file
├── problem_*.pddl                # Generated problem files (5)
└── README_PDDL_GENERATION.md     # This file
```

## Key Features

1. **Automated Analysis**: No manual annotation required
2. **VLM-Powered**: Uses state-of-the-art vision-language models
3. **Validated Output**: All files validated with Unified Planning
4. **Extensible**: Easy to add new actions or object types
5. **Well-Documented**: Comments include VLM analysis descriptions

## Limitations & Future Work

### Current Limitations
1. VLM descriptions may be generic (e.g., "what objects are on the table?")
2. Goal inference is based on simple heuristics
3. Limited to 2-3 blocks per problem
4. Container detection not fully implemented

### Potential Improvements
1. Use more specific prompts based on first-pass analysis
2. Implement multi-stage VLM reasoning
3. Use video-specific language annotations from DROID dataset
4. Implement more sophisticated goal inference
5. Add support for more complex object relationships
6. Use temporal models to better capture action sequences

## References

- [DROID Dataset](https://droid-dataset.github.io/)
- [BLIP Model](https://github.com/salesforce/BLIP)
- [CLIP Model](https://github.com/openai/CLIP)
- [Unified Planning Library](https://github.com/aiplan4eu/unified-planning)
- [PDDL Specification](https://planning.wiki/ref/pddl)

## Author

Generated for Robotics Assignment 3 - Fall 2025


