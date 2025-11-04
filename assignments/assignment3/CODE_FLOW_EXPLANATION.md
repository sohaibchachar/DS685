# Code Flow Explanation: PDDL Generation with Cosmos-Reason1-7B

## Overview
This script generates PDDL (Planning Domain Definition Language) domain and problem files by analyzing robot manipulation videos using Cosmos-Reason1-7B, a vision-language model specialized for sequential reasoning about physical tasks.

## High-Level Flow

```
1. Load Model → 2. Load Annotations → 3. Analyze All Videos → 4. Generate Domain → 5. Generate Problems
```

## Detailed Flow

### STEP 1: Model Loading (`load_cosmos_model()`)
- **Purpose**: Load Cosmos-Reason1-7B model for sequential video understanding
- **Process**:
  - Loads Qwen2.5-VL processor and model from HuggingFace (`nvidia/Cosmos-Reason1-7B`)
  - Uses 4-bit quantization to reduce memory usage
  - Configures for CUDA device (GPU) if available
  - Returns: `processor`, `model`, `device`

### STEP 2: Load Annotations (`load_annotations()`)
- **Purpose**: Load human-provided language instructions for each video episode
- **Process**:
  - Reads `droid_language_annotations.json`
  - Handles Git LFS headers and merge conflicts
  - Returns: Dictionary mapping episode IDs to instructions

### STEP 3: Analyze All Videos (`analyze_sequential_video()`)
- **Purpose**: Extract initial state, final state, and action sequence from each video
- **Process**:
  For each video:
  1. **Load video** using Qwen2.5-VL processor
  2. **Create prompt** asking Cosmos to analyze:
     - Initial state: What objects are present and where they are at the start
     - Sequence of actions: What happens in the video
     - Final state: Where objects end up
  3. **Generate analysis** using Cosmos model
  4. **Parse JSON response** from Cosmos (handles malformed JSON)
  5. Returns: Dictionary with `initial_state`, `final_state`, `sequence_of_actions`, etc.

**Key Function**: `analyze_sequential_video(processor, model, device, video_path, instruction)`
- Uses video input (not individual frames) for sequential understanding
- Extracts structured JSON about object states and actions

### STEP 4: Generate Domain from Analyses (`generate_domain_from_analyses()`)
- **Purpose**: Create a unified PDDL domain file based on patterns discovered across all videos
- **Process**:
  1. **Extract common patterns** from all analyses:
     - **Types**: Object types (block, container, robot) discovered across videos
     - **Predicates**: Relationships (on, in, clear, holding, etc.) used in states
     - **Actions**: Action patterns (pick-up, place, stack, etc.) from action sequences
  
  2. **Extract domain elements**:
     - `extract_objects_and_predicates(analysis)`: Gets types and predicates from each analysis
     - `induce_actions_from_analysis(analysis)`: Extracts action patterns from sequence_of_actions
  
  3. **Combine and standardize**:
     - Merges types/predicates/actions from all videos
     - Adds standard predicates/actions needed for robot manipulation
     - Removes duplicates
  
  4. **Generate PDDL domain file**:
     - Formats types: `block - object`, `container - object`, `robot - object`
     - Formats predicates: `(on ?x - block ?y - block)`, `(in ?x - block ?c - container)`, etc.
     - Formats actions: Complete PDDL action definitions with preconditions and effects

**Why generate domain from analyses?**
- **Discoverability**: We don't know what types/predicates/actions are needed until we see the videos
- **Flexibility**: Domain adapts to what's actually in the videos
- **Completeness**: Ensures domain covers all scenarios seen in videos

### STEP 5: Generate Problem Files (`generate_problem_pddl()`)
- **Purpose**: Create individual problem.pddl files for each video
- **Process**:
  1. **Parse initial state** from analysis:
     - Extract objects from `initial_state.objects` array
     - Create color-based names (e.g., `green_block`, `blue_cup`)
     - Extract initial predicates from object locations:
       - `(on-table green_block)` if location is "on table"
       - `(in green_block white_bowl)` if location is "in white_bowl"
       - `(on orange_block green_block)` if location is "on green_block"
  
  2. **Parse final state** from analysis:
     - Extract objects from `final_state.objects` array
     - Extract goal predicates from final locations
  
  3. **Filter objects**:
     - Only include manipulatable objects (blocks, containers)
     - Exclude surfaces/tables/straws/mugs (not in objects section)
     - Filter predicates to only reference declared objects
  
  4. **Generate PDDL problem file**:
     - `(:objects)`: List of objects with types (e.g., `green_block - block`)
     - `(:init)`: Initial state predicates
     - `(:goal)`: Goal state predicates

## Key Functions

### `extract_objects_and_predicates(analysis)`
- Extracts object types and predicates from Cosmos analysis
- Looks for `object_types` and `predicates` fields
- Infers from `initial_state` structure
- Returns: `(types_list, predicates_list)`

### `induce_actions_from_analysis(analysis)`
- Extracts action patterns from `sequence_of_actions`
- Maps action names to PDDL action templates:
  - "pick up" → `pick-up` action
  - "place" → `place` action
  - "stack" → `stack` action
- Returns: List of PDDL action strings

### `generate_problem_pddl(video_id, instruction, analysis, output_dir)`
- Generates problem.pddl for a single video
- Uses color-based object names (e.g., `green_block` not `block1`)
- Only includes objects in `(:objects)` section (no surfaces)
- Filters predicates to only reference declared objects

## Data Flow Diagram

```
Video Files
    ↓
Cosmos-Reason1-7B Analysis
    ↓
JSON Analysis (per video):
  {
    "initial_state": {"objects": [...], "robot_holding": null},
    "final_state": {"objects": [...]},
    "sequence_of_actions": [...],
    "object_types": [...],
    "predicates": [...]
  }
    ↓
┌─────────────────────────────────────┐
│  Aggregate All Analyses            │
│  - Extract types                   │
│  - Extract predicates              │
│  - Extract actions                 │
└─────────────────────────────────────┘
    ↓
Domain.pddl (unified domain)
    ↓
┌─────────────────────────────────────┐
│  Generate Problem Files            │
│  (one per video)                   │
│  - Objects from initial_state      │
│  - Init from initial_state         │
│  - Goal from final_state           │
└─────────────────────────────────────┘
    ↓
Problem Files (one per video)
```

## Why This Approach?

1. **Domain Generation from Analysis**:
   - ✅ Discovers what's actually needed from videos
   - ✅ Adapts to different video types
   - ✅ More accurate than hardcoding

2. **Sequential Video Understanding**:
   - ✅ Uses full video (not just frames)
   - ✅ Understands temporal relationships
   - ✅ Captures state transitions

3. **Color-Based Naming**:
   - ✅ More descriptive (`green_block` vs `block1`)
   - ✅ Matches video descriptions
   - ✅ Easier to understand

4. **Filtering**:
   - ✅ Only manipulatable objects in `(:objects)`
   - ✅ Surfaces/tables excluded (they're implicit)
   - ✅ Predicates only reference declared objects

## Example Flow

**Input**: Video showing "Put yellow block in blue cup"

**Step 3 Analysis**:
```json
{
  "initial_state": {
    "objects": [
      {"name": "yellow_block", "location": "on table", "color": "yellow"},
      {"name": "blue_cup", "location": "on table", "color": "blue"}
    ]
  },
  "final_state": {
    "objects": [
      {"name": "yellow_block", "location": "in blue_cup", "color": "yellow"},
      {"name": "blue_cup", "location": "on table", "color": "blue"}
    ]
  },
  "sequence_of_actions": ["pick up yellow_block", "place yellow_block in blue_cup"]
}
```

**Step 4 Domain** (aggregated from all videos):
```pddl
(:types block container robot - object)
(:predicates
  (on-table ?x - block)
  (in ?x - block ?c - container)
  ...
)
(:action put-in-container ...)
```

**Step 5 Problem**:
```pddl
(:objects yellow_block - block blue_cup - container robot1 - robot)
(:init
  (empty robot1)
  (on-table yellow_block)
  (on-table blue_cup)
)
(:goal (in yellow_block blue_cup))
```

