# Multimodal PDDL Generation Summary

## Overview
This script uses a **completely new multimodal approach** that combines:
1. **Visual Language Models (VLMs)**: BLIP for image captioning + CLIP for object detection
2. **Text Annotations**: Language instructions from droid_language_annotations.json

## Key Features

### 1. Vision Analysis (BLIP + CLIP)
- **BLIP**: Generates natural language captions for video frames
- **CLIP**: Detects specific objects (blocks, containers, robot) with confidence scores
- Analyzes initial, final, and middle frames to understand state changes

### 2. Text Understanding
- Extracts language instructions from annotations
- Parses instructions to identify:
  - Action types (put, remove, stack, place)
  - Objects involved (blocks, containers)
  - Goal locations (in container, on table, on block)

### 3. Multimodal Fusion
Combines vision + text to generate accurate PDDL:
- Vision confirms object presence and types
- Text provides task intent and goals
- Together they determine initial states and goal states

## Generated PDDL Files

### Video 1: AUTOLab (Put in Container)
**Instruction**: "Put the green block into the black bowl."
**Vision Detected**: Bowl (0.31 confidence), green block
**Task**: put_in_container
**Initial State**: Blocks on table, bowl clear
**Goal**: `(in block1 bowl1)`

### Video 2: GuptaLab (Stack Blocks)
**Instruction**: "Put the orange block on the green block"
**Vision Detected**: Green (0.21), orange (0.17), red (0.19) blocks
**Task**: stack_two
**Initial State**: Both blocks on table
**Goal**: `(on block2 block1)`

### Video 3: RAD (Remove from Container)
**Instruction**: "Remove the green block from the bowl and put it on the table"
**Vision Detected**: Bowl mentioned in text, robot arm visible
**Task**: remove_from_container
**Initial State**: `(in block1 bowl1)` - Block starts in bowl!
**Goal**: `(on-table block1)`

### Video 4: RAIL (Stack Blocks)
**Instruction**: "Put the orange block on the green block"
**Vision Detected**: Multiple colored blocks
**Task**: stack_two
**Goal**: `(on block2 block1)`

### Video 5: TRI (Stack Blocks)
**Instruction**: "Pick up the orange rectangular block and place it horizontally on the green cylindrical block"
**Vision Detected**: Robot gripper, blocks
**Task**: stack_two
**Goal**: `(on block2 block1)`

## Domain File

Defines 6 actions:
1. **pick-up**: Pick block from table
2. **put-down**: Place block on table
3. **stack**: Stack block on another block
4. **unstack**: Remove block from stack
5. **put-in-container**: Place block in container
6. **take-from-container**: Remove block from container

## Validation Results
✅ All 5 problems validated successfully with unified-planning!

## Technical Implementation

### VLM Models Used:
- **BLIP**: `Salesforce/blip-image-captioning-base`
- **CLIP**: `ViT-B/32`

### Frame Extraction:
- Uses FFmpeg at 10 FPS
- Analyzes initial, final, and 2 middle frames

### Task Detection Logic:
1. Check for removal tasks (remove, take out) - most specific
2. Check for container tasks (put into, place in)
3. Check for stacking tasks (stack, on top, put X on Y)
4. Fall back to vision-based inference

### Multimodal Fusion:
- Text provides primary intent
- Vision confirms object presence and validates task feasibility
- Initial states adapt based on task type (e.g., block in container for removal)
