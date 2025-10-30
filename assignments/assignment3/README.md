# Assignment 3: Video2Plan - Learning Domain and Problem Representation from DROID

## Overview

Generate PDDL domain and problem files from the DROID robot manipulation dataset using Visual Language Models (VLMs) to extract domain knowledge.

## Objective

Given the DROID dataset with robot manipulation demonstrations, create:
1. **PDDL Domain File** - Define types, predicates, and actions
2. **PDDL Problem Files** - Generate at least 3 problem files with different initial/goal states

## Dataset

- **DROID**: Large-scale in-the-wild robot manipulation dataset
- 76k demonstration trajectories (350h of interaction data)
- 564 scenes, 86 tasks across diverse environments
- [DROID Dataset Website](https://droid-dataset.github.io/)

### Loading DROID Data

```python
import tensorflow_datasets as tfds

ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")

for episode in ds.take(5):
    for step in episode["steps"]:
        image = step["observation"]["exterior_image_1_left"]
        wrist_image = step["observation"]["wrist_image_left"]
        action = step["action"]
        instruction = step["language_instruction"]
```

## Approach

### 1. Use Visual Language Models (VLMs)
- **CLIP**: Extract visual-semantic features from images
- **BLIP**: Image captioning and visual question answering
- **Other VLMs**: GPT-4V, LLaVA, etc.

### 2. Automated PDDL Generation
Use VLM to:
- Analyze demonstration videos/images
- Extract object types and properties
- Identify actions and their preconditions/effects
- Generate PDDL domain specification
- Create problem files with different scenarios

### 3. Verification with Unified Planning
- Use Unified Planning library for PDDL validation
- Automated syntax checking
- Plan validation

## Requirements

### Tools
- **VS Code PDDL Extension**: [Install](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl)
- **Unified Planning Library**: For PDDL I/O and validation
- **TensorFlow Datasets**: For DROID data loading
- **VLMs**: CLIP, BLIP-2, or similar

### Deliverables
1. PDDL domain file (`domain.pddl`)
2. At least 3 problem files (`problem1.pddl`, `problem2.pddl`, `problem3.pddl`)
3. Python script for automated PDDL generation from DROID data
4. README documenting the approach

## Project Structure

```
assignment3/
├── README.md
├── domain.pddl
├── problem1.pddl
├── problem2.pddl
├── problem3.pddl
├── pddl_generator.py
├── requirements.txt
└── results/
```

## Getting Started

### 1. Install Dependencies

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3
pip install tensorflow-datasets transformers clip-by-openai torch
```

### 2. Load DROID Data

```python
import tensorflow_datasets as tfds
ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")
```

### 3. Extract Domain Knowledge

Use VLMs to analyze demonstrations and extract:
- Object types (graspable objects, containers, tools)
- Predicates (on, in, holding, etc.)
- Actions (pick, place, open, close, etc.)

### 4. Generate PDDL Files

Automatically generate domain and problem files based on extracted knowledge.

## Evaluation

- Correctness of PDDL syntax
- Completeness of domain model
- Diversity of problem files
- Automation quality of generation pipeline

## References

- [DROID Dataset](https://droid-dataset.github.io/)
- [PDDL Specification](https://github.com/AI-Planning/pddl)
- [Unified Planning](https://www.unified-planning.org/)
- [VS Code PDDL Extension](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl)
