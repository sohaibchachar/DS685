# Assignment 3: Usage Guide

## Overview

This project generates PDDL domain and problem files from the DROID robot manipulation dataset using Visual Language Models (VLMs) for automated knowledge extraction.

## Installation

### 1. Install Dependencies

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3
pip install -r requirements.txt
```

### 2. (Optional) Install TensorFlow for DROID Dataset

For accessing the real DROID dataset, install TensorFlow:

```bash
pip install tensorflow
```

### 3. (Optional) Install CLIP for VLM Analysis

For advanced image-based analysis using CLIP:

```bash
pip install clip-by-openai torch
```

**Note**: The generator works without TensorFlow or CLIP, using mock data for development and testing.

## Usage

### Basic Usage

Generate PDDL files using text-based analysis (no VLM):

```bash
python pddl_generator.py
```

### Advanced Usage with VLM

Generate PDDL files with CLIP-based visual analysis:

```bash
python pddl_generator.py --use-vlm
```

### Analyze More Episodes

Analyze more episodes from the dataset:

```bash
python pddl_generator.py --episodes 10
```

## Generated Files

After running the generator, you'll have:

1. **domain.pddl** - PDDL domain file containing:
   - Type definitions (robot, container, graspable, object)
   - Predicates (holding, at, inside, on, clear, etc.)
   - Actions (pick, place, open-container, close-container, put-in, take-out, stack, unstack)

2. **problem1.pddl** - Problem: Put apple and orange in a box
   - Goal: Place two items in a container

3. **problem2.pddl** - Problem: Open drawer and place cup inside
   - Goal: Open container and manipulate items

4. **problem3.pddl** - Problem: Stack blocks
   - Goal: Create a stack of three blocks

## Architecture

### Components

1. **VLMImageAnalyzer** - Analyzes images using CLIP to detect objects and actions
2. **DROIDDatasetLoader** - Loads and processes DROID dataset episodes
3. **PDDLDomainGenerator** - Generates PDDL domain specifications
4. **PDDLProblemGenerator** - Generates PDDL problem files

### Workflow

```
DROID Dataset → Episode Analysis → VLM/Text Extraction → 
Domain Knowledge → PDDL Generation → Validation
```

## Features

### Text-Based Analysis

Extracts information from language instructions:
- Action verbs: pick, place, put, move, open, close, stack, etc.
- Object types: containers, graspable objects
- Spatial relations: in, on, at, etc.

### VLM-Based Analysis (with CLIP)

When enabled (`--use-vlm`):
- Analyzes demonstration images
- Detects objects in scenes
- Identifies manipulative actions
- Extracts visual-semantic features

## Example Output

```
Episode 1:
  Instructions: 1
  Instruction: Put the apple in the box
  Extracted objects: {'container', 'graspable'}
  Extracted actions: {'put'}

Episode 2:
  Instructions: 1
  Instruction: Open the drawer and place the cup inside
  Extracted objects: {'container', 'graspable'}
  Extracted actions: {'place', 'open'}

Extraction Summary:
  Total objects detected: 2
  Total actions detected: 7
```

## Next Steps

### 1. Validate PDDL Files

Use the VS Code PDDL extension to validate syntax:

1. Install the [PDDL Extension](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl)
2. Open `domain.pddl` in VS Code
3. Check for syntax highlighting and validation

### 2. Use Unified Planning

```python
from unified_planning.io import PDDLReader

reader = PDDLReader()
domain = reader.parse_problem("domain.pddl", "problem1.pddl")
print(domain)
```

### 3. Access Real DROID Dataset

The DROID dataset requires Google Cloud credentials:

```bash
# Set up authentication
gcloud auth application-default login

# The dataset will be accessed from gs://gresearch/robotics
```

## Troubleshooting

### TensorFlow Not Found

**Error**: `No module named 'tensorflow'`

**Solution**: The script will automatically use mock data if TensorFlow is not installed.

### CLIP Installation Issues

**Error**: `No module named 'clip'`

**Solution**: Text-based analysis works without CLIP. To enable VLM analysis, install with `pip install clip-by-openai torch`.

### Google Cloud Authentication

**Error**: Access denied to `gs://gresearch/robotics`

**Solution**: The script falls back to mock data. For real dataset access, authenticate with `gcloud auth application-default login`.

## Project Structure

```
assignment3/
├── README.md                # Project overview
├── USAGE.md                 # This file
├── pddl_generator.py        # Main generator script
├── requirements.txt         # Dependencies
├── domain.pddl              # Generated PDDL domain
├── problem1.pddl            # Generated problem 1
├── problem2.pddl            # Generated problem 2
├── problem3.pddl            # Generated problem 3
└── results/                 # Results directory
```

## References

- [DROID Dataset](https://droid-dataset.github.io/)
- [PDDL Specification](https://github.com/AI-Planning/pddl)
- [Unified Planning](https://www.unified-planning.org/)
- [CLIP by OpenAI](https://openai.com/research/clip)
- [VS Code PDDL Extension](https://marketplace.visualstudio.com/items?itemName=jan-dolejsi.pddl)

