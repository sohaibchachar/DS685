# DROID Dataset Setup Guide

## Overview

This guide explains how to access the [DROID dataset](https://droid-dataset.github.io/) and filter for block-related episodes to generate PDDL files.

## Quick Start

### 1. Install Google Cloud SDK

The DROID dataset is stored in Google Cloud Storage (`gs://gresearch/robotics`) and requires authentication.

**Install Google Cloud SDK:**

```bash
# On Ubuntu/Debian
curl https://sdk.cloud.google.com | bash

# On macOS with Homebrew
brew install --cask google-cloud-sdk

# Or follow official guide:
# https://cloud.google.com/sdk/docs/install
```

### 2. Authenticate with Google Cloud

```bash
# Authenticate for application default credentials
gcloud auth application-default login

# This will open a browser window for authentication
# The dataset is publicly accessible but requires authentication
```

### 3. (Optional) Set Project

```bash
# Set your project (if you have one)
gcloud config set project YOUR_PROJECT_ID
```

Note: The DROID dataset is **publicly accessible**, so you don't need a specific project. Authentication is sufficient.

## Accessing DROID Dataset

### Python Code

```python
import tensorflow_datasets as tfds

# Load DROID dataset
ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")

# Iterate through episodes
for episode in ds.take(5):
    for step in episode["steps"]:
        image = step["observation"]["exterior_image_1_left"]
        wrist_image = step["observation"]["wrist_image_left"]
        action = step["action"]
        instruction = step["language_instruction"]
```

### Dataset Structure

According to the [DROID website](https://droid-dataset.github.io/):

- **76k demonstration trajectories** (350h of interaction data)
- **564 scenes** across diverse environments
- **86 tasks** with natural language instructions
- Data collected across **North America, Asia, and Europe**

## Filtering for Blocks

### Using the Generator Script

The `pddl_generator.py` script includes built-in filtering for block-related episodes:

```bash
# Filter for BLOCK episodes only
python pddl_generator.py --blocks --episodes 20

# This will:
# 1. Scan episodes from DROID dataset
# 2. Filter for instructions containing: "block", "blocks", "cube", "brick", "stack", "tower"
# 3. Extract domain knowledge from block manipulation tasks
# 4. Generate PDDL domain and problem files
```

### Block Keywords

The script filters for episodes containing these keywords:
- `block`, `blocks`
- `cube`, `cubes`
- `brick`, `bricks`
- `stack`, `stacking`
- `tower`
- `blocks world`

### Manual Filtering Example

```python
import tensorflow_datasets as tfds

ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")

block_keywords = {"block", "blocks", "cube", "stack", "tower"}
block_episodes = []

for episode in ds:
    episode_data = episode.numpy()
    if "steps" in episode_data:
        for step in episode_data["steps"]:
            if "language_instruction" in step:
                instruction = step["language_instruction"][0].decode('utf-8')
                if any(keyword in instruction.lower() for keyword in block_keywords):
                    block_episodes.append(episode)
                    break  # Found block episode, move to next
```

## Troubleshooting

### Authentication Errors

**Error:** `AccessDeniedException` or `403 Forbidden`

**Solution:**
```bash
# Re-authenticate
gcloud auth application-default login

# Check current credentials
gcloud auth application-default print-access-token
```

### Stores Not Found

**Error:** `No module named 'tensorflow'` or `TensorFlow not found`

**Solution:**
```bash
# Install TensorFlow
pip install tensorflow

# Or install tensorflow-datasets separately
pip install tensorflow-datasets
```

### Slow Download

The dataset is large (76k episodes). Consider:

1. **Limit episodes analyzed:**
   ```bash
   python pddl_generator.py --blocks --episodes 10
   ```

2. **Use dataset streaming:**
   The TensorFlow Datasets library streams data, so you don't need to download everything.

3. **Filter early:**
   Using `--blocks` flag filters during iteration, skipping non-block episodes.

## Dataset Citation

If you use the DROID dataset, please cite:

```bibtex
@article{khazatsky2024droid,
    title   = {DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
    author  = {Alexander Khazatsky and Karl Pertsch and ...},
    year    = {2024},
}
```

## References

- [DROID Dataset Website](https://droid-dataset.github.io/)
- [DROID Paper](https://arxiv.org/abs/2403.12945)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

## Next Steps

1. ✅ Set up Google Cloud authentication
2. ✅ Run: `python pddl_generator.py --blocks --episodes 20`
3. ✅ Review generated PDDL files
4. ✅ Use VS Code PDDL extension to validate syntax

