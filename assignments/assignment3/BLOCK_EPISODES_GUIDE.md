# Block Episodes Processing Guide

This guide explains how to filter and process block-related episodes from the DROID dataset.

## Overview

The DROID dataset contains 75,144 episodes, but only a subset are related to block manipulation tasks. This pipeline:

1. **Filters** episodes by analyzing language instructions in `droid_language_annotations.json`
2. **Matches** filtered episodes in the DROID dataset using language instructions
3. **Processes** episodes with VLM (LLaVA or OpenCLIP) to extract domain knowledge
4. **Generates** PDDL domain and problem files

## Results

From `droid_language_annotations.json`:
- **Total episodes**: 75,144
- **Block episodes found**: 6,881 (9.16%)
- **Output file**: `block_episode_ids.json`

## Step 1: Filter Block Episodes

Run the filter script to identify block-related episodes:

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3

# Default: filter with standard block keywords
python filter_block_episodes.py

# Custom keywords
python filter_block_episodes.py --keywords block cube stack brick

# Custom output file
python filter_block_episodes.py --output my_blocks.json
```

**Output**: `block_episode_ids.json` (or custom filename) containing a list of episode IDs.

**Keywords used by default**:
- block, blocks
- cube, cubes
- brick, bricks
- stack, stacking
- tower
- blocks world

## Step 2: Process Episodes with VLM

Once you have the filtered episode IDs, process them with LLaVA or OpenCLIP:

```bash
# Process with LLaVA (default, recommended)
python process_droid_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 10 \
    --llava \
    --prompts-file prompts.txt \
    --max-frames 8

# Process with OpenCLIP
python process_droid_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 10 \
    --openclip \
    --prompts-file prompts.txt

# Use local TFDS directory (if you downloaded episodes)
python process_droid_episodes.py \
    --episode-ids block_episode_ids.json \
    --tfds-dir /path/to/local/droid/data \
    --llava \
    --max-episodes 5
```

### Matching Strategy

The script uses two matching strategies:

1. **Language Instruction Matching** (more accurate, default):
   - Compares language instructions from the dataset with annotations
   - Uses fuzzy matching to handle variations
   - Requires `--annotations droid_language_annotations.json`

2. **Episode ID Matching** (fallback):
   - Tries to match by episode ID
   - Less accurate if episode IDs don't exactly match

### What It Does

For each matched episode:
1. Extracts frames from video data
2. Runs VLM analysis (LLaVA captioning or OpenCLIP scoring)
3. Saves frames and analysis results to `results/<episode_id>/`
4. Can generate PDDL files (if integrated with PDDL generator)

## Example Workflow

```bash
# 1. Filter for block episodes
python filter_block_episodes.py

# 2. Process first 5 episodes (for testing)
python process_droid_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --llava \
    --prompts-file prompts.txt \
    --annotations droid_language_annotations.json

# 3. Check results
ls -la results/
```

## Requirements

### Authentication

Accessing the DROID dataset requires Google Cloud authentication:

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# The dataset is publicly accessible but requires authentication
```

### Dependencies

```bash
pip install tensorflow-datasets transformers torch pillow numpy
pip install open-clip-torch  # For OpenCLIP option
```

### Environment Variables

The script loads Hugging Face token from `.env`:
```
HF_TOKEN=your_token_here
```

Or set it in the environment:
```bash
export HF_TOKEN=your_token_here
```

## File Structure

```
assignments/assignment3/
├── filter_block_episodes.py          # Step 1: Filter episodes
├── process_droid_episodes.py          # Step 2: Process episodes
├── video_to_pddl.py                  # Generate PDDL from frames
├── droid_language_annotations.json    # Source annotations (20MB)
├── block_episode_ids.json             # Filtered episode IDs (generated)
├── prompts.txt                        # VLM prompts
└── results/                           # Output directory
    └── <episode_id>/
        └── frames/                    # Extracted frames
```

## Troubleshooting

### Authentication Errors

**Error**: `AccessDeniedException` or `403 Forbidden`

**Solution**:
```bash
gcloud auth application-default login
```

### Model Download Errors

**Error**: `401 Client Error: Unauthorized`

**Solution**: Set Hugging Face token:
```bash
export HF_TOKEN=your_token_here
# Or add to .env file
```

### No Episodes Found

The dataset is large (76k episodes). The script scans sequentially, so:
- It may take time to find matches
- Progress is shown every 1000 episodes scanned
- Use `--max-episodes` to limit for testing

### Performance Tips

1. **Limit episodes for testing**: Use `--max-episodes 5` initially
2. **Use local TFDS**: Download episodes locally first if processing many
3. **GPU required**: LLaVA processing is much faster on GPU
4. **Limit frames**: Use `--max-frames 4` for faster processing

## Next Steps

After processing episodes:

1. **Review extracted frames**: Check `results/<episode_id>/frames/`
2. **Integrate with PDDL generation**: Use `video_to_pddl.py` on extracted frames
3. **Generate domain files**: Aggregate knowledge from multiple episodes
4. **Validate PDDL**: Use Unified Planning library

## Citation

If you use the DROID dataset, please cite:

```bibtex
@article{khazatsky2024droid,
    title   = {DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
    author  = {Khazatsky, Alexander and Wang, Yufei and Chebotar, Yevgen and ...},
    journal = {arXiv preprint arXiv:2403.12945},
    year    = {2024}
}
```

## Resources

- [DROID Dataset Website](https://droid-dataset.github.io/)
- [DROID Paper](https://arxiv.org/abs/2403.12945)
- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets)


