# Download and Process Block Episodes - Two-Step Workflow

This guide explains how to download block-related episodes from DROID and then process them separately.

## Overview

The workflow is split into two steps:

1. **Download Step** (`download_block_episodes.py`): Downloads episodes from DROID dataset to local directory
2. **Process Step** (`process_downloaded_episodes.py`): Processes locally downloaded episodes with VLM to generate PDDL

This approach allows you to:
- Download episodes once (requires network access)
- Process them multiple times without network access
- Share downloaded episodes with others
- Avoid repeated downloads

## Step 1: Download Block Episodes

Download the first 5 block-related episodes (videos) to your local directory:

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3

# Download 5 episodes (default)
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --output-dir data/block_episodes

# Or specify different number
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --output-dir data/block_episodes \
    --annotations droid_language_annotations.json
```

**What it does:**
- Loads filtered block episode IDs from `block_episode_ids.json`
- Scans DROID dataset to find matching episodes
- Downloads and saves each episode to:
  - `data/block_episodes/<episode_id>/`
    - `episode_info.json` - Episode metadata and instructions
    - `episode_data.pkl` - Full episode data (pickle format)
    - `frames/` - Extracted frame images (up to 50 frames per episode)

**Requirements:**
- Google Cloud authentication: `gcloud auth application-default login`
- TensorFlow and tensorflow-datasets installed
- Network access to Google Cloud Storage

**Output structure:**
```
data/block_episodes/
├── IPRL_c850f181_2023-06-18-22h-55m-15s/
│   ├── episode_info.json
│   ├── episode_data.pkl
│   └── frames/
│       ├── 0000.jpg
│       ├── 0001.jpg
│       └── ...
├── ILIAD_7ae1bcff_2023-05-30-18h-47m-45s/
│   └── ...
└── ... (5 episodes total)
```

## Step 2: Process Downloaded Episodes

Once episodes are downloaded, process them with VLM:

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3

# Process all downloaded episodes with LLaVA
python process_downloaded_episodes.py \
    --episodes-dir data/block_episodes \
    --llava \
    --prompts-file prompts.txt \
    --max-frames 8

# Or use OpenCLIP
python process_downloaded_episodes.py \
    --episodes-dir data/block_episodes \
    --openclip \
    --prompts-file prompts.txt
```

**What it does:**
- Scans `data/block_episodes/` for downloaded episodes
- For each episode with `frames/` directory:
  - Loads frames
  - Runs VLM analysis (LLaVA or OpenCLIP)
  - Saves analysis results to `results/<episode_id>/vlm_analysis.json`

**Requirements:**
- Downloaded episodes in local directory
- VLM model (LLaVA or OpenCLIP) installed
- Hugging Face token in `.env` (for LLaVA)

**Output structure:**
```
results/
├── IPRL_c850f181_2023-06-18-22h-55m-15s/
│   └── vlm_analysis.json
├── ILIAD_7ae1bcff_2023-05-30-18h-47m-45s/
│   └── vlm_analysis.json
└── ... (analysis for each episode)
```

## Step 3: Generate PDDL Files

After processing, generate PDDL files for each episode:

```bash
cd /workspaces/eng-ai-agents/assignments/assignment3

# For each episode, generate PDDL
python video_to_pddl.py \
    --frames-dir data/block_episodes/IPRL_c850f181_2023-06-18-22h-55m-15s/frames \
    --out-name episode1 \
    --prompts-file prompts.txt \
    --llava \
    --max-frames 8

python video_to_pddl.py \
    --frames-dir data/block_episodes/ILIAD_7ae1bcff_2023-05-30-18h-47m-45s/frames \
    --out-name episode2 \
    --prompts-file prompts.txt \
    --llava \
    --max-frames 8

# ... repeat for all 5 episodes
```

Or create a batch script:

```bash
#!/bin/bash
# process_all.sh

for episode_dir in data/block_episodes/*/; do
    episode_name=$(basename "$episode_dir")
    frames_dir="$episode_dir/frames"
    
    if [ -d "$frames_dir" ]; then
        echo "Processing $episode_name..."
        python video_to_pddl.py \
            --frames-dir "$frames_dir" \
            --out-name "$episode_name" \
            --prompts-file prompts.txt \
            --llava \
            --max-frames 8
    fi
done
```

## Complete Workflow Example

```bash
# 1. Download 5 block episodes
python download_block_episodes.py \
    --episode-ids block_episode_ids.json \
    --max-episodes 5 \
    --output-dir data/block_episodes \
    --annotations droid_language_annotations.json

# 2. Process downloaded episodes with VLM
python process_downloaded_episodes.py \
    --episodes-dir data/block_episodes \
    --llava \
    --prompts-file prompts.txt \
    --max-frames 8

# 3. Generate PDDL for each episode
for episode_dir in data/block_episodes/*/; do
    episode_name=$(basename "$episode_dir")
    python video_to_pddl.py \
        --frames-dir "$episode_dir/frames" \
        --out-name "$episode_name" \
        --prompts-file prompts.txt \
        --llava \
        --max-frames 8
done
```

## Troubleshooting

### Download Issues

**Error: Authentication failed**
```bash
gcloud auth application-default login
```

**Error: Network timeout**
- The dataset is large, try again later
- Check your internet connection
- Episodes are downloaded sequentially, be patient

**Error: Episodes not found**
- The script matches by language instructions
- Make sure `--annotations` points to the correct file
- Some episodes might not be in the dataset

### Processing Issues

**Error: No frames directory found**
- Re-run the download script
- Check that frames were extracted during download

**Error: LLaVA model not found**
- Set `HF_TOKEN` in `.env` file
- Or run: `huggingface-cli login`

**Error: GPU not available**
- Script will fall back to CPU (slower but works)
- LLaVA processing may take longer on CPU

## File Sizes

**Downloaded episodes:**
- Each episode: ~10-50 MB (depends on number of frames)
- 5 episodes: ~50-250 MB total

**Processed results:**
- Each analysis: ~1-5 MB (JSON files)
- Total: ~5-25 MB for 5 episodes

## Summary

- **Download once**: Requires network, Google Cloud auth
- **Process multiple times**: No network needed, uses local files
- **Each episode = 1 video** demonstration of robot manipulating blocks
- **5 episodes = 5 videos** ready for PDDL generation


