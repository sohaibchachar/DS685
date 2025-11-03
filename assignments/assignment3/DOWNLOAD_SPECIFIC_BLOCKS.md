# Download Specific Block Episodes

This script downloads videos for episodes matching the IDs in `specific_block_episode_ids.json` using the same logic as the CalibrationExample notebook.

## How It Works

The script uses the calibration notebook's episode matching logic:

1. **Loads episode_id_to_path.json**: Maps episode IDs to episode paths
2. **Extracts episode path from metadata**: Uses the same logic as Cell 3 in the notebook:
   ```python
   file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
   episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
   ```
3. **Matches episodes**: Looks up the episode_path in the mapping to get the episode_id
4. **Downloads matching episodes**: Saves frames and metadata for matched episodes

## Usage

### Basic Usage (download all 172 episodes)

```bash
python download_specific_blocks.py
```

### Download First 5 Episodes (for testing)

```bash
python download_specific_blocks.py --max-episodes 5
```

### Custom Output Directory

```bash
python download_specific_blocks.py --output-dir data/my_blocks
```

### Use Local TFDS Directory

```bash
python download_specific_blocks.py --tfds-dir /path/to/droid_100
```

### Custom Episode IDs File

```bash
python download_specific_blocks.py --episode-ids my_episode_ids.json
```

## Command Line Options

- `--episode-ids`: JSON file containing episode IDs (default: `specific_block_episode_ids.json`)
- `--output-dir`: Output directory for downloaded episodes (default: `data/specific_blocks`)
- `--max-episodes`: Maximum number of episodes to download (default: all)
- `--tfds-dir`: Local TFDS data directory (default: use GCS)
- `--repo-path`: Path to DROID repository with calibration files (default: current directory)

## Prerequisites

1. **Calibration files** must be in the current directory:
   - `episode_id_to_path.json` (already downloaded)
   - Other calibration files (already downloaded)

2. **Google Cloud authentication** (if using GCS):
   ```bash
   gcloud auth login
   ```

3. **Python packages**:
   ```bash
   pip install tensorflow-datasets tqdm Pillow numpy
   ```

## Output Structure

Each downloaded episode is saved in its own directory:

```
data/specific_blocks/
├── AUTOLab+0d4edc83+2023-10-27-20h-25m-34s/
│   ├── episode_info.json      # Episode metadata and language instruction
│   └── frames/
│       ├── 0000.jpg           # Extracted frames from video
│       ├── 0001.jpg
│       └── ...
├── AUTOLab+0d4edc83+2023-10-27-20h-43m-06s/
│   └── ...
└── ...
```

## Example Output

```
======================================================================
Download Specific Block Episodes
======================================================================
📁 Output directory: data/specific_blocks
📁 Repository path: /workspaces/eng-ai-agents/assignments/assignment3
📋 Episode IDs file: specific_block_episode_ids.json
🔢 Max episodes to download: 5

📋 Loading episode IDs from specific_block_episode_ids.json...
   ✅ Loaded 172 episode IDs
📋 Loading episode ID mapping...
   ✅ Loaded 74795 episode path mappings

📦 Loading DROID dataset...
   ✅ Loaded from Google Cloud Storage

🔍 Searching for 172 target episodes in dataset...

🎯 Found episode: AUTOLab+0d4edc83+2023-10-27-20h-25m-34s (1/172)
    💾 Saved episode to: data/specific_blocks/AUTOLab+0d4edc83+2023-10-27-20h-25m-34s
       - 156 frames extracted
       - Instruction: Put the green block into the black bowl...
   ✅ Saved successfully (1 total)

...

======================================================================
Download Summary
======================================================================
📋 Target episodes: 172
✅ Found episodes: 5
💾 Saved episodes: 5
📁 Output directory: data/specific_blocks
```

## Notes

- The script uses the **exact same matching logic** as the CalibrationExample notebook
- Episodes are matched by extracting the episode path from metadata and looking it up in `episode_id_to_path.json`
- Frames are extracted from available camera streams (exterior_image_1, exterior_image_2, wrist_image)
- Up to 200 frames are saved per episode to manage disk space
- The script handles network timeouts and continues processing other episodes


