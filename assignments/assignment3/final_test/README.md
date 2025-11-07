# Final Test - Video Description to PDDL Pipeline

This folder contains a streamlined version of the video description extraction and PDDL generation pipeline. All video descriptions are saved to a single `video_descriptions.txt` file.

**Key Feature**: This version analyzes the **full video** directly (not frames) using the Cosmos-Reason1-7B model for more accurate descriptions.

## Files

- **`extract_single_video.py`** - Extracts description from a single full video and appends it to `video_descriptions.txt`
- **`process_all_videos.py`** - Processes all videos in `../raw_videos/` by calling `extract_single_video.py` for each one
- **`generate_pddl_from_descriptions.py`** - Generates PDDL domain and problem files from the descriptions in `video_descriptions.txt`

## Usage

### Step 1: Extract Video Descriptions

Process all videos in the `raw_videos` folder:

```bash
python process_all_videos.py
```

This will:
1. Find all MP4 files in `../raw_videos/*/recordings/MP4/`
2. Call `extract_single_video.py` for each video
3. Append all descriptions to `video_descriptions.txt`

You can also process a single video manually:

```bash
python extract_single_video.py ../raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4
```

### Step 2: Generate PDDL Files

Once all descriptions are collected in `video_descriptions.txt`, generate PDDL files:

```bash
python generate_pddl_from_descriptions.py
```

This requires the `OPENAI_API_KEY` environment variable to be set (or in a `.env` file in the workspace root).

Output:
- `domain.pddl` - PDDL domain file covering all scenarios
- `problems/*.pddl` - Individual problem files for each video

## Requirements

- Python 3.8+
- CUDA-enabled GPU (for video extraction)
- PyTorch with CUDA support
- transformers
- opencv-python (cv2)
- openai
- python-dotenv (optional)

## Output Format

### video_descriptions.txt

```
================================================================================
Episode ID: RAD+c6cf6b42+2023-08-31-14h-00m-49s
Video Path: ../raw_videos/RAD+c6cf6b42+2023-08-31-14h-00m-49s/recordings/MP4/32907025.mp4
================================================================================
{
  "initial_state": [
    {"object": "red_block", "location": "on table"},
    ...
  ],
  ...
}

================================================================================
Episode ID: ...
...
```

## Notes

- The extraction process uses the Cosmos-Reason1-7B model for analyzing **full videos** directly
- No frame extraction is performed - the model processes the entire video
- The model is loaded fresh for each video to ensure GPU memory is properly managed
- All descriptions are appended to a single file for easier processing
- This approach provides more context and potentially more accurate descriptions than frame-based analysis

