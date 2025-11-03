# DROID Calibration Example

This script demonstrates how to access DROID dataset episodes and extract calibration data (intrinsics/extrinsics).

**Source**: [Hugging Face - KarlP/droid - CalibrationExample.ipynb](https://huggingface.co/KarlP/droid/blob/main/CalibrationExample.ipynb)

## Overview

The script performs the following operations:

1. **Load DROID Dataset**: Loads the `droid_100` subset from Google Cloud Storage
2. **Load Calibration Files**: Loads camera intrinsics, extrinsics, episode mappings, and camera serials from JSON files
3. **Find Episodes**: Iterates through the dataset to find episodes with calibration data
4. **Extract Calibration**: Extracts camera-to-base extrinsics and camera intrinsics
5. **Transform Coordinates**: Converts gripper positions from base frame to camera frame and projects to pixel space
6. **Visualize**: Creates visualization showing the projected gripper position on camera images

## Files Created

- `CalibrationExample.ipynb`: Original Jupyter notebook (downloaded)
- `droid_calibration_example.py`: Python script version (converted from notebook)

## Dependencies

The script requires:
- `tensorflow_datasets`
- `numpy`
- `tqdm`
- `PIL` (Pillow)
- `matplotlib`
- `mediapy`
- `scipy`

## Calibration Files Required

The script expects the following JSON files in the DROID repository directory:
- `cam2base_extrinsics.json`: Camera-to-base extrinsic parameters
- `intrinsics.json`: Camera intrinsic parameters
- `episode_id_to_path.json`: Mapping from episode IDs to file paths
- `camera_serials.json`: Camera serial numbers mapping

## Usage

```bash
# Update path_to_droid_repo in the script
python droid_calibration_example.py
```

## Notes

- The script uses `gs://gresearch/robotics` as the data directory for loading TFDS data
- Requires Google Cloud authentication for accessing the GCS bucket
- The `path_to_droid_repo` variable should be set to the directory containing the calibration JSON files
- The visualization uses `mediapy.show_video()` to display the annotated images


