# Assignment 3: Video2Plan: Learning Domain and Problem Representation

**Name:** Sohaib Chachar

**Date:** 11/09/2025

## Overview
This repository contains scripts for processing videos to generate PDDL domain and problem representations for planning tasks.

## Scripts
- **download_specified_videos.py**: Uses `gsutil` to list and download full episode recordings from the DROID dataset based on `episode_id_to_path.json`, storing them under [raw_videos](raw_videos).
- **extract_video_descriptions.py**: Runs the NVIDIA Cosmos Reason-1 7B vision-language model on each episode video, enforcing a structured prompt to narrate robot manipulation steps, writes human-readable summaries plus parsed JSON to the [video_descriptions](video_descriptions) folder.
- **generate_pddl_from_descriptions.py**: Loads all extracted descriptions and uses the OpenAI `gpt-4o` reasoning model to derive both a unified domain and per-episode problem files, writing the results to [`domain.pddl`](domain.pddl) and the [problems](problems) directory.

## Usage
1. `python download_specified_videos.py`
   - Four DROID episode IDs are predefined inside the script. Running it downloads every MP4 available for each episode (all camera viewpoints) into `raw_videos/episode_id/recordings/MP4/`.
   - After download, keep a single representative MP4 per episode (the others were manually deleted in this workspace).
2. `python extract_video_descriptions.py`
   - Requires an NVIDIA GPU. Generates text summaries and structured JSON for each remaining MP4, saving them under `video_descriptions/` and updating `video_descriptions/all_descriptions.json`.
3. `python generate_pddl_from_descriptions.py`
   - Reads all descriptions, invokes the OpenAI `gpt-4o` reasoning model, and writes the [`domain.pddl`](domain.pddl) plus per-episode problems in [problems](problems).

## Domain and Problem Files
- The PDDL domain definition is available in [domain.pddl](domain.pddl).
- Problem files are in [problems](problems).

## Other Folders and Files
- Videos are downloaded in [raw_videos](raw_videos).
- Additional data files include `droid_language_annotations.json`, `episode_id_to_path.json`, and `block_episode_ids.json`.
