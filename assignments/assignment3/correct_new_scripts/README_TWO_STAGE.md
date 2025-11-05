# Two-Stage PDDL Generation Pipeline

This pipeline separates video understanding from PDDL generation:

1. **Stage 1**: Extract text descriptions from videos using Cosmos-Reason1-7B (Vision-Language Model)
2. **Stage 2**: Generate PDDL files from descriptions using a text LLM

## Benefits

- **Better separation of concerns**: Vision model handles video understanding, text LLM handles structured output
- **More reliable**: Text LLMs are better at generating structured PDDL syntax
- **Easier debugging**: Can review intermediate descriptions before PDDL generation
- **Flexible**: Can use different LLM backends (local, OpenAI, Anthropic)

## Stage 1: Extract Video Descriptions

Uses Cosmos-Reason1-7B to analyze videos and extract natural language descriptions.

```bash
cd assignments/assignment3
python correct_new_scripts/extract_video_descriptions.py
```

**Output:**
- `correct_new_scripts/video_descriptions/` - Individual description files (`.txt`)
- `correct_new_scripts/video_descriptions/all_descriptions.json` - Combined JSON file

## Stage 2: Generate PDDL from Descriptions

Uses an LLM to generate PDDL domain and problem files from the descriptions.

### LLM Backend Options

#### Option 1: Local LLM (Default)
Uses Phi-3-mini (or another local model) - no API keys needed.

```bash
python correct_new_scripts/generate_pddl_from_descriptions.py
```

Or explicitly set:
```bash
export LLM_BACKEND=local
python correct_new_scripts/generate_pddl_from_descriptions.py
```

#### Option 2: OpenAI API
Uses GPT-4o-mini or GPT-4o (better quality but costs money).

```bash
export LLM_BACKEND=openai
export OPENAI_API_KEY=your-key-here
python correct_new_scripts/generate_pddl_from_descriptions.py
```

#### Option 3: Anthropic API
Uses Claude 3.5 Sonnet (high quality, costs money).

```bash
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=your-key-here
python correct_new_scripts/generate_pddl_from_descriptions.py
```

**Output:**
- `correct_new_scripts/domain.pddl` - Domain file (generated once)
- `correct_new_scripts/problems/` - Individual problem files (one per video)

## Configuration

Edit the scripts to change:
- `VIDEO_DIR` - Directory containing videos
- `FRAME_RATE` - Frames per second for video analysis
- `LOCAL_MODEL_ID` - Local LLM model to use
- `OPENAI_MODEL` - OpenAI model to use
- `ANTHROPIC_MODEL` - Anthropic model to use

## Workflow

1. Run Stage 1 to extract descriptions from all videos
2. Review descriptions in `video_descriptions/` if needed
3. Run Stage 2 to generate PDDL files
4. Review generated PDDL files in `problems/`

## Advantages Over Single-Stage Approach

- **Better accuracy**: Text LLMs excel at structured output
- **Easier debugging**: Can fix descriptions before PDDL generation
- **Reusability**: Can regenerate PDDL with different prompts/models without re-analyzing videos
- **Cost efficiency**: Only need to analyze videos once (Stage 1), can try multiple LLM prompts (Stage 2)

