"""Video2Plan (BLIP-2 + Flan-T5, no HDF5)

Implements the user's logic:
- Use BLIP-2 (VLM) on the first frame to produce PDDL-style init predicates
- Use Flan-T5 (LLM) to convert a final-scene description into a single PDDL goal predicate
- Read videos directly from raw_videos/<episode>/recordings/MP4/*.mp4 (no trajectory.h5)

Outputs PDDL problems compatible with domain.pddl (domain name 'robot-manipulation').

Usage:
  python video2plan_blip2_flan_no_h5.py --raw-videos-dir raw_videos --out-dir generated_pddl_problems --max-videos 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from PIL import Image

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, T5ForConditionalGeneration

try:  # Unified Planning is optional
    from unified_planning.io import PDDLReader  # type: ignore
    UP_AVAILABLE = True
except Exception:  # pragma: no cover
    UP_AVAILABLE = False


# --- Configuration ---
DEFAULT_RAW_VIDEOS_DIR = "raw_videos"
DEFAULT_OUTPUT_DIR = "generated_pddl_problems"
DOMAIN_FILE = "domain.pddl"  # matches repo domain
DOMAIN_NAME = "robot-manipulation"

# Models
VLM_MODEL_ID = "Salesforce/blip2-opt-2.7b"
LLM_MODEL_ID = "google/flan-t5-base"

# Prompts
INIT_STATE_PROMPT = (
    "Question: Describe the state of the blocks on the table using 'on(block, location)' "
    "and 'clear(block)' predicates. Use snake_case for names like 'red_block'. "
    "Only output a comma-separated list of predicates. Answer:"
)

# For the goal, we will first ask BLIP-2 to describe the final frame in simple text,
# then pass that text to Flan-T5 with the below translator prompt.
GOAL_STATE_PROMPT = (
    "Translate this instruction into a single PDDL goal predicate. Use snake_case for names. "
    "Example: 'stack the red block on the blue block' -> '(on red_block blue_block)'. Instruction: "
)


def load_models() -> Dict[str, object]:
    print("Loading models... This may take a moment.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # BLIP-2
    vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    vlm_model = Blip2ForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # Flan-T5
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_ID).to(device)

    print("Models loaded successfully.")
    return {
        "vlm_processor": vlm_processor,
        "vlm_model": vlm_model,
        "llm_tokenizer": llm_tokenizer,
        "llm_model": llm_model,
        "device": device,
    }


def get_first_frame(video_path: Path) -> Image.Image | None:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def get_last_frame(video_path: Path) -> Image.Image | None:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, total - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def blip2_generate(models: Dict[str, object], image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = models["vlm_processor"](
        images=image, text=prompt, return_tensors="pt"
    ).to(models["device"], torch.float16 if models["device"] == "cuda" else torch.float32)
    with torch.no_grad():
        out = models["vlm_model"].generate(**inputs, max_new_tokens=max_new_tokens)
    return models["vlm_processor"].batch_decode(out, skip_special_tokens=True)[0].strip()


def flan_translate_to_goal(models: Dict[str, object], instruction: str, max_new_tokens: int = 24) -> str:
    prompt = f"{GOAL_STATE_PROMPT}'{instruction}'"
    enc = models["llm_tokenizer"](prompt, return_tensors="pt").to(models["device"])
    out = models["llm_model"].generate(**enc, max_new_tokens=max_new_tokens)
    return models["llm_tokenizer"].batch_decode(out, skip_special_tokens=True)[0].strip()


def build_init(init_predicates_csv: str) -> tuple[str, List[str]]:
    preds = [p.strip() for p in init_predicates_csv.split(',') if p.strip()]
    init_str = "\n\t\t(handempty)"
    for p in preds:
        p = p.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
        if not p.startswith("("):
            p = f"({p})"
        init_str += f"\n\t\t{p}"
    return init_str, preds


def build_goal(goal_predicate_text: str) -> tuple[str, List[str]]:
    g = goal_predicate_text.strip()
    g = g.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    if not g.startswith("("):
        g = f"({g})"
    return f"\n\t\t(and {g})", [g]


def parse_object_names(init_preds: List[str], goal_preds: List[str]) -> Dict[str, List[str]]:
    objects = set()
    all_text = " ".join(init_preds + goal_preds)
    inside = re.findall(r"\((.*?)\)", all_text)
    for group in inside:
        parts = group.replace(",", " ").split()
        if len(parts) < 2:
            continue
        for token in parts[1:]:
            token = token.strip()
            if token:
                objects.add(token)
    by_type = {"block": [], "surface": []}
    for obj in sorted(objects):
        if "table" in obj:
            by_type["surface"].append(obj)
        else:
            by_type["block"].append(obj)
    return by_type


def assemble_pddl_problem(problem_name: str, objects_by_type: Dict[str, List[str]], init_state: str, goal_state: str) -> str:
    obj_lines: List[str] = []
    if objects_by_type["block"]:
        obj_lines.append(f"\t\t{' '.join(objects_by_type['block'])} - block")
    if objects_by_type["surface"]:
        obj_lines.append(f"\t\t{' '.join(objects_by_type['surface'])} - surface")
    objects_str = "\n".join(obj_lines)
    return f"""
(define (problem {problem_name})
    (:domain {DOMAIN_NAME})
    (:objects
{objects_str}
    )
    (:init
{init_state}
    )
    (:goal
{goal_state}
    )
)
"""


def find_video_in_episode_dir(episode_dir: Path) -> Path | None:
    mp4_dir = episode_dir / "recordings" / "MP4"
    if not mp4_dir.exists():
        return None
    videos = sorted(mp4_dir.glob("*.mp4"))
    non_stereo = [v for v in videos if "-stereo" not in v.name]
    return non_stereo[0] if non_stereo else (videos[0] if videos else None)


def verify_pddl(problem_file_path: Path) -> None:
    if not UP_ENABLED:
        return
    try:
        reader = PDDLReader()
        _ = reader.parse_problem(DOMAIN_FILE, str(problem_file_path))
        print(f"  [Success] Verified PDDL: {problem_file_path.name}")
    except Exception as e:
        print(f"  [!! FAILED !!] PDDL verification failed for {problem_file_path.name}: {e}")


UP_ENABLED = UP_AVAILABLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Video2Plan with BLIP-2 + Flan-T5 (no HDF5)")
    parser.add_argument("--raw-videos-dir", type=str, default=DEFAULT_RAW_VIDEOS_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    print("Starting Video2Plan (BLIP-2 + Flan-T5, no HDF5)...")
    models = load_models()

    base_dir = Path(args.raw_videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    if args.max_videos:
        episode_dirs = episode_dirs[: args.max_videos]

    generated = 0
    for episode_dir in episode_dirs:
        video_path = find_video_in_episode_dir(episode_dir)
        if not video_path:
            print(f"  ⚠️  No MP4 found in {episode_dir}")
            continue

        print(f"\n--- Episode: {episode_dir.name}")
        print(f"  Video: {video_path.name}")

        first = get_first_frame(video_path)
        last = get_last_frame(video_path)
        if first is None or last is None:
            print("  ⚠️  Could not read frames; skipping")
            continue

        # 1) INIT from BLIP-2 on first frame
        init_csv = blip2_generate(models, first, INIT_STATE_PROMPT, max_new_tokens=80)
        init_state_str, init_preds = build_init(init_csv)
        print(f"  VLM (:init) -> {init_csv}")

        # 2) GOAL: get a concise description of final state via BLIP-2, then translate to PDDL with Flan
        final_caption = blip2_generate(
            models,
            last,
            "Question: In one short sentence, describe the final arrangement of blocks and containers. Answer:",
            max_new_tokens=48,
        )
        goal_text = flan_translate_to_goal(models, final_caption, max_new_tokens=24)
        goal_state_str, goal_preds = build_goal(goal_text)
        print(f"  LLM (:goal) -> {goal_text}")

        # 3) Assemble problem
        objects_by_type = parse_object_names(init_preds, goal_preds)
        problem_name = f"problem_{episode_dir.name}"
        content = assemble_pddl_problem(problem_name, objects_by_type, init_state_str, goal_state_str)
        out_path = out_dir / f"{problem_name}.pddl"
        out_path.write_text(content)
        print(f"  ✅ Wrote {out_path}")

        # 4) Verify (optional)
        if UP_ENABLED:
            try:
                reader = PDDLReader()
                _ = reader.parse_problem(DOMAIN_FILE, str(out_path))
                print("  ✅ Unified Planning verification passed")
            except Exception as e:
                print(f"  ⚠️  Unified Planning verification failed: {e}")

        generated += 1

    print(f"\nAutomation complete. Generated {generated} problem file(s).")


if __name__ == "__main__":
    main()


