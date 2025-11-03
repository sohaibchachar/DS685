"""Video2Plan (no HDF5): Generate PDDL problems from raw videos using BLIP-2.

This script:
- Scans `raw_videos/<episode>/recordings/MP4/*.mp4`
- Takes the first frame as the initial state and the last frame as the final state
- Uses BLIP-2 to extract PDDL-style predicates for init and goal
- Writes problem files compatible with the existing `domain.pddl` (domain name `robot-manipulation`)
- Validates the generated problems with Unified Planning if available

Usage:
  python video2plan_blip2_no_h5.py --raw-videos-dir raw_videos --out-dir generated_pddl_problems --max-videos 5

Dependencies:
  pip install transformers torch pillow opencv-python unified-planning
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from PIL import Image


try:
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing deps. Install: pip install transformers torch pillow"
    ) from e


# Domain alignment with existing repo artifacts
DOMAIN_FILE = "domain.pddl"  # keep in sync with generated domain
DOMAIN_NAME = "robot-manipulation"

# Prompts
INIT_STATE_PROMPT = (
    "Question: Describe the state of the blocks on the table using 'on-table(block)', "
    "'on(block, object)', and 'clear(block)' predicates. "
    "Use snake_case names like 'red_block'. Only output a comma-separated list of predicates. Answer:"
)

GOAL_STATE_PROMPT = (
    "Question: Based on this final scene, output a single PDDL goal predicate that best describes "
    "the achieved arrangement, e.g. '(on red_block blue_block)' or '(in red_block bowl1)' or "
    "'(on-table red_block)'. Use snake_case names. Only output the predicate. Answer:"
)


def load_vlm(vlm_model_id: str) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(vlm_model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        vlm_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def grab_first_frame(video_path: Path) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def grab_last_frame(video_path: Path) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    # Seek to last frame - 1 (robust guard for some codecs)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, total - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def vlm_generate_predicates(models: Dict[str, object], image: Image.Image, prompt: str) -> str:
    processor = models["processor"]
    model = models["model"]
    device = models["device"]

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        device, torch.float16 if device == "cuda" else torch.float32
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=80)
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return text


def build_init_from_text(text: str) -> Tuple[str, List[str]]:
    # Expect comma-separated predicates, e.g.: "on-table(red_block), clear(red_block), on(blue_block table1)"
    parts = [p.strip() for p in text.split(",") if p.strip()]
    init_lines = ["\t\t(handempty)"]
    for p in parts:
        # Normalize spacing and parentheses
        p = p.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
        p = p.replace("  ", " ")
        if not p.startswith("("):
            p = f"({p})"
        init_lines.append(f"\t\t{p}")
    return "\n".join(init_lines), parts


def build_goal_from_text(text: str) -> Tuple[str, List[str]]:
    # Expect a single predicate, ensure it is wrapped in parentheses
    pred = text.strip()
    pred = pred.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    if not pred.startswith("("):
        pred = f"({pred})"
    return f"\n\t\t(and {pred})", [pred]


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

    by_type = {"block": [], "surface": [], "container": []}
    for obj in sorted(objects):
        if "table" in obj:
            by_type["surface"].append(obj)
        elif any(k in obj for k in ["bowl", "cup", "container"]):
            by_type["container"].append(obj)
        else:
            by_type["block"].append(obj)
    return by_type


def assemble_pddl_problem(problem_name: str, objects_by_type: Dict[str, List[str]], init_state: str, goal_state: str) -> str:
    obj_lines: List[str] = []
    if objects_by_type["block"]:
        obj_lines.append(f"\t\t{' '.join(objects_by_type['block'])} - block")
    if objects_by_type["container"]:
        obj_lines.append(f"\t\t{' '.join(objects_by_type['container'])} - container")
    if objects_by_type["surface"]:
        obj_lines.append(f"\t\t{' '.join(objects_by_type['surface'])} - surface")

    objects_str = "\n".join(obj_lines) if obj_lines else "\t\tblock1 block2 - block\n\t\ttable1 - surface"

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


def main():
    parser = argparse.ArgumentParser(description="Generate PDDL problems from raw videos using BLIP-2 (no HDF5)")
    parser.add_argument("--raw-videos-dir", type=str, default="raw_videos")
    parser.add_argument("--out-dir", type=str, default="generated_pddl_problems")
    parser.add_argument("--vlm-model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(args.raw_videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BLIP-2 model...")
    models = load_vlm(args.vlm_model)
    print("Model ready.")

    episode_dirs = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    if args.max_videos:
        episode_dirs = episode_dirs[: args.max_videos]

    generated = 0
    for episode_dir in episode_dirs:
        video_path = find_video_in_episode_dir(episode_dir)
        if not video_path:
            print(f"  ⚠️  No MP4 found in {episode_dir}")
            continue

        print(f"\n🎬 Processing: {episode_dir.name} -> {video_path.name}")

        first = grab_first_frame(video_path)
        last = grab_last_frame(video_path)
        if first is None or last is None:
            print("  ⚠️  Could not decode frames; skipping")
            continue

        try:
            init_text = vlm_generate_predicates(models, first, INIT_STATE_PROMPT)
            goal_text = vlm_generate_predicates(models, last, GOAL_STATE_PROMPT)
        except Exception as e:  # pragma: no cover
            print(f"  ❌ Generation error: {e}")
            continue

        init_state_str, init_preds = build_init_from_text(init_text)
        goal_state_str, goal_preds = build_goal_from_text(goal_text)
        obj_by_type = parse_object_names(init_preds, goal_preds)

        problem_name = f"problem_{episode_dir.name}"
        content = assemble_pddl_problem(problem_name, obj_by_type, init_state_str, goal_state_str)
        out_file = out_dir / f"{problem_name}.pddl"
        out_file.write_text(content)
        print(f"  ✅ Wrote {out_file}")

        # Try to validate if unified-planning is present
        try:
            from unified_planning.io import PDDLReader  # type: ignore

            reader = PDDLReader()
            _ = reader.parse_problem(DOMAIN_FILE, str(out_file))
            print("  ✅ Unified Planning validation passed")
        except Exception as e:
            print(f"  ⚠️  Validation skipped/failed: {e}")

        generated += 1

    print(f"\n✅ Done. Generated {generated} PDDL problems in {out_dir}")


if __name__ == "__main__":
    main()


