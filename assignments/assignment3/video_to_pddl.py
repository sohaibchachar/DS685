"""Generate PDDL domain and problem from a single video/frames using OpenCLIP.

Usage:
  # Using an existing frames directory (recommended)
  python video_to_pddl.py --frames-dir results/video_episode/frames \
      --out-name screenrecording --prompts-file prompts.txt

  # Or from a video (will extract frames first)
  python video_to_pddl.py --video "Screen Recording 2025-10-29 220851.mp4" \
      --out-name screenrecording --fps 1 --prompts-file prompts.txt

Outputs:
  - domain.pddl (in current dir)
  - problem_<out-name>.pddl
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 1) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
        str(frames_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def load_prompts(prompts_file: Path | None) -> list[str]:
    if prompts_file and prompts_file.exists():
        txt = prompts_file.read_text(encoding="utf-8")
        # One prompt per line; ignore empty lines
        return [line.strip() for line in txt.splitlines() if line.strip()]
    # Fallback minimal list if none provided
    return [
        "a robot picking an object", "placing an object", "stacking objects",
        "opening a container", "closing a container", "moving an object",
        "blocks", "cubes", "tabletop objects",
    ]


def score_frames_with_openclip(frames_dir: Path, prompts: list[str]) -> dict:
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCLIP and its deps are required. Install: pip install open-clip-torch torch pillow"
        ) from e

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    text = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Aggregate top labels across frames
    label_counts: dict[str, int] = {p: 0 for p in prompts}
    top_examples: list[tuple[str, float, str]] = []

    jpgs = sorted(frames_dir.glob("*.jpg"))
    for img_path in jpgs[:50]:  # cap for speed
        image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
        best_idx = int(probs.argmax())
        best_label = prompts[best_idx]
        label_counts[best_label] += 1
        top_examples.append((best_label, float(probs.max()), img_path.name))

    return {
        "label_counts": label_counts,
        "top_examples": sorted(top_examples, key=lambda x: -x[1])[:5],
        "frames": len(jpgs),
        "prompts": prompts,
    }


def write_domain(domain_path: Path) -> None:
    content = """(define (domain robot-manipulation)
    (:requirements :strips :typing)
    (:types
        object
        robot
    )
    (:predicates
        (at ?r - robot ?o - object)
        (clear ?o - object)
        (holding ?r - robot ?o - object)
        (on ?o1 - object ?o2 - object)
        (open ?o - object)
        (closed ?o - object)
    )
    (:action pick
        :parameters (?r - robot ?o - object)
        :precondition (and (clear ?o))
        :effect (and (holding ?r ?o) (not (clear ?o)))
    )
    (:action place
        :parameters (?r - robot ?o - object ?dst - object)
        :precondition (and (holding ?r ?o))
        :effect (and (on ?o ?dst) (not (holding ?r ?o)) (clear ?o))
    )
)
"""
    domain_path.write_text(content)


def write_problem(problem_path: Path, name: str, inferred: dict) -> None:
    # Build problem from inferred prompts without hardcoding specific labels.
    # We keep a generic object set; embed the top labels as a comment for traceability.
    init = ["(clear obj1)", "(clear obj2)"]
    # Choose a neutral goal unless caller provides prompts oriented to a specific relation
    # Here we use a simple default; users can refine by editing the problem.
    goal = "(on obj1 table)"

    top_labels = inferred.get("top_examples", [])
    prompts_used = inferred.get("prompts", [])

    content = f"""(define (problem {name})
    (:domain robot-manipulation)
    (:objects
        robot1 - robot
        obj1 obj2 table - object
    )
    (:init
        {' '.join(init)}
    )
    (:goal
        {goal}
    )
    ; Top labels (label, prob, frame): {top_labels}
    ; Prompts used: {prompts_used}
)
"""
    problem_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Single video/frames to PDDL (OpenCLIP)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", type=str, help="Path to video file")
    grp.add_argument("--frames-dir", type=str, help="Existing frames directory")
    parser.add_argument("--out-name", default="video_problem", type=str, help="Problem name suffix")
    parser.add_argument("--fps", default=1, type=int, help="Frame extraction rate")
    parser.add_argument("--prompts-file", type=str, default=None, help="Path to prompts.txt (one prompt per line)")
    args = parser.parse_args()

    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.exists():
            print(f"Frames dir not found: {frames_dir}")
            sys.exit(1)
        print(f"Using existing frames at: {frames_dir}")
    else:
        video = Path(args.video)
        if not video.exists():
            print(f"Video not found: {video}")
            sys.exit(1)
        frames_dir = Path("results") / (Path(args.out_name).stem + "_frames")
        print(f"Extracting frames to: {frames_dir}")
        extract_frames(video, frames_dir, fps=args.fps)

    prompts = load_prompts(Path(args.prompts_file) if args.prompts_file else None)
    print("Scoring frames with OpenCLIP using provided prompts...")
    inferred = score_frames_with_openclip(frames_dir, prompts)
    print(f"Frames: {inferred['frames']}")
    print(f"Top labels: {inferred['top_examples']}")

    # Write domain (generic)
    write_domain(Path("domain.pddl"))

    # Write problem specific to this video
    problem_file = Path(f"problem_{args.out_name}.pddl")
    write_problem(problem_file, f"problem_{args.out_name}", inferred)
    print(f"✅ Wrote domain.pddl and {problem_file}")


if __name__ == "__main__":
    main()


