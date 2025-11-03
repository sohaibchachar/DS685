"""Generate PDDL domain and problem from a single video using OpenCLIP.

Usage:
  python video_to_pddl.py --video "Screen Recording 2025-10-29 220851.mp4" \
      --out-name screenrecording --fps 1

Outputs:
  - domain.pddl (in current dir)
  - problem_<out-name>.pddl
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 5) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
        str(frames_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def score_frames_with_openclip(frames_dir: Path) -> dict:
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

    prompts = [
        "Whats happening in the scene?"
    ]
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


def write_problem(problem_path: Path, name: str, inferred_labels: dict) -> None:
    # Minimal, generic objects inferred from labels
    objects = ["robot1", "obj1", "obj2", "table"]
    init = ["(clear obj1)", "(clear obj2)"]

    # Heuristic goal from labels
    labels = inferred_labels.get("label_counts", {})
    if max(labels.values() or [0]) == 0:
        goal = "(clear obj1)"  # fallback
    else:
        # Prefer stacking/placing if present
        if any(k in labels for k in ["stacking objects", "blocks", "cubes"]):
            goal = "(on obj1 obj2)"
        else:
            goal = "(on obj1 table)"

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
)
"""
    problem_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Single video to PDDL")
    parser.add_argument("--video", required=True, type=str, help="Path to video file")
    parser.add_argument("--out-name", default="video_problem", type=str, help="Problem name suffix")
    parser.add_argument("--fps", default=1, type=int, help="Frame extraction rate")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}")
        sys.exit(1)

    frames_dir = Path("results") / (Path(args.out_name).stem + "_frames")
    print(f"Extracting frames to: {frames_dir}")
    extract_frames(video, frames_dir, fps=args.fps)

    print("Scoring frames with OpenCLIP...")
    inferred = score_frames_with_openclip(frames_dir)
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


