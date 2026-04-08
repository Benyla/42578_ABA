"""Deploy to Hugging Face Spaces (Docker SDK).

Usage:
    pip install huggingface_hub
    huggingface-cli login          # paste your HF token
    python scripts/deploy_hf.py --username YOUR_HF_USERNAME
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "aba-target-detector"


def _collect_dir(base: Path, prefix: str) -> list[tuple[Path, str]]:
    result = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(base)
        result.append((p, f"{prefix}/{rel.as_posix()}"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to HF Spaces")
    parser.add_argument("--username", required=True, help="HF username or org")
    parser.add_argument("--private", action="store_true", help="Make Space private")
    args = parser.parse_args()

    repo_id = f"{args.username}/{SPACE_NAME}"
    api = HfApi()

    print(f"Creating Space {repo_id} ...")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=args.private,
    )

    files: list[tuple[Path, str]] = [
        (ROOT / "Dockerfile", "Dockerfile"),
        (ROOT / "pyproject.toml", "pyproject.toml"),
        (ROOT / "hf_README.md", "README.md"),
        (ROOT / "models" / "stage1_target.pth", "models/stage1_target.pth"),
        (ROOT / "models" / "stage2_bullet.pth", "models/stage2_bullet.pth"),
    ]
    files.extend(_collect_dir(ROOT / "src", "src"))
    files.extend(_collect_dir(ROOT / "configs", "configs"))

    operations = []
    for local_path, repo_path in files:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {repo_path}  ({size_mb:.1f} MB)")
        operations.append(
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
        )

    print(f"\nPushing {len(operations)} files in single commit ...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy two-stage RF-DETR to HF Spaces",
    )

    print(f"\nDone! Space will build at: https://huggingface.co/spaces/{repo_id}")
    print("First build may take 5-10 minutes.")


if __name__ == "__main__":
    main()
