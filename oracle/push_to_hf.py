"""Accumulate oracle checkpoints/results and push them to HuggingFace.

Walks an entry-point directory (default: oracle/results — i.e. EVERY
timestamped runner invocation under it), collects model checkpoints (*.pth)
plus the small JSON files needed to interpret them (*.result.json,
summary.json), and uploads everything to ONE HuggingFace model repo,
mirroring the on-disk relative paths, e.g.:

    run_20260612_093011/exp01/checkpoints/n2_s0/ep030000.pth
    run_20260612_093011/exp01/n2_s0.result.json
    run_20260612_093011/exp06/summary.json

(`--root <one run dir>` pushes a single run; its run_<ts>/ prefix is kept so
repo paths never collide across runs. Symlinks like results/latest are
skipped, so nothing uploads twice.)

The repo's own file listing is the "already pushed" manifest — anything whose
path already exists in the repo is skipped (no local state to lose; --force
re-uploads). So the loop is: train → `python -m modular_addition.oracle.push_to_hf`
→ only new checkpoints move. Run it after every sweep; dozens of runs
accumulate side by side.

Requires a HuggingFace login (`hf auth login`, or HF_TOKEN in the env).

Usage:
  python -m modular_addition.oracle.push_to_hf --dry-run
  python -m modular_addition.oracle.push_to_hf
  python -m modular_addition.oracle.push_to_hf --repo me/my-checkpoints --root path/
"""
import argparse
import os
import sys
from pathlib import Path

try:
    _root = str(Path(__file__).resolve().parents[2])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

# Default = the results BASE (all timestamped runs accumulate), not the
# `latest` run: pushing is an archival operation. ORACLE_RESULTS_DIR (e.g.
# scratch disk) takes precedence. Torch-free on purpose.
DEFAULT_ROOT = Path(os.environ.get("ORACLE_RESULTS_DIR")
                    or Path(__file__).resolve().parent / "results")
DEFAULT_REPO_NAME = "oracle-encodings-checkpoints"
INCLUDE_SUFFIXES = (".pth",)
INCLUDE_NAMES = ("summary.json",)
INCLUDE_DOUBLE_SUFFIXES = (".result.json",)
EXCLUDE_DIRS = {"figures", "wandb", "__pycache__"}
BATCH = 64   # files per HF commit

README = """---
tags: [grokking, mechanistic-interpretability, oracle-encodings]
---

# oracle-encodings checkpoints

Model checkpoints from the oracle-encodings modular-addition experiments
(frozen "oracle" Fourier features injected into the residual stream of a
1-layer transformer grokking (i + j) mod p).

Layout: `run_<timestamp>/<experiment>/checkpoints/<label>/ep<NNNNNN>.pth` —
one `run_<ts>/` tree per runner invocation; the label encodes the sweep axes
(e.g. `n2_s0` = 2 injected frequency pairs, seed 0; `delay4000_n2_s1`,
`amp2_n3_s0`, `rel0.5_n2_s2`, ...). Each .pth holds
`{model: state_dict, config: Config fields, label, epochs_done}` — the frozen
oracle is NOT in the state_dict; rebuild it from the run spec embedded in the
sibling `<experiment>/<label>.result.json`
(`modular_addition.oracle.sweep.build_oracle`).
"""


def wanted(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    return (path.suffix in INCLUDE_SUFFIXES
            or path.name in INCLUDE_NAMES
            or any(path.name.endswith(s) for s in INCLUDE_DOUBLE_SUFFIXES))


def collect(root: Path):
    """BFS from the entry point; returns [(abs_path, path_in_repo)].

    Symlinks (results/latest) are skipped to avoid duplicate uploads. When the
    root itself is one timestamped run dir, its run_<ts>/ name is kept as the
    repo path prefix so single-run pushes line up with whole-base pushes.
    """
    prefix = root.name if root.name.startswith("run_") else ""
    queue, found = [root], []
    while queue:
        d = queue.pop(0)
        for child in sorted(d.iterdir()):
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name not in EXCLUDE_DIRS:
                    queue.append(child)
            elif wanted(child):
                rel = child.relative_to(root).as_posix()
                found.append((child, f"{prefix}/{rel}" if prefix else rel))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="entry point to walk (default: oracle/results)")
    ap.add_argument("--repo", default=None,
                    help="repo id (default: <username>/oracle-encodings-checkpoints)")
    ap.add_argument("--force", action="store_true",
                    help="upload even if the path already exists in the repo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()

    files = collect(args.root.resolve())
    total_mb = sum(p.stat().st_size for p, _ in files) / 1e6
    print(f"found {len(files)} files ({total_mb:.0f} MB) under {args.root}")
    if not files:
        return

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    repo = args.repo or f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"

    if args.dry_run:
        try:
            existing = set(api.list_repo_files(repo))
        except Exception:
            existing = set()
        new = [rel for _, rel in files if rel not in existing]
        print(f"would push {len(new)} new files to {repo} "
              f"({len(files) - len(new)} already there)")
        for rel in new[:20]:
            print("  +", rel)
        if len(new) > 20:
            print(f"  ... and {len(new) - 20} more")
        return

    api.create_repo(repo_id=repo, repo_type="model", private=args.private,
                    exist_ok=True)
    existing = set(api.list_repo_files(repo))
    todo = [(p, rel) for p, rel in files
            if args.force or rel not in existing]
    print(f"pushing {len(todo)} new files to {repo} "
          f"({len(files) - len(todo)} already pushed)")

    if "README.md" not in existing:
        api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type="model")

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p))
               for p, rel in chunk]
        api.create_commit(repo_id=repo, repo_type="model", operations=ops,
                          commit_message=f"push {len(ops)} files "
                                         f"({i + len(ops)}/{len(todo)})")
        print(f"  committed {i + len(ops)}/{len(todo)}")
    print(f"done → https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
