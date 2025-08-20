#!/usr/bin/env python3
"""
Push/pull dated snapshots to/from Hugging Face Hub with auto tar chunking — now with
*named* datasets so you can sync multiple folders independently (e.g., data1, data2),
and a progress bar for the packing step.

Repo (fixed): albertkuilin/observation-conditioned-reachability (dataset repo recommended)

Commands:
  push <name> <local_folder>               Package & upload folder into snapshots/<name>/<TAG>/
  pull <name> <dest_folder> [--tag TAG]    Download & extract latest or specific snapshot for <name>
  list [name]                              List available snapshot tags (all names or a single name)

Why tar parts?
  - Hugging Face performs better with fewer, larger files vs thousands of tiny ones.
  - We pack your folder into ~15 GiB tar parts (configurable), plus a manifest.json.
  - On pull, we download the chosen snapshot subtree and auto-extract to your destination.

Auth:
  - Run `huggingface-cli login` once OR set env var HUGGINGFACE_TOKEN.

Examples:
  python sync_data.py push data1 /data/exp_A
  python sync_data.py push data2 ./my_big_folder

  python sync_data.py pull data1 ./results                # latest snapshot for data1
  python sync_data.py pull data2 ./results2 --tag 2025-08-15_14-30

  python sync_data.py list                                # show all names and latest tags
  python sync_data.py list data1                          # show tags for just data1
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import HfApi, snapshot_download

from tqdm import tqdm

# ======================== CONFIG ========================
REPO_ID = "albertkuilin/observation-conditioned-reachability"
REPO_TYPE = "dataset"
SNAPSHOTS_PREFIX = "snapshots"
DATE_FMT = "%Y-%m-%d_%H-%M"   # include seconds with "%Y-%m-%d_%H-%M-%S" if desired

# Packaging
MAX_PART_SIZE_GIB = 15           # target per tar part size (GiB)
COMPRESS = True                  # False => .tar (fast), True => .tar.gz (smaller, more CPU)
FOLLOW_SYMLINKS = True           # dereference symlinks when adding to tar

# Upload ignore filters (when packaging)
IGNORE_PATTERNS = ["**/.DS_Store", "**/__pycache__/**", "**/*.tmp", "**/.git/**", "*.mp4", "*.png"]
# ========================================================

# Match: snapshots/<name>/<TAG>/...
RE_NAME_TAG = re.compile(
    rf"^{re.escape(SNAPSHOTS_PREFIX)}/([^/]+)/"  # <name>
    r"(\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}(?:-\d{2})?))/?"  # <TAG>
)

@dataclass
class Manifest:
    dataset_name: str
    snapshot_tag: str
    compress: bool
    max_part_size_gib: int
    num_parts: int
    root_rel: str
    total_bytes: int
    files: List[str]

# ---------------------- helpers ----------------------
def _parse_tag(tag: str) -> datetime:
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(tag, fmt)
        except ValueError:
            pass
    return datetime.min


def _index_snapshots(paths: Iterable[str]) -> Dict[str, Set[str]]:
    """Return mapping: name -> set(tags) from repo file paths."""
    idx: Dict[str, Set[str]] = {}
    for p in paths:
        m = RE_NAME_TAG.match(p)
        if not m:
            continue
        name, tag = m.group(1), m.group(2)
        idx.setdefault(name, set()).add(tag)
    return idx


def _latest_for_name(tags: Set[str]) -> Optional[str]:
    if not tags:
        return None
    return max(tags, key=_parse_tag)


def _open_tar(path: Path, compress: bool) -> tarfile.TarFile:
    mode = "w:gz" if compress else "w"
    # honor FOLLOW_SYMLINKS setting
    return tarfile.open(path, mode=mode, dereference=FOLLOW_SYMLINKS)


def _list_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if any(Path(rel).match(pat) for pat in IGNORE_PATTERNS):
                continue
            files.append(p)
    return files


def _pack_into_parts(name: str, src_root: Path, workdir: Path, tag: str) -> Manifest:
    files = _list_files(src_root)
    if not files:
        raise SystemExit(f"[!] No files to package under {src_root}")

    # Precompute total size for progress bar
    sizes = [f.stat().st_size for f in files]
    total_bytes = sum(sizes)

    parts_dir = workdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(MAX_PART_SIZE_GIB * (1024**3))
    part_idx, current_bytes = 0, 0
    tar_path = parts_dir / (f"part-{part_idx:03d}.tar.gz" if COMPRESS else f"part-{part_idx:03d}.tar")
    tarf = _open_tar(tar_path, COMPRESS)

    included_relpaths: List[str] = []

    # Progress bar over total bytes (updates after each file is added)
    bar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"Packing {name}")

    def finalize():
        nonlocal tarf
        if tarf:
            tarf.close()

    try:
        for f, fsize in zip(files, sizes):
            rel = f.relative_to(src_root).as_posix()

            if current_bytes > 0 and (current_bytes + fsize > max_bytes):
                finalize()
                part_idx += 1
                current_bytes = 0
                tar_path = parts_dir / (f"part-{part_idx:03d}.tar.gz" if COMPRESS else f"part-{part_idx:03d}.tar")
                tarf = _open_tar(tar_path, COMPRESS)
            tarf.add(f, arcname=rel, recursive=False)
            included_relpaths.append(rel)
            current_bytes += fsize

            bar.update(fsize)

        finalize()
    except Exception:
        finalize()
        raise
    finally:
        bar.close()

    manifest = Manifest(
        dataset_name=name,
        snapshot_tag=tag,
        compress=COMPRESS,
        max_part_size_gib=MAX_PART_SIZE_GIB,
        num_parts=part_idx + 1,
        root_rel=".",
        total_bytes=total_bytes,
        files=included_relpaths,
    )
    (workdir / "manifest.json").write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")
    return manifest

# ---- local metadata + empty-destination guard ----
def _require_empty_dir(path: Path):
    """Refuse to pull into non-empty directory to avoid overwriting local data."""
    if path.exists():
        if any(path.iterdir()):
            raise SystemExit(f"[!] Destination '{path}' must be EMPTY. Choose a new folder or clear it first.")
    else:
        path.mkdir(parents=True, exist_ok=True)

def _write_local_meta(dest: Path, *, name: str, tag: str, prefix: str, manifest: dict):
    """Record provenance for what was pulled where."""
    meta = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "dataset_name": name,
        "snapshot_tag": tag,
        "source_prefix": prefix,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": {
            "compress": manifest.get("compress"),
            "max_part_size_gib": manifest.get("max_part_size_gib"),
            "num_parts": manifest.get("num_parts"),
            "total_bytes": manifest.get("total_bytes"),
        },
    }
    (dest / ".hf_snapshot_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# ---------------------- commands ----------------------
def cmd_push(name: str, local_folder: Path):
    if not local_folder.is_dir():
        raise SystemExit(f"[!] Not a directory: {local_folder}")
    tag = datetime.now().astimezone().strftime(DATE_FMT)
    path_in_repo = f"{SNAPSHOTS_PREFIX}/{name}/{tag}/"

    print(f"→ Packaging '{local_folder}' into tar parts (≤{MAX_PART_SIZE_GIB} GiB each)")
    with tempfile.TemporaryDirectory(prefix="hfpack_") as td:
        workdir = Path(td)
        manifest = _pack_into_parts(name, local_folder, workdir, tag)

        api = HfApi()
        print(f"→ Uploading to {REPO_ID}:{path_in_repo}")
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(workdir),
            path_in_repo=path_in_repo,
            commit_message=f"{name} sync {tag}",
            commit_description=(
                f"{name}: packed from {local_folder}  total={manifest.total_bytes/1e9:.2f} GB  parts={manifest.num_parts}"
            ),
        )
    print("✓ Upload complete.")


def _extract_all(parts_dir: Path, dest_dir: Path):
    tars = sorted(list(parts_dir.glob("*.tar*")))
    if not tars:
        raise SystemExit(f"[!] No tar parts found in {parts_dir}")
    for t in tars:
        print(f"→ Extracting {t.name}")
        mode = "r:gz" if t.suffixes and t.suffixes[-1] == ".gz" else "r"
        with tarfile.open(t, mode) as tf:
            tf.extractall(path=dest_dir)


def cmd_pull(name: str, dest_folder: Path, tag: Optional[str]):
    _require_empty_dir(dest_folder)
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    idx = _index_snapshots(files)

    tags = idx.get(name, set())
    if not tags:
        raise SystemExit(f"[!] No snapshots found for name '{name}'.")

    chosen = tag or _latest_for_name(tags)
    if chosen not in tags:
        raise SystemExit(f"[!] Tag '{chosen}' not found for name '{name}'.")

    prefix = f"{SNAPSHOTS_PREFIX}/{name}/{chosen}/"

    with tempfile.TemporaryDirectory(prefix="hfdl_") as td:
        tmp = Path(td)
        print(f"→ Downloading snapshot: {prefix}")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=[f"{prefix}**"],
            local_dir=str(tmp),  # Current hub writes real files in this folder
        )

        snap_root = tmp / prefix
        manifest_path = snap_root / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit("[!] manifest.json missing in snapshot")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parts_dir = snap_root / "parts"
        print(f"→ Extracting {manifest['num_parts']} part(s) to {dest_folder}")
        _extract_all(parts_dir, dest_folder)

        # Write local provenance metadata
        _write_local_meta(dest_folder, name=name, tag=chosen, prefix=prefix, manifest=manifest)

    print("✓ Download + extract complete.")
    print(f"✓ Wrote provenance metadata: {dest_folder / '.hf_snapshot_meta.json'}")


def cmd_list(name: Optional[str]):
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    idx = _index_snapshots(files)

    if not idx:
        print("(no snapshots found)")
        return

    if name:
        tags = idx.get(name, set())
        if not tags:
            print(f"(no snapshots for '{name}')")
            return
        for t in sorted(tags, key=_parse_tag, reverse=True):
            print(t)
        return

    # Show all names with their latest tag
    rows: List[Tuple[str, Optional[str], int]] = []
    for n, ts in idx.items():
        rows.append((n, _latest_for_name(ts), len(ts)))
    rows.sort(key=lambda x: x[0])

    col1, col2, col3 = "name", "latest_tag", "count"
    print(f"{col1:20}  {col2:20}  {col3:>5}")
    print("-" * 50)
    for n, lt, cnt in rows:
        print(f"{n:20}  {str(lt) if lt else '-':20}  {cnt:5d}")


# ---------------------- CLI ----------------------
def build_parser() -> argparse.ArgumentParser:
    epilog = f"""
Examples:
  # Push a folder into a new dated snapshot for a given name
  %(prog)s push data1 /path/to/local_dir

  # Pull the latest snapshot for a given name into an EMPTY ./data folder
  %(prog)s pull data1 ./data

  # Pull a specific snapshot by tag for a given name (destination must be empty)
  %(prog)s pull data2 ./dir --tag 2025-08-15_14-30

  # List all names with their latest tag and counts
  %(prog)s list

  # List all tags for one name
  %(prog)s list data1

Environment:
  - Authenticate with `huggingface-cli login` OR set HUGGINGFACE_TOKEN.
  - Repo is fixed to: {REPO_ID} (type: {REPO_TYPE})

Behavior & Limits:
  - Files are packed into {'tar.gz' if COMPRESS else 'tar'} parts up to ~{MAX_PART_SIZE_GIB} GiB each.
  - Hub hard cap: 50 GB per file. Use fewer/larger files for performance.
  - On pull, files are downloaded directly to a local folder and then extracted.
  - Layout on Hub: {SNAPSHOTS_PREFIX}/<name>/<TAG>/{'{parts/, manifest.json}'}
"""
    parser = argparse.ArgumentParser(
        description="Sync named, dated snapshots to/from Hugging Face Hub with auto tar chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Upload a local folder as a dated snapshot under a name")
    p_push.add_argument("name", type=str, help="Dataset name (e.g., data1, data2)")
    p_push.add_argument("local_folder", type=Path, help="Path to the local folder to package & upload")

    p_pull = sub.add_parser("pull", help="Download & extract latest or specific snapshot for a name (requires EMPTY dest)")
    p_pull.add_argument("name", type=str, help="Dataset name (e.g., data1, data2)")
    p_pull.add_argument("dest_folder", type=Path, help="EMPTY destination directory for extracted files")
    p_pull.add_argument("--tag", type=str, help="Snapshot tag to pull (e.g., YYYY-MM-DD_HH-MM). Defaults to latest.")

    p_list = sub.add_parser("list", help="List available snapshot tags")
    p_list.add_argument("name", nargs="?", help="Optional dataset name to filter by")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "push":
        cmd_push(args.name, args.local_folder.expanduser())
    elif args.cmd == "pull":
        cmd_pull(args.name, args.dest_folder.expanduser(), args.tag)
    elif args.cmd == "list":
        cmd_list(args.name)
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
