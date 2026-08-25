from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue


RESOLUTION_DIRS = {
    "1080p": "1080p",
    "2k": "2k",
    "4k": "4k",
}

CONFIGS = {
    "1080p": "configs/pipeline_1080p.json",
    "2k": "configs/pipeline_2k.json",
    "4k": "configs/pipeline_4k.json",
}

BACKGROUNDS = {
    "1080p": "bg/bg_1080P.png",
    "2k": "bg/bg_2K.png",
    "4k": "bg/bg_4K.png",
}

RESOLUTION_ORDER = {
    "1080p": 0,
    "2k": 1,
    "4k": 2,
}

MANIFEST_FIELDS = [
    "resolution",
    "sample_id",
    "image",
    "output_dir",
    "gpu",
    "status",
    "returncode",
]


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**9, path.name


def manifest_key(row: dict) -> tuple:
    sample_id = str(row["sample_id"])
    try:
        sample_num = int(sample_id)
    except ValueError:
        sample_num = 10**9

    return (
        RESOLUTION_ORDER.get(row["resolution"], 99),
        sample_num,
        sample_id,
    )


def image_files(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ],
        key=numeric_key,
    )


def cleanup_minimal_outputs(out_dir: Path) -> None:
    keep_files = {
        "run_report.json",
        "boxes.json",
        "boxes.png",
        "final_refined_boxes.json",
        "final_refined_boxes.png",
        "wallpaper_refine_report.json",
        "sam_results.json",
        "prompts.json",
        "yellow_candidates.json",
        "fused_candidates.json",
    }

    keep_dirs: set[str] = set()

    for child in out_dir.iterdir():
        if child.is_dir() and child.name not in keep_dirs:
            shutil.rmtree(child)
        elif child.is_file() and child.name not in keep_files:
            child.unlink()

def is_complete_run(out_dir: Path) -> bool:
    required = [
        out_dir / "final_refined_boxes.json",
        out_dir / "final_refined_boxes.png",
        out_dir / "run_report.json",
    ]

    return all(
        path.exists() and path.stat().st_size > 0
        for path in required
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run shape_prior_sam on input/<resolution> datasets."
    )

    p.add_argument("--input-root", default="input")
    p.add_argument("--output-root", default="outputs/dataset_eval")

    p.add_argument(
        "--resolutions",
        nargs="+",
        choices=["1080p", "2k", "4k"],
        default=["1080p", "2k", "4k"],
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based start index after numeric sorting.",
    )

    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--split-file", default=None, help="Optional split JSON containing train/val/test image ids.")
    p.add_argument("--split", default=None, choices=["train", "val", "test"], help="Optional split name used with --split-file.")

    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun samples even when complete outputs exist.",
    )

    p.add_argument("--python", default=sys.executable)

    p.add_argument(
        "--pipeline",
        default="pipelines/shape_prior_sam/shape_prior_sam_test.py",
    )

    p.add_argument("--folder-gt", default="assets/folder_templates")

    p.add_argument("--config-1080p", default=None)
    p.add_argument("--config-2k", default=None)
    p.add_argument("--config-4k", default=None)

    p.add_argument(
        "--edge-close-kernel",
        type=int,
        default=3,
    )

    p.add_argument(
        "--edge-close-iterations",
        type=int,
        default=1,
    )

    p.add_argument(
        "--wallpaper-template-keep-threshold",
        type=float,
        default=None,
    )

    p.add_argument(
        "--max-wallpaper-recovery",
        type=int,
        default=None,
    )

    p.add_argument("--max-prompts", type=int, default=None)

    p.add_argument(
        "--sam-single-mask",
        action="store_true",
    )

    p.add_argument(
        "--minimal-output",
        action="store_true",
        help="Keep only final visualizations and JSON needed for metrics.",
    )

    p.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Optional GPU IDs for parallel execution, e.g. --gpus 0 1. Omit to let PyTorch choose the device.",
    )

    return p.parse_args()


def build_command(
    args: argparse.Namespace,
    resolution: str,
    image_path: Path,
    out_dir: Path,
) -> list[str]:

    cmd = [
        args.python,
        args.pipeline,
        "--image",
        str(image_path),
        "--folder-gt",
        args.folder_gt,
        "--output-dir",
        str(out_dir),
        "--config",
        (
            args.config_1080p if resolution == "1080p" and args.config_1080p
            else args.config_2k if resolution == "2k" and args.config_2k
            else args.config_4k if resolution == "4k" and args.config_4k
            else CONFIGS[resolution]
        ),
        "--resolution",
        resolution,
        "--edge-close-kernel",
        str(args.edge_close_kernel),
        "--edge-close-iterations",
        str(args.edge_close_iterations),
        "--wallpaper-bg",
        BACKGROUNDS[resolution],
    ]

    if args.max_prompts is not None:
        cmd.extend(
            [
                "--max-prompts",
                str(args.max_prompts),
            ]
        )

    if args.sam_single_mask:
        cmd.append("--sam-single-mask")

    if args.wallpaper_template_keep_threshold is not None:
        cmd.extend([
            "--wallpaper-template-keep-threshold",
            str(args.wallpaper_template_keep_threshold),
        ])

    if args.max_wallpaper_recovery is not None:
        cmd.extend([
            "--max-wallpaper-recovery",
            str(args.max_wallpaper_recovery),
        ])

    return cmd


def resolve_gpu(gpu_id: int) -> str:
    """
    gpu_id is interpreted relative to the currently visible GPU list.
    """

    current_visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        "",
    ).strip()

    if not current_visible:
        return str(gpu_id)

    visible = [
        item.strip()
        for item in current_visible.split(",")
        if item.strip()
    ]

    if 0 <= gpu_id < len(visible):
        return visible[gpu_id]

    return str(gpu_id)


def load_split_filter(split_file: str | None, split_name: str | None) -> set[str] | None:
    if not split_file and not split_name:
        return None
    if not split_file or not split_name:
        raise ValueError("--split-file and --split must be provided together.")
    with open(split_file, "r", encoding="utf-8") as f:
        data = __import__("json").load(f)
    if split_name not in data:
        raise KeyError(f"Split not found in {split_file}: {split_name}")
    return {str(value) for value in data[split_name]}


def run_sample(
    args: argparse.Namespace,
    task: dict,
    gpu_id: int | None,
) -> dict:

    resolution = task["resolution"]
    idx = task["idx"]
    total = task["total"]
    image_path = task["image_path"]
    out_dir = task["out_dir"]

    sample_id = image_path.stem

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = build_command(
        args,
        resolution,
        image_path,
        out_dir,
    )

    # ============================================================
    # GPU affinity
    # ============================================================
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    gpu_label = "auto" if gpu_id is None else str(gpu_id)

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = resolve_gpu(gpu_id)

    print(
        f"[GPU {gpu_label}] "
        f"[{resolution} {idx}/{total}] "
        f"run {image_path}",
        flush=True,
    )

    try:
        result = subprocess.run(
            cmd,
            text=True,
            env=env,
        )

        returncode = result.returncode

    except Exception as exc:
        print(
            f"[GPU {gpu_label}] "
            f"[{resolution}] "
            f"ERROR {image_path}: {exc}",
            flush=True,
        )

        returncode = 1

    if returncode == 0 and args.minimal_output:
        try:
            cleanup_minimal_outputs(out_dir)
        except Exception as exc:
            print(
                f"[GPU {gpu_label}] "
                f"cleanup failed for {sample_id}: {exc}",
                flush=True,
            )
            returncode = 1

    status = "ok" if returncode == 0 else "failed"

    print(
        f"[GPU {gpu_label}] "
        f"[{resolution} {idx}/{total}] "
        f"{status.upper()} {sample_id}",
        flush=True,
    )

    return {
        "resolution": resolution,
        "sample_id": sample_id,
        "image": str(image_path),
        "output_dir": str(out_dir),
        "gpu": "" if gpu_id is None else gpu_id,
        "status": status,
        "returncode": returncode,
    }


def write_manifest(
    manifest_path: Path,
    rows: list[dict],
) -> None:

    sorted_rows = sorted(
        rows,
        key=manifest_key,
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()
        writer.writerows(sorted_rows)


def gpu_worker(
    gpu_id: int | None,
    task_queue: Queue,
    args: argparse.Namespace,
    rows: list[dict],
    rows_lock: threading.Lock,
    manifest_path: Path,
    failure_codes: list[int],
) -> None:

    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            return

        try:
            row = run_sample(
                args,
                task,
                gpu_id,
            )

            with rows_lock:
                rows.append(row)

                write_manifest(
                    manifest_path,
                    rows,
                )

                if row["returncode"] != 0:
                    failure_codes.append(
                        row["returncode"]
                    )

        finally:
            task_queue.task_done()


def main() -> int:
    args = parse_args()

    gpu_ids = args.gpus if args.gpus else [None]

    if args.gpus and len(set(args.gpus)) != len(args.gpus):
        print(
            f"Duplicate GPU IDs are not allowed: "
            f"{args.gpus}"
        )
        return 1

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_filter = load_split_filter(
        args.split_file,
        args.split,
    )

    manifest_path = (
        output_root
        / "run_manifest.csv"
    )

    rows: list[dict] = []
    tasks: list[dict] = []

    # ============================================================
    # Build task list
    # ============================================================

    for resolution in args.resolutions:

        data_dir = (
            input_root
            / RESOLUTION_DIRS[resolution]
        )

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {data_dir}"
            )

        files = image_files(data_dir)

        if split_filter is not None:
            files = [
                path
                for path in files
                if path.stem in split_filter
            ]

        selected = files[args.start:]

        if args.max_samples is not None:
            selected = selected[
                :args.max_samples
            ]

        for idx, image_path in enumerate(
            selected,
            1,
        ):

            sample_id = image_path.stem

            out_dir = (
                output_root
                / resolution
                / sample_id
            )

            if is_complete_run(out_dir) and not args.force:
                rows.append(
                    {
                        "resolution": resolution,
                        "sample_id": sample_id,
                        "image": str(image_path),
                        "output_dir": str(out_dir),
                        "gpu": "",
                        "status": "skipped_complete",
                        "returncode": 0,
                    }
                )

                print(
                    f"[{resolution} "
                    f"{idx}/{len(selected)}] "
                    f"skip existing {sample_id}",
                    flush=True,
                )

                continue

            tasks.append(
                {
                    "resolution": resolution,
                    "idx": idx,
                    "total": len(selected),
                    "image_path": image_path,
                    "out_dir": out_dir,
                }
            )

    write_manifest(
        manifest_path,
        rows,
    )

    if not tasks:
        print("No samples need to be processed.")
        return 0

    # ============================================================
    # Multi-GPU queue
    # ============================================================

    task_queue: Queue = Queue()

    for task in tasks:
        task_queue.put(task)

    rows_lock = threading.Lock()

    failure_codes: list[int] = []

    print()
    print("=" * 70)
    print("Multi-GPU Dataset Runner")
    print("=" * 70)
    print(f"GPUs          : {'auto' if args.gpus is None else args.gpus}")
    print(f"GPU workers   : {len(gpu_ids)}")
    print(f"Pending tasks : {len(tasks)}")
    print(f"Output root   : {output_root}")
    print("=" * 70)
    print()

    threads: list[threading.Thread] = []

    # One persistent worker thread per GPU.
    for gpu_id in gpu_ids:

        thread = threading.Thread(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                args,
                rows,
                rows_lock,
                manifest_path,
                failure_codes,
            ),
            name=f"gpu-worker-{'auto' if gpu_id is None else gpu_id}",
        )

        thread.start()
        threads.append(thread)

    # Wait for every task to finish.
    task_queue.join()

    for thread in threads:
        thread.join()

    print()
    print("=" * 70)
    print("Dataset run finished")
    print("=" * 70)

    ok_count = sum(
        row["status"] == "ok"
        for row in rows
    )

    skip_count = sum(
        row["status"] == "skipped_complete"
        for row in rows
    )

    failed_count = sum(
        row["status"] == "failed"
        for row in rows
    )

    print(f"OK      : {ok_count}")
    print(f"Skipped : {skip_count}")
    print(f"Failed  : {failed_count}")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)

    if failure_codes:
        return failure_codes[0]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
