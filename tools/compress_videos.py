import argparse
import multiprocessing as mp
import os
from functools import partial

import nncore


def _proc(blob, fps, size):
    """
    Process a single video: resize and re-encode with ffmpeg,
    and report detailed errors instead of using assert.
    """
    src_path, tgt_path = blob
    if nncore.is_file(tgt_path):
        return # Skip if target file already exists

    # Ensure target directory exists
    nncore.mkdir(nncore.dir_name(tgt_path))

    # Build ffmpeg command
    cmd = (
        f'ffmpeg -y -i {src_path} '
        f'-filter:v "scale=\'if(gt(a,1),trunc(oh*a/2)*2,{size})\':\'if(gt(a,1),{size},trunc(ow*a/2)*2)\'" '
        f'-map 0:v -r {fps} {tgt_path}'
    )

    try:
        # Execute ffmpeg command
        nncore.exec(cmd)
    except Exception as e:
        # Report execution error with full context
        print(f"Error processing '{src_path}': {e}")
        # raise RuntimeError(f"Error running ffmpeg on '{src_path}': {e}")

    # Verify output file
    if not nncore.is_file(tgt_path):
        print(f"Output file not found: '{tgt_path}'")
        # raise FileNotFoundError(f"Output file not found: '{tgt_path}'")
    # print(f"Successfully processed: {src_path} -> {tgt_path} ({size_bytes} bytes)")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default='videos')
    parser.add_argument('--tgt_dir', default='videos_compressed')
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.src_dir != args.tgt_dir, \
        "Source and target directories must differ."

    blobs = []
    for pattern in ('*.mp4', '*.mkv', '*.webp'):
        for path in nncore.find(args.src_dir, pattern):
            rel = path[len(args.src_dir) + 1:]
            base, _ = nncore.split_ext(rel)
            tgt_path = nncore.join(args.tgt_dir, f'{base}.mp4')
            blobs.append((path, tgt_path))

    if not blobs:
        raise FileNotFoundError(
            f"No videos found in '{args.src_dir}'."
        )

    print(f'Total number of videos: {len(blobs)}')
    for b in blobs[:3]:
        print(b)

    worker_count = os.cpu_count() if args.workers is None else args.workers
    print(f'Starting {worker_count} workers...')
    proc = partial(_proc, fps=args.fps, size=args.size)

    prog_bar = nncore.ProgressBar(num_tasks=len(blobs))
    with mp.Pool(worker_count) as pool:
        for _ in pool.imap_unordered(proc, blobs):
            prog_bar.update()
