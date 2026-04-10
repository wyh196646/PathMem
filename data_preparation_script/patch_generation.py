import os
import h5py
import argparse
import openslide
import multiprocessing as mp

from pathlib import Path
from PIL import Image


def process_coord_chunk(args):
    """
    Worker process:
    Each process opens the slide independently and processes a chunk of coordinates.
    """
    slide_path, coord_chunk, save_dir, patch_size, level = args

    success_count = 0
    fail_count = 0

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"[Worker Error] Cannot open slide {slide_path}: {e}")
        return 0, len(coord_chunk)

    for (x, y) in coord_chunk:
        x, y = int(x), int(y)
        patch_name = f"{x}_{y}.jpg"
        patch_path = os.path.join(save_dir, patch_name)

        # Resume support: skip if already exists
        if os.path.exists(patch_path):
            success_count += 1
            continue

        try:
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch.save(patch_path, "JPEG")
            success_count += 1
        except Exception as e:
            print(f"Error extracting patch at {x},{y}: {e}")
            fail_count += 1

    slide.close()
    return success_count, fail_count


def split_list(data, chunk_size):
    """
    Split a list/array into chunks.
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def extract_patches_from_h5_parallel(
    h5_path,
    slide_path,
    save_dir,
    patch_size=4096,
    level=0,
    num_workers=4,
    chunk_size=64
):
    """
    Extract patches from a single WSI based on coordinates in an h5 file, using multiprocessing.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Read coordinates
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    total_coords = len(coords)
    if total_coords == 0:
        print(f"[Warning] No coords found in {h5_path}")
        return

    # Convert numpy array to plain Python list of tuples for safer multiprocessing serialization
    coords = [(int(x), int(y)) for x, y in coords]

    # Resume support: filter out already existing patches before dispatching
    remaining_coords = []
    for x, y in coords:
        patch_path = os.path.join(save_dir, f"{x}_{y}.jpg")
        if not os.path.exists(patch_path):
            remaining_coords.append((x, y))

    if len(remaining_coords) == 0:
        print(f"[Skip] All patches already extracted: {Path(h5_path).stem}")
        return

    print(f"Total coords: {total_coords}")
    print(f"Remaining coords: {len(remaining_coords)}")
    print(f"Using {num_workers} workers, chunk_size={chunk_size}")

    coord_chunks = split_list(remaining_coords, chunk_size)
    worker_args = [
        (slide_path, chunk, save_dir, patch_size, level)
        for chunk in coord_chunks
    ]

    success_total = 0
    fail_total = 0

    with mp.Pool(processes=num_workers) as pool:
        for success_count, fail_count in pool.imap_unordered(process_coord_chunk, worker_args):
            success_total += success_count
            fail_total += fail_count

    print(
        f"[Done] {Path(h5_path).stem} | "
        f"Success: {success_total}, Failed: {fail_total}, Total Remaining: {len(remaining_coords)}"
    )


def batch_extract(
    h5_dir,
    slide_dir,
    output_root,
    patch_size=4096,
    level=0,
    num_workers=4,
    chunk_size=64
):
    """
    Batch process all h5 files in a directory.
    """
    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    total_files = len(h5_files)

    print(f"Found {total_files} h5 files to process.")

    for idx, h5_file in enumerate(h5_files, start=1):
        case_id = h5_file.stem
        save_dir = Path(output_root) / case_id

        with h5py.File(h5_file, "r") as f:
            coords_count = f["coords"].shape[0]

        # Resume / Skip check
        if save_dir.exists():
            existing_files = list(save_dir.glob("*.jpg"))
            if len(existing_files) == coords_count:
                print(f"[Skipping {idx}/{total_files}] {case_id} already finished ({coords_count} patches)")
                continue

        print(f"\n[{idx}/{total_files}] Processing: {h5_file.name}")

        # Default: .svs
        svs_path = Path(slide_dir) / f"{case_id}.tiff"

        if not svs_path.exists():
            print(f"[Skipping] Slide file not found: {svs_path}")
            continue

        extract_patches_from_h5_parallel(
            h5_path=str(h5_file),
            slide_path=str(svs_path),
            save_dir=str(save_dir),
            patch_size=patch_size,
            level=level,
            num_workers=num_workers,
            chunk_size=chunk_size
        )

    print(f"\n[Done] Processed {total_files} h5 files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI using H5 coordinates with multiprocessing")

    # Path arguments
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing .h5 coordinate files"
    )
    parser.add_argument(
        "--slide_dir",
        type=str,
        required=True,
        help="Directory containing raw WSI slides (.svs)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to save extracted patches"
    )

    # Parameter arguments
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4096,
        help="Size of the patches to extract (default: 4096)"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Mag level to extract from (default: 0)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of worker processes (default: cpu_count() // 2)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Number of coordinates handled by each task chunk (default: 64)"
    )

    args = parser.parse_args()

    print("-" * 40)
    print(f"H5 Dir:       {args.h5_dir}")
    print(f"Slide Dir:    {args.slide_dir}")
    print(f"Output Root:  {args.output_root}")
    print(f"Patch Size:   {args.patch_size}")
    print(f"Level:        {args.level}")
    print(f"Num Workers:  {args.num_workers}")
    print(f"Chunk Size:   {args.chunk_size}")
    print("-" * 40)

    batch_extract(
        h5_dir=args.h5_dir,
        slide_dir=args.slide_dir,
        output_root=args.output_root,
        patch_size=args.patch_size,
        level=args.level,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )
