"""
One-time preprocessing script: filter CMS L1T parquet files to keep only
events that contain at least one AK8 jet, then save to data/filtered/.

This avoids re-reading millions of empty events during every training epoch.

Usage:
    cd Tokenizer/
    python scripts/filter_l1t_parquet.py

Reduction achieved:
    minbias:  105,000,000 → 1,264 events  (99.999% reduction)
    QCD:      101,181,400 → 3,019,612 events (97% reduction)
    ggHbb:     10,493,169 →   686,871 events (93% reduction)
    VBFHbb:    10,391,302 → 1,077,601 events (90% reduction)
"""

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import time

DATA_DIR = Path(__file__).parents[2] / "data"
OUT_DIR = DATA_DIR / "filtered"

FILES = [
    "minbias_kinematics.parquet",
    "QCD_HT50toInf_kinematics.parquet",
    "ggHbb_kinematics.parquet",
    "VBFHbb_kinematics.parquet",
]


def filter_file(src: Path, dst: Path) -> None:
    print(f"\n{src.name}")
    t0 = time.time()

    table = pq.read_table(src)
    n_before = len(table)

    # Keep only events with at least one jet
    lengths = pc.list_value_length(table["L1T_JetPuppiAK8_PT"])
    mask = pc.greater_equal(lengths, 1)
    table = table.filter(mask)
    n_after = len(table)

    pq.write_table(table, dst)
    elapsed = time.time() - t0

    print(f"  {n_before:>12,} -> {n_after:>10,} events  "
          f"({100 * n_after / n_before:.3f}% kept)  {elapsed:.1f}s")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    for fname in FILES:
        src = DATA_DIR / fname
        dst = OUT_DIR / fname
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        if dst.exists():
            print(f"  SKIP (already exists): {dst.name}")
            continue
        filter_file(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
