#!/usr/bin/env python3
"""
Convert saliency-map PNGs into a single Arrow file, ordered to match the
existing sample/ailment traversal used by your evaluation script.

Expected input layout:
    SAL_MAP_ROOT/
      sample_00001/
        Atelectasis.png
        Cardiomegaly.png
        Consolidation.png
        Edema.png
        Pleural Effusion.png
      sample_00002/
        ...

This script writes one Arrow row per (sample, ailment) PNG with columns:
    - sample_idx        : 0-based integer sample index
    - sample_id         : 1-based integer sample id
    - ailment           : string
    - image_u8          : uint8 nested list shaped [H, W, 3]
    - source_path       : original PNG path
    - width             : image width
    - height            : image height

Optional:
    - If a CheXpert label CSV is provided and has a Path column, it also stores:
        - cxr_path

This is a clean way to preserve ordering for later OpenAI evaluation while
keeping the saliency images in Arrow instead of individual PNG files.
"""

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

SAL_MAP_ROOT = "saliency_maps_correct/pretrain_final_no_labels"
LABEL_CSV = "data/test/test_labels.csv"   # optional; set to None to disable
OUTPUT_ARROW = "data/test/saliency_maps.arrow"
LIMIT = 100

AILMENTS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# Set to None to process everything found in LABEL_CSV / SAL_MAP_ROOT
MAX_SAMPLES = None

# Whether to fail if any expected PNG is missing
STRICT_MISSING = False


# ============================================================
# HELPERS
# ============================================================

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_sample_dir(sample_idx_1based: int) -> str:
    return os.path.join(SAL_MAP_ROOT, f"sample_{sample_idx_1based:05d}")


def get_saliency_path(sample_idx_1based: int, ailment: str) -> str:
    return os.path.join(get_sample_dir(sample_idx_1based), f"{ailment}.png")


def load_png_rgb(path: str) -> np.ndarray:
    """
    Load a PNG as RGB uint8 array of shape [H, W, 3].
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape {arr.shape} for {path}")

    return arr


def infer_num_samples_from_dirs(root: str) -> int:
    """
    Counts sample_XXXXX directories.
    """
    count = 0
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and name.startswith("sample_"):
            count += 1
    return count


def get_num_samples(label_csv: Optional[str], root: str) -> int:
    if label_csv is not None and os.path.exists(label_csv):
        df = pd.read_csv(label_csv)
        return len(df)
    return infer_num_samples_from_dirs(root)


# ============================================================
# MAIN CONVERSION
# ============================================================

def build_rows() -> List[Dict]:
    label_df = None
    if LABEL_CSV is not None and os.path.exists(LABEL_CSV):
        label_df = pd.read_csv(LABEL_CSV)

    n_total = get_num_samples(LABEL_CSV, SAL_MAP_ROOT)
    n = n_total if MAX_SAMPLES is None else min(MAX_SAMPLES, n_total)

    rows: List[Dict] = []
    missing_count = 0

    for idx in tqdm(range(n), total=n):
        if idx > LIMIT:
            print("reached limit")
            break
        sample_id = idx + 1

        cxr_path = None
        if label_df is not None and "Path" in label_df.columns:
            cxr_path = label_df.loc[idx, "Path"]

        for ailment in AILMENTS:
            png_path = get_saliency_path(sample_id, ailment)

            if not os.path.exists(png_path):
                msg = f"Missing saliency PNG: {png_path}"
                if STRICT_MISSING:
                    raise FileNotFoundError(msg)
                print(f"[WARN] {msg}")
                missing_count += 1
                continue

            arr = load_png_rgb(png_path)  # [H, W, 3], uint8
            h, w, _ = arr.shape

            row = {
                "sample_idx": idx,           # 0-based
                "sample_id": sample_id,      # 1-based
                "ailment": ailment,
                "image_u8": arr.tolist(),    # Arrow-friendly nested list
                "source_path": png_path,
                "width": w,
                "height": h,
            }

            if cxr_path is not None:
                row["cxr_path"] = cxr_path

            rows.append(row)

        # if (idx + 1) % 50 == 0 or idx == n - 1:
        #     print(f"[INFO] Processed {idx + 1}/{n} samples")

    print(f"[INFO] Finished row collection: {len(rows)} rows")
    if missing_count > 0:
        print(f"[INFO] Missing PNG count: {missing_count}")

    return rows


def rows_to_arrow_table(rows: List[Dict]) -> pa.Table:
    if not rows:
        raise ValueError("No rows collected. Nothing to write.")

    columns = {
        key: [row.get(key) for row in rows]
        for key in rows[0].keys()
    }
    return pa.table(columns)


def write_arrow_file(table: pa.Table, out_path: str) -> None:
    ensure_parent_dir(out_path)
    with pa.OSFile(out_path, "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def main():
    if not os.path.isdir(SAL_MAP_ROOT):
        raise FileNotFoundError(f"SAL_MAP_ROOT does not exist: {SAL_MAP_ROOT}")

    rows = build_rows()
    print("tabling...")
    table = rows_to_arrow_table(rows)
    print("writing...")
    write_arrow_file(table, OUTPUT_ARROW)

    print("\n[INFO] Wrote Arrow file successfully")
    print(f"[INFO] Output: {OUTPUT_ARROW}")
    print(f"[INFO] Rows:   {table.num_rows}")
    print(f"[INFO] Cols:   {table.column_names}")
    print(f"[INFO] Schema:\n{table.schema}")


if __name__ == "__main__":
    main()