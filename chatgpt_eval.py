import os
import io
import json
import time
import base64
from typing import Dict, Tuple, Any, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score


# ============================================================
# CONFIG
# ============================================================

OPENAI_MODEL = "gpt-4.1-mini"

LABEL_CSV = "data/test/test_labels.csv"
RAW_ARROW_FILE = "data/test/test_images.arrow"
SALIENCY_ARROW_FILE = "data/test/saliency_maps.arrow"

OUTPUT_DIR = "openai_eval_outputs"
RAW_PRED_CSV = os.path.join(OUTPUT_DIR, "raw_predictions.csv")
SAL_PRED_CSV = os.path.join(OUTPUT_DIR, "saliency_predictions.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "precision_recall_metrics.csv")

AILMENTS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# For debugging, set e.g. MAX_SAMPLES = 5
MAX_SAMPLES = 5

# Mild throttling / pacing
SLEEP_SECONDS = 0.2

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# IO HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_arrow_table(path: str) -> pa.Table:
    """
    Load Arrow table from IPC file or stream.
    """
    try:
        with pa.memory_map(path, "r") as source:
            return pa_ipc.open_file(source).read_all()
    except Exception:
        with pa.memory_map(path, "r") as source:
            return pa_ipc.open_stream(source).read_all()


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# LABELS
# ============================================================

def normalize_label(v) -> int:
    """
    CheXpert mapping for eval:
      1 -> 1
      everything else (0, -1, NaN, blank) -> 0
    """
    if pd.isna(v):
        return 0
    try:
        fv = float(v)
    except Exception:
        return 0
    return 1 if fv == 1.0 else 0


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in AILMENTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in label CSV: {missing}")

    for c in AILMENTS:
        df[c] = df[c].apply(normalize_label)

    return df.reset_index(drop=True)


# ============================================================
# ARROW IMAGE DECODING
# ============================================================

def _maybe_numpy_from_buffer(value: Any) -> Optional[np.ndarray]:
    """
    Try decoding bytes-like buffers.
    Assumes uint8 image data with common layouts:
      - CHW: 3 x 224 x 224
      - HWC: 224 x 224 x 3
    """
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None

    buf = np.frombuffer(value, dtype=np.uint8)
    if buf.size == 3 * 224 * 224:
        return buf.reshape(3, 224, 224)
    if buf.size == 224 * 224 * 3:
        return buf.reshape(224, 224, 3)

    raise ValueError(f"Byte buffer has {buf.size} uint8 elements; cannot infer image shape.")


def decode_arrow_image_value(value: Any) -> np.ndarray:
    """
    Convert one Arrow cell into a numpy image array.

    Handles:
    - nested Python lists
    - numpy arrays
    - bytes buffers
    - odd scalar wrappers via np.asarray fallback

    Returns:
      np.ndarray shaped either [3,H,W] or [H,W,3]
    """
    arr = _maybe_numpy_from_buffer(value)
    if arr is None:
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
        else:
            arr = np.asarray(value)

    if arr.ndim == 0:
        raise ValueError(
            f"Could not decode Arrow image value. "
            f"Type={type(value)}, repr={repr(value)[:200]}"
        )

    # Remove trivial leading dimension if present, e.g. [1,3,H,W] or [1,H,W,3]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")

    return arr


def to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """
    Accept either:
      - CHW [3,H,W]
      - HWC [H,W,3]
    Convert to PIL RGB.
    """
    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")

    # CHW -> HWC
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        # If floats in [0,1], scale up
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def get_raw_image_array(raw_table: pa.Table, idx: int) -> np.ndarray:
    value = raw_table["image_u8"][idx].as_py()
    return decode_arrow_image_value(value)


def get_saliency_image_array(sal_table: pa.Table, row_idx: int) -> np.ndarray:
    value = sal_table["image_u8"][row_idx].as_py()
    return decode_arrow_image_value(value)


# ============================================================
# SALIENCY INDEX
# ============================================================

def build_saliency_index(sal_table: pa.Table) -> Dict[Tuple[int, str], int]:
    """
    Maps (sample_idx, ailment) -> row index in saliency Arrow.
    """
    sample_idxs = sal_table["sample_idx"].to_pylist()
    ailments = sal_table["ailment"].to_pylist()

    index: Dict[Tuple[int, str], int] = {}
    for i, (sample_idx, ailment) in enumerate(zip(sample_idxs, ailments)):
        key = (int(sample_idx), str(ailment))
        index[key] = i

    return index


# ============================================================
# OPENAI SCHEMA / CALLS
# ============================================================

def build_schema() -> Dict:
    props = {}
    for ailment in AILMENTS:
        props[ailment] = {
            "type": "object",
            "properties": {
                "evidence": {"type": "string"},
                "present": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
            "required": ["evidence", "present", "confidence"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "image_description": {"type": "string"},
            "findings": {
                "type": "object",
                "properties": props,
                "required": AILMENTS,
                "additionalProperties": False,
            }
        },
        "required": ["image_description", "findings"],
        "additionalProperties": False,
    }


def call_openai_on_raw_image(img: Image.Image) -> Dict:
    prompt = (
        "You are an expert thoracic radiologist systematically analyzing a chest X-ray.\n"
        "Examine the image carefully and determine if the following 5 findings are present. "
        "Use these standard radiological visual criteria:\n"
        "- Atelectasis: Look for volume loss, linear/wedge-shaped opacities, or anatomical displacement.\n"
        "- Cardiomegaly: Check if the cardiac silhouette is enlarged (cardiothoracic ratio > 0.5).\n"
        "- Consolidation: Look for dense airspace opacification, air bronchograms, or obscured anatomical margins.\n"
        "- Edema: Check for hazy lung fields, prominent interstitial markings, or Kerley B lines.\n"
        "- Pleural Effusion: Look for fluid meniscus or blunting of the costophrenic angles at the lung bases.\n\n"
        "Return JSON only.\n"
        "First, provide an `image_description` outlining the general visual appearance of the lungs, heart, and pleura.\n"
        "Then, for each finding, provide:\n"
        "- evidence: one concise sentence explaining your reasoning based on your image description.\n"
        "- present: true or false\n"
        "- confidence: your diagnostic confidence (number from 0.0 to 1.0)\n\n"
        "Be conservative. Only report findings that are definitively supported by visual evidence."
    )

    img_b64 = pil_to_base64_png(img)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "cxr_raw_eval",
                "schema": build_schema(),
                "strict": True,
            }
        },
        store=False,
    )

    return json.loads(response.output_text)


def call_openai_on_saliency_image(img: Image.Image, target_ailment: str) -> Dict:
    prompt = (
        "You are analyzing a saliency visualization derived from a chest X-ray model.\n"
        f"The target finding for this image is: {target_ailment}.\n"
        "This is a model-generated highlight image, not a ground-truth annotation.\n\n"
        "Task:\n"
        f"- Decide whether the image supports the finding {target_ailment}.\n"
        "- For the four non-target findings, always return:\n"
        "  - evidence='Not evaluated because this saliency image targets a different finding.'\n"
        "  - present=false\n"
        "  - confidence=0.0\n\n"
        "Return JSON only.\n"
        "First, provide an `image_description` describing what anatomical regions are highlighted in the saliency map.\n"
        "Then, for each finding, provide:\n"
        "- evidence: explain if the highlighted regions match the typical anatomy for the target finding.\n"
        "- present: true or false\n"
        "- confidence: number from 0.0 to 1.0"
    )

    img_b64 = pil_to_base64_png(img)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "cxr_saliency_eval",
                "schema": build_schema(),
                "strict": True,
            }
        },
        store=False,
    )

    return json.loads(response.output_text)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame, system_name: str) -> pd.DataFrame:
    rows = []

    for ailment in AILMENTS:
        y_true = gt_df[ailment].astype(int).values
        y_pred = pred_df[f"pred_{ailment}"].astype(int).values

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        rows.append({
            "system": system_name,
            "label": ailment,
            "precision": precision,
            "recall": recall,
            "support_positive": int(y_true.sum()),
            "predicted_positive": int(y_pred.sum()),
            "n_samples": len(y_true),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUNNERS
# ============================================================

def run_raw_eval(gt_df: pd.DataFrame, raw_table: pa.Table) -> pd.DataFrame:
    print("running raw")
    rows = []
    n = len(gt_df) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(gt_df))

    for idx in tqdm(range(n), total=n):
        try:
            raw_arr = get_raw_image_array(raw_table, idx)
            raw_img = to_pil_rgb(raw_arr)

            result = call_openai_on_raw_image(raw_img)
            findings = result["findings"]

            row = {
                "sample_idx": idx,
                "Path": gt_df.loc[idx, "Path"] if "Path" in gt_df.columns else "",
            }
            for ailment in AILMENTS:
                row[f"pred_{ailment}"] = int(bool(findings[ailment]["present"]))
                row[f"conf_{ailment}"] = float(findings[ailment]["confidence"])
                row[f"evidence_{ailment}"] = findings[ailment]["evidence"]

        except Exception as e:
            print(f"[RAW] Error on sample {idx}: {e}")
            row = {
                "sample_idx": idx,
                "Path": gt_df.loc[idx, "Path"] if "Path" in gt_df.columns else "",
            }
            for ailment in AILMENTS:
                row[f"pred_{ailment}"] = 0
                row[f"conf_{ailment}"] = 0.0
                row[f"evidence_{ailment}"] = f"API error: {e}"

        rows.append(row)

        # if (idx + 1) % 10 == 0 or idx == n - 1:
        #     print(f"[RAW] Completed {idx + 1}/{n}")

        time.sleep(SLEEP_SECONDS)

    return pd.DataFrame(rows)


def run_saliency_eval(gt_df: pd.DataFrame, sal_table: pa.Table, sal_index: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    print("running saliency")
    rows = []
    n = len(gt_df) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(gt_df))

    for idx in tqdm(range(n), total=n):
        row = {
            "sample_idx": idx,
            "Path": gt_df.loc[idx, "Path"] if "Path" in gt_df.columns else "",
        }

        for ailment in AILMENTS:
            try:
                row_idx = sal_index[(idx, ailment)]
                sal_arr = get_saliency_image_array(sal_table, row_idx)
                sal_img = to_pil_rgb(sal_arr)

                result = call_openai_on_saliency_image(sal_img, ailment)
                findings = result["findings"]

                row[f"pred_{ailment}"] = int(bool(findings[ailment]["present"]))
                row[f"conf_{ailment}"] = float(findings[ailment]["confidence"])
                row[f"evidence_{ailment}"] = findings[ailment]["evidence"]

            except Exception as e:
                print(f"[SAL] Error on sample {idx}, ailment {ailment}: {e}")
                row[f"pred_{ailment}"] = 0
                row[f"conf_{ailment}"] = 0.0
                row[f"evidence_{ailment}"] = f"API error: {e}"

            time.sleep(SLEEP_SECONDS)

        rows.append(row)

        # if (idx + 1) % 10 == 0 or idx == n - 1:
        #     print(f"[SAL] Completed {idx + 1}/{n}")

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_DIR)

    gt_df = load_ground_truth(LABEL_CSV)

    raw_table = load_arrow_table(RAW_ARROW_FILE)
    sal_table = load_arrow_table(SALIENCY_ARROW_FILE)

    print(f"[INFO] Loaded raw Arrow: {RAW_ARROW_FILE}")
    print(f"[INFO] Raw columns: {raw_table.column_names}")
    print(f"[INFO] Loaded saliency Arrow: {SALIENCY_ARROW_FILE}")
    print(f"[INFO] Saliency columns: {sal_table.column_names}")

    if "image_u8" not in raw_table.column_names:
        raise ValueError(f"Raw Arrow must contain 'image_u8'. Found: {raw_table.column_names}")

    required_sal_cols = {"sample_idx", "ailment", "image_u8"}
    missing_sal_cols = required_sal_cols - set(sal_table.column_names)
    if missing_sal_cols:
        raise ValueError(f"Saliency Arrow missing required columns: {missing_sal_cols}")

    if len(raw_table) < len(gt_df):
        raise ValueError(f"Raw Arrow has {len(raw_table)} rows but label CSV has {len(gt_df)} rows.")

    # Optional debug inspection for the first row
    try:
        v0 = raw_table["image_u8"][0].as_py()
        arr0 = decode_arrow_image_value(v0)
        print(f"[DEBUG] Raw sample 0 decoded shape: {arr0.shape}, dtype: {arr0.dtype}")
    except Exception as e:
        print(f"[DEBUG] Could not decode raw sample 0 during startup: {e}")

    sal_index = build_saliency_index(sal_table)

    raw_pred_df = run_raw_eval(gt_df, raw_table)
    sal_pred_df = run_saliency_eval(gt_df, sal_table, sal_index)

    raw_pred_df.to_csv(RAW_PRED_CSV, index=False)
    sal_pred_df.to_csv(SAL_PRED_CSV, index=False)

    n_eval = min(len(gt_df), len(raw_pred_df), len(sal_pred_df))
    gt_eval_df = gt_df.iloc[:n_eval].copy()

    raw_metrics = compute_metrics(gt_eval_df, raw_pred_df.iloc[:n_eval], "raw_image")
    sal_metrics = compute_metrics(gt_eval_df, sal_pred_df.iloc[:n_eval], "saliency_map")

    metrics_df = pd.concat([raw_metrics, sal_metrics], ignore_index=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n=== Precision / Recall by label ===")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {RAW_PRED_CSV}")
    print(f"- {SAL_PRED_CSV}")
    print(f"- {METRICS_CSV}")


if __name__ == "__main__":
    main()