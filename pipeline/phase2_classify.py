#!/usr/bin/env python3
"""
phase2_classify.py — Perch Hoplite Phase 2: Search, Label, Classify & Inference
=================================================================================
Loads a Hoplite vector database produced by phase1_embed.py, then provides a
suite of sub-commands for the complete agile modeling workflow:

  search      Embed a query audio clip and find nearest neighbors.
  label       Import a CSV of labels (recording_id, offset_s, label, type) into the DB.
  train       Train a linear classifier on the current DB labels.
  review      Run the trained classifier over the DB and write top-scoring results.
  infer       Run full inference over all embeddings and write a results CSV.
  stats       Print DB statistics (label counts, embedding counts).

The GUI for interactive labeling (audio playback + click-to-label) is served as
a Gradio web app via --serve, accessible from any browser on the MBARI network.

Environment
-----------
MBARI NVIDIA DGX SPARC nodes: spark-ae0e (<gpu-host>)
                               spark-0626 (<gpu-host-2>)
NFS base : /mnt/PAM_Analysis/perch-hoplite/
Audio    : /mnt/PAM_Archive/<year>/<deployment>/

Path constants are in:
  /mnt/PAM_Analysis/perch-hoplite/.perch_env
  source that file to avoid typing long paths.

Usage examples
--------------
# 0. Source path constants (optional but convenient):
source /mnt/PAM_Analysis/perch-hoplite/.perch_env

# 1. Search and launch Gradio labeling GUI:
#    Browser: http://<gpu-host>:7860  (spark-ae0e)
#          or http://<gpu-host-2>:7860  (spark-0626)
python3 phase2_classify.py search \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018 \
    --query-audio /mnt/PAM_Analysis/perch-hoplite/queries/cetaceans/orca_call.wav \
    --query-label orca_call \
    --num-results 200 \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_2018_orca_search.csv \
    --serve --port 7860

# 2. Import labels from Raven Pro / PAMGuard CSV:
#    CSV columns: recording_id, offset_s, end_offset_s, label, label_type
#    label_type values: positive | negative | weak_negative
python3 phase2_classify.py label \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018 \
    --labels-csv /mnt/PAM_Analysis/perch-hoplite/labels/orca_raven.csv \
    --annotator-id duane

# 3. Train a linear classifier:
python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018 \
    --classifier-out /mnt/PAM_Analysis/perch-hoplite/models/orca_v1.pt \
    --num-steps 256 --learning-rate 0.001

# 4. Active learning — review classifier results and add more labels:
python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v1.pt \
    --target-label orca_call \
    --num-results 100 \
    --serve --port 7860

# 5. Full inference — write detections CSV:
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v1.pt \
    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_2018_orca_detections.csv \
    --logit-threshold 0.0

# 6. Check DB statistics:
python3 phase2_classify.py stats \
    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_2018

GUI
---
Gradio is already installed (v6.15.1). Add --serve --port 7860 to any
search or review command. The terminal will print the URL; open it in
any browser on the MBARI network. Press Ctrl+C to stop the server.

For multi-analyst annotation campaigns, use Label Studio (Docker):
  docker run -d -p 8080:8080 \
      -v /mnt/PAM_Analysis/perch-hoplite/labelstudio:/label-studio/data \
      heartexlabs/label-studio:latest
  Access at http://<gpu-host>:8080

Known harmless warnings
-----------------------
  "Unable to register cuFFT/cuDNN/cuBLAS factory" — two TF builds
  registering CUDA plugins; GPU works correctly regardless.
  "MessageFactory has no attribute GetPrototype" — protobuf mismatch,
  cosmetic only.
  "NUMA node read from SysFS had negative value" — BIOS limitation,
  TF defaults to node zero correctly.
"""

import argparse
import csv
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Local src modules — extracted for maintainability
try:
    from src.spectrogram import make_spectrogram_image as _src_make_spectrogram
    from src.audio import make_audio_b64 as _src_make_audio_b64
    from src.audio import load_30s_context as _src_load_30s_context
    from src.torch_model import inject_tf_mock as _src_inject_tf_mock
    from src.torch_model import load_model_from_db as _src_load_model_from_db
    from src.train import torch_train_linear_classifier as _src_torch_train
    from src.infer import run_inference as _src_run_inference
    from src.review import launch_labeling_gui as _src_launch_labeling_gui
    _SRC_AVAILABLE = True
except ImportError:
    _SRC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def _setup_logging(log_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(ch)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase2_classify.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(fh)
    logging.info("Logging to %s", log_file)

    # Suppress absl per-SQL-statement INFO spam — floods terminal and
    # generates huge NFS writes during training. Still captures WARNING+.
    for _noisy in ("absl", "absl-py"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Provenance / audit trail ──────────────────────────────────────────────
# Every labeling session and training run writes a JSON record to:
#   <nfs_base>/provenance/labels/labels_<timestamp>_<annotator>.json
#   <nfs_base>/provenance/training/train_<timestamp>_<model>.json
# These records — combined with the audio files — allow full reproduction
# of any classifier from scratch.

PROVENANCE_BASE = "/mnt/PAM_Analysis/perch-hoplite/provenance"


def _provenance_path(subdir: str, stem: str) -> Path:
    p = Path(PROVENANCE_BASE) / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}.json"


def _save_label_provenance(
    session_id: str,
    db_dir: str,
    classifier_path: str | None,
    annotator_id: str,
    query_label: str,
    annotations: list[dict],   # list of {window_id, filename, offset_s, end_s, label, score}
) -> Path | None:
    """Write a labeling session provenance record."""
    import datetime as _dt
    record = {
        "session_id":       session_id,
        "timestamp":        _dt.datetime.now().isoformat(),
        "db_dir":           str(db_dir),
        "classifier":       str(classifier_path) if classifier_path else None,
        "annotator_id":     annotator_id,
        "query_label":      query_label,
        "annotation_count": len(annotations),
        "positive_count":   sum(1 for a in annotations if a["label"] == "positive"),
        "negative_count":   sum(1 for a in annotations if a["label"] == "negative"),
        "annotations":      annotations,
    }
    stem = f"labels_{session_id}"
    out = _provenance_path("labels", stem)
    try:
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        log.info("Label provenance saved to %s", out)
        return out
    except Exception as exc:
        log.warning("Could not save label provenance: %s", exc)
        return None


def _save_training_provenance(
    db_dir: str,
    classifier_out: str,
    target_labels: list[str],
    train_args: dict,
    eval_scores: dict,
    annotation_counts: dict,
    elapsed_s: float,
    full_annotations: list[dict] | None = None,
    audio_sources: list[dict] | None = None,
) -> Path | None:
    """Write a training run provenance record.

    full_annotations — complete list of every annotation used for training,
    with window_id, filename, offset_s, end_s, label, label_type.
    Combined with the audio files and embedding parameters, this allows
    full reproduction of the classifier from scratch.

    audio_sources — list of audio source configs from the DB metadata,
    recording exactly which audio directories and file globs were used.
    """
    import datetime as _dt
    model_name = Path(classifier_out).stem
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "session_id":        f"{ts}_{model_name}",
        "timestamp":         _dt.datetime.now().isoformat(),
        "db_dir":            str(db_dir),
        "classifier_out":    str(classifier_out),
        "target_labels":     target_labels,
        "elapsed_s":         round(elapsed_s, 1),
        "train_args":        train_args,
        "eval_scores":       eval_scores,
        "annotation_counts": annotation_counts,
        "audio_sources":     audio_sources or [],
        "annotations":       full_annotations or [],
    }
    stem = f"train_{ts}_{model_name}"
    out = _provenance_path("training", stem)
    try:
        with open(out, "w") as f:
            json.dump(record, f, indent=2)
        log.info("Training provenance saved to %s", out)
        return out
    except Exception as exc:
        log.warning("Could not save training provenance: %s", exc)
        return None

class _LT:
    """LabelType constants — works across perch-hoplite versions."""
    try:
        from perch_hoplite.db import annotations as _a
        POSITIVE      = _a.LabelType.POSITIVE
        NEGATIVE      = _a.LabelType.NEGATIVE
        WEAK_NEGATIVE = _a.LabelType.WEAK_NEGATIVE
    except Exception:
        try:
            from perch_hoplite.db import interface as _b
            POSITIVE      = _b.LabelType.POSITIVE
            NEGATIVE      = _b.LabelType.NEGATIVE
            WEAK_NEGATIVE = _b.LabelType.NEGATIVE
        except Exception:
            POSITIVE      = 1
            NEGATIVE      = 2
            WEAK_NEGATIVE = 3


def _get_label_type_enum():
    """Return the LabelType enum from wherever perch-hoplite 1.0.1 exposes it.

    The location changed across versions:
      - perch_hoplite.db.interface          (older builds)
      - perch_hoplite.db.annotations        (1.0.x)
      - perch_hoplite.agile.source_info     (some builds)
    Falls back to a simple namespace object so labeling still works.
    """
    for mod_path, attr in (
        ("perch_hoplite.db.annotations",    "LabelType"),
        ("perch_hoplite.db.interface",       "LabelType"),
        ("perch_hoplite.agile.source_info",  "LabelType"),
        ("perch_hoplite.db.sqlite_usearch_impl", "LabelType"),
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            lt = getattr(mod, attr, None)
            if lt is not None:
                return lt
        except ImportError:
            continue

    # Ultimate fallback: plain namespace with integer values
    # (perch-hoplite uses these ints internally in SQLite)
    log.warning(
        "Could not import LabelType from perch_hoplite — "
        "using integer fallback (0=positive, 1=negative, 2=weak_negative)."
    )
    class _LabelType:
        POSITIVE      = 1
        NEGATIVE      = 2
        WEAK_NEGATIVE = 3
        UNCERTAIN     = 3
    return _LabelType

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pure PyTorch/numpy linear classifier — replaces perch_hoplite's TF version
# ---------------------------------------------------------------------------

def _torch_train_linear_classifier(
    data_manager,
    learning_rate: float,
    weak_neg_weight: float,
    num_train_steps: int,
    loss: str = "bce",
):
    """Train a linear classifier using PyTorch — no TensorFlow required.

    Drop-in replacement for classifier_mod.train_linear_classifier().
    Uses the same DataManager interface so all DB/label logic is unchanged.
    Returns (LinearClassifier, eval_scores) matching the original API.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from tqdm import tqdm

    embedding_dim = data_manager.db.get_embedding_dim()
    target_labels = data_manager.get_target_labels()
    num_classes = len(target_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Linear model: embedding → logits (no bias in perch convention)
    linear = nn.Linear(embedding_dim, num_classes, bias=True).to(device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)

    def bce_loss_fn(logits, y_true, is_labeled):
        y = y_true if isinstance(y_true, torch.Tensor) else torch.tensor(y_true, dtype=torch.float32, device=device)
        m = is_labeled if isinstance(is_labeled, torch.Tensor) else torch.tensor(is_labeled, dtype=torch.float32, device=device)
        log_p     = torch.nn.functional.logsigmoid(logits)
        log_not_p = torch.nn.functional.logsigmoid(-logits)
        raw_bce   = -y * log_p - (1.0 - y) * log_not_p
        weights   = (1.0 - m) * weak_neg_weight + m
        return (raw_bce * weights).mean()

    def hinge_loss_fn(logits, y_true, is_labeled):
        y_t = y_true if isinstance(y_true, torch.Tensor) else torch.tensor(y_true, dtype=torch.float32, device=device)
        y = 2 * y_t - 1
        m = is_labeled if isinstance(is_labeled, torch.Tensor) else torch.tensor(is_labeled, dtype=torch.float32, device=device)
        weights = (1.0 - m) * weak_neg_weight + m
        raw = torch.clamp(1.0 - y * logits, min=0.0)
        return (raw * weights).mean()

    loss_fn = hinge_loss_fn if loss == "hinge" else bce_loss_fn

    # Get train/eval split
    train_ids, eval_ids = data_manager.get_train_test_split()

    # ── Pre-load ALL embeddings + labels into GPU memory ─────────────────
    # Much faster than per-batch DB reads (17280 × 1536 float32 ≈ 100 MB).
    # Memory scales with LABEL COUNT only, not DB size — so this is safe
    # even for multi-month DBs with millions of embeddings.
    # Safeguard: warn if label count would require >4 GB GPU memory.
    _LABEL_WARN_THRESHOLD = 50_000
    _n_labels = len(train_ids) + len(eval_ids)
    _est_gb = _n_labels * embedding_dim * 4 / 1e9
    if _n_labels > _LABEL_WARN_THRESHOLD:
        log.warning(
            "Large label set: %d examples × %d dims = %.1f GB GPU memory. "
            "Consider reducing --train-ratio or using --batch-size to limit memory.",
            _n_labels, embedding_dim, _est_gb
        )
    else:
        log.info("Pre-loading %d labeled examples (%.1f MB) into %s...",
                 _n_labels, _est_gb * 1000, device)

    def _load_ids_to_tensors(ids, add_weak_negatives):
        """Load a set of window IDs into (embeddings, multihot, is_labeled) tensors."""
        batches = list(data_manager.batched_example_iterator(
            ids, add_weak_negatives=add_weak_negatives, repeat=False))
        if not batches:
            return None, None, None
        emb  = np.concatenate([b.embedding     for b in batches], axis=0)
        mh   = np.concatenate([b.multihot       for b in batches], axis=0)
        ilm  = np.concatenate([b.is_labeled_mask for b in batches], axis=0)
        idxs = np.concatenate([b.idx            for b in batches], axis=0)
        return (torch.tensor(emb,  dtype=torch.float32, device=device),
                torch.tensor(mh,   dtype=torch.float32, device=device),
                torch.tensor(ilm,  dtype=torch.float32, device=device),
                idxs)

    train_emb, train_mh, train_ilm, _     = _load_ids_to_tensors(train_ids, add_weak_negatives=True)
    eval_emb,  eval_mh,  eval_ilm,  eval_idxs = _load_ids_to_tensors(eval_ids,  add_weak_negatives=False)
    n_train = train_emb.shape[0] if train_emb is not None else 0
    log.info("Loaded %d train + %d eval examples onto %s", n_train,
             eval_emb.shape[0] if eval_emb is not None else 0, device)

    # ── Training loop — pure in-memory mini-batches ───────────────────────
    linear.train()
    rng = np.random.default_rng(seed=42)
    batch_size = min(512, n_train)

    with tqdm(total=num_train_steps, desc="Training") as pbar:
        for step in range(num_train_steps):
            # Random mini-batch
            idx = torch.tensor(
                rng.choice(n_train, size=batch_size, replace=False),
                device=device)
            emb_b = train_emb[idx]
            mh_b  = train_mh[idx]
            ilm_b = train_ilm[idx]

            logits = linear(emb_b)
            loss_val = loss_fn(logits, mh_b, ilm_b)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if step % 32 == 0:
                pbar.set_postfix({"Loss": f"{loss_val.item():.8f}"})
            pbar.update(1)

    # Extract weights as numpy — match LinearClassifier format
    linear.eval()
    with torch.no_grad():
        beta      = linear.weight.T.cpu().numpy()   # (embedding_dim, num_classes)
        beta_bias = linear.bias.cpu().numpy()        # (num_classes,)

    # Evaluate on pre-loaded eval set
    from perch_hoplite.agile import classifier as _clf_mod
    pred_logits = [np.dot(eval_emb.cpu().numpy(), beta) + beta_bias]
    true_labels = [eval_mh.cpu().numpy()]
    got_ids     = [eval_idxs]

    pred_logits = np.concatenate(pred_logits, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    got_ids     = np.concatenate(got_ids, axis=0)

    from perch_hoplite.agile import metrics as _metrics
    labeled = np.where(true_labels.sum(axis=1) > 0)
    top_preds = np.argmax(pred_logits, axis=1)
    top1 = true_labels[np.arange(top_preds.shape[0]), top_preds][labeled].mean()
    rocs  = _metrics.roc_auc(logits=pred_logits, labels=true_labels, sample_threshold=1)
    cmaps = _metrics.cmap(logits=pred_logits, labels=true_labels, sample_threshold=1)

    eval_scores = {
        "top1_acc": float(top1),
        "roc_auc":  float(rocs["macro"]),
        "cmap":     float(cmaps["macro"]),
    }
    # Per-class F1 on the same held-out eval split (see src/f1_metrics.py). This is
    # the fallback trainer; if src/ isn't importable, skip F1 rather than fail.
    try:
        from src.f1_metrics import per_class_f1
        _f1 = per_class_f1(pred_logits, true_labels, target_labels)
        eval_scores["macro_f1"]     = _f1["macro_f1_at_0"]
        eval_scores["macro_f1_opt"] = _f1["macro_f1_opt"]
        eval_scores["per_class_f1"] = _f1
    except Exception as _f1_exc:
        log.warning("Skipping per-class F1 (f1_metrics unavailable): %s", _f1_exc)

    # Build LinearClassifier object matching the saved format
    # embedding_model_config is stored in the .pt file for reproducibility
    # but not used at inference time — pass a minimal placeholder
    from ml_collections import config_dict as _cd
    _emb_cfg = _cd.ConfigDict()
    lin_cls = _clf_mod.LinearClassifier(
        beta=beta,
        beta_bias=beta_bias,
        classes=target_labels,
        embedding_model_config=_emb_cfg,
    )
    return lin_cls, eval_scores


def _torch_write_inference_csv(
    db,
    linear_classifier,
    output_csv: str,
    logit_threshold: float = 0.0,
    pad_end_s: float = 0.0,
):
    """Write inference results to CSV — pure numpy, no TF."""
    import numpy as np
    import csv
    import struct

    beta      = linear_classifier.beta       # (embedding_dim, num_classes)
    beta_bias = linear_classifier.beta_bias  # (num_classes,)
    labels    = linear_classifier.classes

    con = __import__("sqlite3").connect(
        __import__("os").path.join(str(db.db_path), "hoplite.sqlite"))

    rows = con.execute("""
        SELECT w.id, r.filename, w.offsets
        FROM windows w JOIN recordings r ON r.id = w.recording_id
        ORDER BY r.filename, w.offsets
    """).fetchall()

    # Batch score all embeddings
    all_ids = [r[0] for r in rows]
    emb_matrix = db.get_embeddings_batch(all_ids).astype(np.float32)
    all_scores = emb_matrix @ beta.astype(np.float32) + beta_bias.astype(np.float32)

    written = 0
    label_counts = {}
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "project", "filename", "window_start",
                         "window_end", "label", "logits"])
        for i, (wid, fname, off_blob) in enumerate(rows):
            scores = all_scores[i]
            best_idx   = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score < logit_threshold:
                continue
            if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
                start_s, end_s = struct.unpack_from("<dd", off_blob)
            else:
                start_s, end_s = 0.0, 5.0
            best_label = labels[best_idx]
            writer.writerow([wid, "", fname,
                             round(start_s, 3), round(end_s + pad_end_s, 3),
                             best_label, round(best_score, 7)])
            written += 1
            label_counts[best_label] = label_counts.get(best_label, 0) + 1

    con.close()
    return written, label_counts


def _require_perch():
    try:
        from perch_hoplite.agile import (
            audio_loader,
            embedding_display, source_info,
        )
        from perch_hoplite.db import (
            brutalism, interface, score_functions,
            search_results, sqlite_usearch_impl,
        )
        import numpy as np
    except ImportError as exc:
        log.error(
            "perch-hoplite not installed. Run: pip install perch-hoplite\n"
            "Error: %s", exc,
        )
        sys.exit(1)
    # perch_hoplite.agile.classifier imports TF unconditionally at module level.
    # Inject a minimal mock so the import succeeds without TF installed.
    # LinearClassifier and train_linear_classifier only use numpy/sklearn at runtime.
    if _SRC_AVAILABLE:
        _injected_tf_mock = _src_inject_tf_mock()
    else:
        # Inline fallback
        import types as _types
        if 'tensorflow' not in sys.modules:
            import importlib.machinery as _imach
            _tf_mock = _types.ModuleType('tensorflow')
            _tf_mock.__spec__ = _imach.ModuleSpec('tensorflow', loader=None)
            _tf_mock.__version__ = '0.0.0-mock'
            _tf_mock.Tensor = object
            _tf_mock.keras = _types.ModuleType('tensorflow.keras')
            _tf_mock.keras.__spec__ = _imach.ModuleSpec('tensorflow.keras', loader=None)
            _tf_mock.keras.Model = object
            _tf_mock.keras.layers = _types.ModuleType('tensorflow.keras.layers')
            _tf_mock.keras.optimizers = _types.ModuleType('tensorflow.keras.optimizers')
            _tf_mock.keras.losses = _types.ModuleType('tensorflow.keras.losses')
            sys.modules['tensorflow'] = _tf_mock
            sys.modules['tensorflow.keras'] = _tf_mock.keras
            sys.modules['tensorflow.keras.layers'] = _tf_mock.keras.layers
            sys.modules['tensorflow.keras.optimizers'] = _tf_mock.keras.optimizers
            sys.modules['tensorflow.keras.losses'] = _tf_mock.keras.losses
            _injected_tf_mock = True
        else:
            _injected_tf_mock = False
    try:
        from perch_hoplite.agile import classifier, classifier_data
    except Exception as _e:
        log.warning("Could not import perch_hoplite.agile.classifier: %s", _e)
        classifier = classifier_data = None
    # model_configs intentionally NOT imported here — it triggers TensorFlow.
    model_configs = None
    return (audio_loader, classifier, classifier_data,
            embedding_display, source_info,
            brutalism, interface, score_functions,
            search_results, sqlite_usearch_impl,
            model_configs, np)


def _require_gradio():
    try:
        import gradio as gr
        return gr
    except ImportError:
        log.error(
            "Gradio is not installed. Install with: pip install gradio\n"
            "Or use --output-csv to write results without the GUI."
        )
        sys.exit(1)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError as exc:
        log.error("matplotlib not available: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="phase2_classify.py",
        description="Perch Hoplite Phase 2 — Search, Label, Train, Infer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    top.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    top.add_argument("--log-dir", default=None, help="Directory for log files.")

    sub = top.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- Common arguments ----
    def add_db(p):
        p.add_argument("--db-dir", "-d", required=True,
                       help="Path to Hoplite database directory.")

    def add_serve(p):
        p.add_argument("--spectrogram-type", default="linear",
            choices=["linear", "mel", "perch", "pcen"],
            help=(
                "Spectrogram display mode: "
                "'linear' = linear-frequency STFT, best for orca/dolphin (default); "
                "'mel' = mel-scale log power, 10 Hz floor, best for humpback; "
                "'perch' = exact Perch 2.0 frontend (what the model sees); "
                "'pcen' = PCEN mel, makes quiet calls pop."
            ))
        p.add_argument("--colormap", default=None,
            help=(
                "Override spectrogram colormap. Examples: "
                "'gray' (classic grayscale), 'gray_r' (inverted gray), "
                "'viridis' (blue-green-yellow), 'inferno' (default for linear/mel), "
                "'magma', 'plasma', 'cividis'. "
                "None = use per-mode default."
            ))
        p.add_argument("--serve", action="store_true", default=False,
                       help="Launch the Gradio labeling GUI.")
        p.add_argument("--port", type=int, default=7860,
                       help="Port for Gradio server (default: 7860).")
        p.add_argument("--host", default="0.0.0.0",
                       help="Bind address for Gradio server (default: 0.0.0.0).")
        p.add_argument("--share", action="store_true", default=False,
                       help="Create a public Gradio share link (requires internet).")

    # ---- search ----
    ps = sub.add_parser("search", help="Embed a query clip and find nearest neighbors.")
    add_db(ps)
    ps.add_argument("--query-audio", "-q", required=True,
                    help="Path or GCS URI to query audio clip (.wav/.flac).")
    ps.add_argument("--query-label", "-l", required=True,
                    help="Label string for this query class (e.g. 'orca_call').")
    ps.add_argument("--offset-s", type=float, default=0.0,
                    help="Start offset within the query audio (seconds, default: 0).")
    ps.add_argument("--window-s", type=float, default=5.0,
                    help="Window duration for query audio (seconds, default: 5).")
    ps.add_argument("--num-results", type=int, default=100,
                    help="Number of nearest neighbors to retrieve (default: 100).")
    ps.add_argument("--score-fn", choices=["dot", "cos", "neg_euclidean"], default="dot",
                    help="Similarity function (default: dot).")
    ps.add_argument("--exact", action="store_true", default=True,
                    help="Use exact brute-force search (default: True).")
    ps.add_argument("--approx", dest="exact", action="store_false",
                    help="Use approximate nearest-neighbor search (faster, less accurate).")
    ps.add_argument("--target-score", type=float, default=None,
                    help="If set, search for examples near this score (margin sampling).")
    ps.add_argument("--sample-rate-hz", type=int, default=None,
                    help="Audio loader sample rate override.")
    ps.add_argument("--output-csv", default=None,
                    help="Write search results to this CSV (recording_id, offset_s, score).")
    ps.add_argument("--plot-scores", default=None,
                    help="Save score histogram PNG to this path.")
    ps.add_argument("--annotator-id", default="analyst",
                    help="Annotator identifier attached to saved labels.")
    add_serve(ps)

    # ---- label ----
    pl = sub.add_parser("label", help="Import labels from a CSV into the DB.")
    add_db(pl)
    pl.add_argument("--labels-csv", required=True,
                    help=(
                        "CSV file with columns: "
                        "recording_id, offset_s, end_offset_s, label, label_type "
                        "(label_type: positive|negative|weak_negative)."
                    ))
    pl.add_argument("--annotator-id", default="analyst",
                    help="Annotator ID to attach to imported labels.")
    pl.add_argument("--dry-run", action="store_true",
                    help="Validate CSV without writing to DB.")

    # ---- train ----
    pt = sub.add_parser("train", help="Train a linear classifier on DB labels.")
    add_db(pt)
    pt.add_argument("--classifier-out", "-o", required=True,
                    help="Output path for the trained classifier (.pt file).")
    pt.add_argument("--target-labels", nargs="+", default=None,
                    help="Restrict training to these label classes (default: all).")
    pt.add_argument("--learning-rate", type=float, default=1e-3)
    pt.add_argument("--num-steps", type=int, default=128)
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--weak-neg-batch-size", type=int, default=128)
    pt.add_argument("--weak-neg-weight", type=float, default=0.05)
    pt.add_argument("--l2-mu", type=float, default=0.0)
    pt.add_argument("--train-ratio", type=float, default=0.9)
    pt.add_argument("--loss-fn", choices=["bce", "hinge"], default="bce")
    pt.add_argument("--seed", type=int, default=42)

    # ---- review ----
    pr = sub.add_parser(
        "review",
        help="Run classifier over DB, display top results for active-learning labeling.",
    )
    add_db(pr)
    pr.add_argument("--classifier", "-c", required=True,
                    help="Path to trained classifier .pt file.")
    pr.add_argument("--target-label", required=True,
                    help="Label class to review.")
    pr.add_argument("--num-results", type=int, default=100)
    pr.add_argument("--sample-size", type=int, default=10_000,
                    help="Randomly sample this many DB entries to search over.")
    pr.add_argument("--margin-target-score", type=float, default=None,
                    help="If set, use margin sampling around this logit.")
    pr.add_argument("--output-csv", default=None,
                    help="Write review results to this CSV.")
    pr.add_argument("--plot-scores", default=None,
                    help="Save score histogram PNG.")
    pr.add_argument("--annotator-id", default="analyst")
    pr.add_argument("--sample-rate-hz", type=int, default=None)
    pr.add_argument("--audio-dir", default=None,
        help="Override audio base path stored in DB (needed when DB was built "
             "on a different machine e.g. Colab). "
             "Example: /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04")
    pr.add_argument("--detections-csv", default=None,
        help="Path to an inference detections CSV (from 'infer' command). "
             "If set, skip scoring and load exactly these windows for review. "
             "Useful for reviewing and re-labeling a set of detections.")
    pr.add_argument("--detections-offset", type=int, default=0,
        help="Skip this many rows in --detections-csv before taking --num-results. "
             "Use to page through detections in batches. "
             "Example: --detections-offset 0 --num-results 25 (batch 1), "
             "--detections-offset 25 --num-results 25 (batch 2), etc.")
    pr.add_argument("--classes", default=None,
        help="Comma-separated list of annotation class labels to show as "
             "radio buttons. Default: 'positive,negative,unlabeled'. "
             "Example: --classes orca_call,dolphin_call,other,unlabeled")
    add_serve(pr)

    # ---- infer ----
    pi = sub.add_parser("infer", help="Run inference over all embeddings and save CSV.")
    add_db(pi)
    pi.add_argument("--classifier", "-c", required=True,
                    help="Path to trained classifier .pt file.")
    pi.add_argument("--output-csv", "-o", required=True,
                    help="Output CSV path for detections.")
    pi.add_argument("--logit-threshold", type=float, default=0.0,
                    help="Minimum logit to include in output (default: 0.0).")
    pi.add_argument("--labels", nargs="+", default=None,
                    help="Restrict inference to these label classes (default: all).")
    pi.add_argument("--plot-distribution", default=None,
                    help="Save logit distribution histogram PNG.")

    # ---- stats ----
    pst = sub.add_parser("stats", help="Print DB statistics.")
    add_db(pst)

    return top


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _patch_usearch_get_embeddings_batch(db) -> None:
    """Patch the USearch index .get() method to handle API version differences.

    Newer USearch (>=2.9) changed index.get(keys) to always return a tuple of
    1-D arrays instead of a single stacked np.ndarray.  perch-hoplite 1.0.1
    expects the old behavior and raises RuntimeError on the new API.

    threaded_brute_search spawns threads that each call db.ui.get() directly,
    bypassing any patch on db.get_embeddings_batch.  So we patch db.ui.get()
    itself — the USearch index object — which IS shared across threads.
    """
    import numpy as np

    ui = getattr(db, "ui", None)
    if ui is None:
        log.warning("DB has no .ui attribute — cannot patch USearch get().")
        return

    original_get = ui.get  # bound method on the index

    def patched_get(keys, *args, **kwargs):
        result = original_get(keys, *args, **kwargs)
        # Old API: returns np.ndarray of shape (n_keys, dim) — pass through
        if isinstance(result, np.ndarray):
            return result
        # New API: returns a tuple of 1-D arrays — stack into 2-D ndarray
        if isinstance(result, tuple):
            arrays = [arr for arr in result if isinstance(arr, np.ndarray)]
            if arrays:
                return np.stack(arrays, axis=0)
        # Unexpected — return as-is and let caller handle it
        return result

    ui.get = patched_get
    log.debug("Applied USearch index.get() compatibility patch.")


def load_db(db_dir: str):
    from perch_hoplite.db import sqlite_usearch_impl
    db_dir = str(db_dir)
    log.info("Loading database from %s", db_dir)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir)
    count = db.count_embeddings()
    log.info("Database loaded — %d embeddings", count)
    if count == 0:
        log.warning("Database contains zero embeddings. Run phase1_embed.py first.")
    _patch_usearch_get_embeddings_batch(db)
    return db


def _get_source(db, window_id):
    """Retrieve embedding source metadata (filename + offsets) for a window.

    Tries the known perch-hoplite method names in order, then falls back to
    a direct SQLite query using the db_config path stored on the DB object.
    """
    # Try every known method name
    for method_name in (
        "get_source_by_id",       # perch-hoplite 1.0.x
        "get_embedding_source",   # older builds
        "get_window_source",
        "get_source",
    ):
        method = getattr(db, method_name, None)
        if method is not None:
            try:
                return method(window_id)
            except Exception:
                pass

    # Last resort: direct SQLite query.
    # SQLiteUSearchDB stores its path in db_config.db_path
    import sqlite3 as _sqlite3, os as _os
    db_path_str = None
    # Try db_config path (the real location in SQLiteUSearchDB)
    db_cfg = getattr(db, "db_config", None)
    if db_cfg is not None:
        db_path_str = str(getattr(db_cfg, "db_path", None) or "")
    # Fallback: other attribute names
    if not db_path_str:
        for attr in ("db_path", "_db_path", "path", "sqlite_path"):
            v = getattr(db, attr, None)
            if v:
                db_path_str = str(v)
                break

    if not db_path_str:
        available = [a for a in dir(db)
                     if not a.startswith("__")
                     and ("path" in a.lower() or "dir" in a.lower()
                          or "source" in a.lower() or "config" in a.lower())]
        raise RuntimeError(
            f"Cannot find SQLite path on {type(db).__name__}. "
            f"Relevant attrs: {available}"
        )

    # db_path_str may be the directory; find the sqlite file inside it
    sqlite_file = db_path_str
    if _os.path.isdir(sqlite_file):
        for fname in ("hoplite.sqlite", "hoplite.db", "db.sqlite"):
            candidate = _os.path.join(sqlite_file, fname)
            if _os.path.isfile(candidate):
                sqlite_file = candidate
                break

    con = _sqlite3.connect(sqlite_file)
    # Schema: windows(id, recording_id, offsets)
    #         recordings(id, filename, datetime, deployment_id)
    #         deployments(id, name, project, ...)
    # offsets is stored as a FLOAT_LIST blob — two little-endian float64s.
    row = con.execute("""
        SELECT r.filename, d.project, w.offsets
        FROM windows w
        JOIN recordings r ON r.id = w.recording_id
        LEFT JOIN deployments d ON d.id = r.deployment_id
        WHERE w.id = ?
    """, (int(window_id),)).fetchone()
    con.close()

    if row is None:
        raise KeyError(f"window_id {window_id} not found in DB at {sqlite_file}")

    filename, project, offsets_blob = row

    # Decode FLOAT_LIST blob: two little-endian float64 values (start_s, end_s)
    import struct as _struct
    if isinstance(offsets_blob, (bytes, bytearray)) and len(offsets_blob) >= 16:
        start_s, end_s = _struct.unpack_from("<dd", offsets_blob, 0)
    elif isinstance(offsets_blob, (bytes, bytearray)) and len(offsets_blob) >= 8:
        start_s = _struct.unpack_from("<d", offsets_blob, 0)[0]
        end_s = start_s + 5.0
    else:
        # Fallback: offsets_blob might be a string like "[0.0, 5.0]"
        try:
            import json as _json
            vals = _json.loads(str(offsets_blob))
            start_s, end_s = float(vals[0]), float(vals[1])
        except Exception:
            start_s, end_s = 0.0, 5.0

    class _EmbSource:
        source_id    = filename
        dataset      = project or ""
        offsets      = (start_s, end_s)
        recording_id = None
    return _EmbSource()


def load_model_from_db(db):
    """Load the Perch V2 embedding model for a given DB.

    Delegates to src.torch_model.load_model_from_db when available,
    otherwise falls back to the inline implementation.
    """
    if _SRC_AVAILABLE:
        return _src_load_model_from_db(db, cuda_available_fn=_cuda_available)
    # Inline fallback
    import os as _os2
    from perch_hoplite.agile import source_info
    db_model_config = db.get_metadata("model_config")
    embed_config    = db.get_metadata("audio_sources")
    audio_sources   = source_info.AudioSources.from_config_dict(embed_config)
    model_key       = db_model_config.model_key
    use_tf = _os2.environ.get("PERCH_USE_TF", "0") == "1"
    if not use_tf and model_key in ("taxonomy_model_tf", "perch_torch"):
        import sys as _sys2
        _pytorch_dir = _os2.path.expanduser("~/perch-pytorch")
        if _pytorch_dir not in _sys2.path:
            _sys2.path.insert(0, _pytorch_dir)
        try:
            from perch_hoplite_torch_adapter import PerchTorchModel
            embedding_model = PerchTorchModel(
                weights_dir=_os2.path.join(_pytorch_dir, "perch_weights"),
                exact_mel=_os2.path.join(_pytorch_dir, "const__pad1_output_0.npy"),
                device="cuda" if _cuda_available() else "cpu",
            )
            log.info("Loaded embedding model: PerchTorchModel (PyTorch, no TF)")
        except ImportError as e:
            log.warning("PerchTorchModel not available (%s); falling back to TF", e)
            use_tf = True
    if use_tf:
        from perch_hoplite.zoo import model_configs
        model_class = model_configs.get_model_class(model_key)
        embedding_model = model_class.from_config(db_model_config.model_config)
        log.info("Loaded embedding model: %s (TensorFlow)", model_key)
    return embedding_model, audio_sources


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def make_audio_loader(db, embedding_model, audio_sources, sample_rate_hz=None):
    """Return (loader_fn, sample_rate_hz, window_size_s) using the real
    perch-hoplite audio_loader.make_filepath_loader() API.

    The returned loader normalizes audio to -3 dBFS peak so clips are
    audible in the browser (MARS 32 kHz files have very low amplitude).
    """
    import numpy as _np
    from perch_hoplite.agile import audio_loader as _al

    if sample_rate_hz is None:
        sample_rate_hz = embedding_model.sample_rate
    window_size_s = getattr(embedding_model, "window_size_s", 5.0)

    _raw_loader = _al.make_filepath_loader(
        audio_sources=audio_sources,
        sample_rate_hz=sample_rate_hz,
        window_size_s=window_size_s,
    )

    _target_peak = 10 ** (-3.0 / 20)   # -3 dBFS ≈ 0.708

    def loader(source_id: str, offset_s: float):
        # Returns np.ndarray only — sample_rate_hz is fixed at construction time
        audio = _raw_loader(source_id, offset_s)
        peak = _np.abs(audio).max()
        if peak > 1e-6:
            audio = audio * (_target_peak / peak)
        return audio, sample_rate_hz

    return loader, sample_rate_hz, window_size_s


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


# ---------------------------------------------------------------------------
# Sub-command: stats
# ---------------------------------------------------------------------------

def cmd_stats(args) -> int:
    db = load_db(args.db_dir)
    from perch_hoplite.db import interface as iface
    from ml_collections import config_dict
    projects = db.get_all_projects()
    total = db.count_embeddings()
    log.info("=" * 60)
    log.info("DATABASE: %s", args.db_dir)
    log.info("Total embeddings : %d", total)
    for proj in projects:
        ids = db.match_window_ids(
            deployments_filter=config_dict.create(eq=dict(project=proj))
        )
        log.info("  Project %-36s  %d embeddings", proj, len(ids))
    ann_count = len(db.get_all_annotations())
    log.info("Total annotations: %d", ann_count)
    if ann_count > 0:
        try:
            pos = db.count_each_label(label_type=_LT.POSITIVE)
            neg = db.count_each_label(label_type=_LT.NEGATIVE)
            log.info("Positive labels: %s", dict(pos))
            log.info("Negative labels: %s", dict(neg))
        except Exception as exc:
            log.debug("Label count error: %s", exc)
    log.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: label (CSV import)
# ---------------------------------------------------------------------------

LABEL_TYPE_MAP = {
    "positive": None,  # resolved at runtime from interface module
    "negative": None,
    "weak_negative": None,
}

def cmd_label(args) -> int:
    db = load_db(args.db_dir)
    from perch_hoplite.db import interface as iface

    label_type_map = {
        "positive": _LT.POSITIVE,
        "negative": _LT.NEGATIVE,
        "weak_negative": getattr(interface.LabelType, "WEAK_NEGATIVE",
                         _LT.NEGATIVE),
    }

    csv_path = Path(args.labels_csv)
    if not csv_path.exists():
        log.error("Labels CSV not found: %s", csv_path)
        return 1

    inserted = 0
    skipped = 0
    errors = 0

    # Build filename -> integer recording_id map from DB embedding sources.
    # insert_annotation requires the integer primary key, not a filename string.
    # We scan all window IDs and collect (filename, recording_id) pairs.
    # This is the only reliable way in perch-hoplite 1.0.1 — there is no
    # get_all_recordings() method on the SQLiteUSearchDB.
    log.info("Building filename -> recording_id lookup from DB (scanning embeddings)...")
    filename_to_rec_id: dict[str, int] = {}
    try:
        all_ids = db.match_window_ids(limit=None)
        for wid in all_ids:
            src = _get_source(db, wid)
            fname = getattr(src, "source_id", None) or ""
            rec_id = getattr(src, "recording_id", None)
            if fname and rec_id is not None:
                filename_to_rec_id[fname] = int(rec_id)
                # Also index by stem (without extension) for flexible matching
                stem = fname.rsplit(".", 1)[0] if "." in fname else fname
                filename_to_rec_id[stem] = int(rec_id)
        log.info("  Found %d unique recordings in DB.", len({v for v in filename_to_rec_id.values()}))
    except Exception as exc:
        log.warning("Could not build filename lookup: %s — will try direct int cast.", exc)

    log.info("Reading labels from %s", csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"recording_id", "offset_s", "label", "label_type"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            log.error(
                "CSV missing required columns. Expected: %s  Got: %s",
                required_cols, reader.fieldnames,
            )
            return 1

        for i, row in enumerate(reader):
            try:
                lt_str = row["label_type"].strip().lower()
                lt = label_type_map.get(lt_str)
                if lt is None:
                    log.warning("Row %d: unknown label_type '%s', skipping.", i, lt_str)
                    skipped += 1
                    continue

                csv_filename = row["recording_id"].strip()
                # Resolve to integer recording_id
                rec_int_id = filename_to_rec_id.get(csv_filename)
                if rec_int_id is None:
                    stem = csv_filename.rsplit(".", 1)[0] if "." in csv_filename else csv_filename
                    rec_int_id = filename_to_rec_id.get(stem)
                if rec_int_id is None:
                    # File not in this DB — expected when label CSV covers more
                    # dates than the current DB (e.g. full-month labels on a
                    # single-day DB).
                    skipped += 1
                    continue

                offset_s = float(row["offset_s"])
                end_offset_s = float(row.get("end_offset_s") or offset_s + 5.0)
                offsets = (offset_s, end_offset_s)

                if args.dry_run:
                    log.debug(
                        "DRY RUN row %d: %s (rec_id=%d) offset=%.2f label=%s type=%s",
                        i, csv_filename, rec_int_id, offset_s, row["label"], lt_str,
                    )
                    inserted += 1
                    continue

                db.insert_annotation(
                    recording_id=rec_int_id,
                    offsets=offsets,
                    label=row["label"].strip(),
                    label_type=lt,
                    provenance=f"csv_import:{args.annotator_id}",
                    handle_duplicates="update",
                )
                inserted += 1

            except Exception as exc:
                log.warning("Row %d error: %s  Row data: %s", i, exc, row)
                errors += 1

    action = "DRY RUN — would insert" if args.dry_run else "Inserted"
    log.info("%s %d labels (%d skipped, %d errors).", action, inserted, skipped, errors)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: search
# ---------------------------------------------------------------------------

def _run_search(db, embedding_model, args_query_audio, args_offset_s,
                args_window_s, args_sample_rate_hz, args_num_results,
                args_score_fn, args_exact, args_target_score, np):
    """Core search logic; returns (results, all_scores, sample_rate_hz, window_size_s)."""
    from perch_hoplite.db import brutalism, score_functions, search_results
    from perch_hoplite.agile import embedding_display

    sr = args_sample_rate_hz or embedding_model.sample_rate
    window_size_s = getattr(embedding_model, "window_size_s", 5.0)

    log.info("Loading query audio: %s", args_query_audio)
    query_display = embedding_display.QueryDisplay(
        uri=args_query_audio,
        offset_s=args_offset_s,
        window_size_s=args_window_s or window_size_s,
        sample_rate_hz=sr,
    )

    log.info("Embedding query audio...")
    audio_window = query_display.get_audio_window()
    query_embedding = embedding_model.embed(audio_window).embeddings[0, 0]
    log.info("Query embedding shape: %s", query_embedding.shape)

    score_fn = score_functions.get_score_fn(args_score_fn, target_score=args_target_score)

    log.info(
        "Searching DB (exact=%s, num_results=%d, score_fn=%s)...",
        args_exact, args_num_results, args_score_fn,
    )
    t0 = time.monotonic()
    if args_exact:
        results_obj = brutalism.threaded_brute_search(
            db, query_embedding, args_num_results, score_fn=score_fn
        )
        all_scores = np.array([r.sort_score for r in results_obj.search_results])
    else:
        ann_matches = db.ui.search(query_embedding, count=args_num_results)
        results_obj = search_results.TopKSearchResults(top_k=args_num_results)
        for k, dist in zip(ann_matches.keys, ann_matches.distances):
            results_obj.update(search_results.SearchResult(k, dist))
        all_scores = np.array([r.sort_score for r in results_obj.search_results])

    elapsed = time.monotonic() - t0
    log.info(
        "Search complete in %.2fs — %d results, score range [%.4f, %.4f]",
        elapsed,
        len(results_obj.search_results),
        float(all_scores.min()) if len(all_scores) else 0,
        float(all_scores.max()) if len(all_scores) else 0,
    )
    return results_obj, all_scores, sr, window_size_s


def _write_search_csv(results_obj, db, output_csv: str) -> None:
    """Write search results (recording_id, offset_s, score) to CSV."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "recording_id", "offset_s", "end_offset_s", "score"])
        for r in results_obj.search_results:
            wid = r.window_id
            source = _get_source(db, wid)
            writer.writerow([
                int(wid),
                source.source_id if hasattr(source, "source_id") else str(source),
                getattr(source, "offsets", (None, None))[0],
                getattr(source, "offsets", (None, None))[1],
                f"{r.sort_score:.6f}",
            ])
            count += 1
    log.info("Wrote %d search results to %s", count, out_path)


def _save_histogram(all_scores, hit_scores, output_path: str, title: str) -> None:
    """Save a score distribution histogram to a PNG file."""
    plt, np = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(all_scores, bins=100, color="#1a6fa0", alpha=0.7, label="All scores")
    ax.scatter(
        hit_scores, np.zeros_like(hit_scores),
        marker="|", color="red", alpha=0.7, s=200, label="Top hits",
    )
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Score histogram saved to %s", out)


def cmd_search(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     embedding_display_mod, source_info_mod,
     brutalism_mod, interface_mod, score_functions_mod,
     search_results_mod, sqlite_usearch_impl_mod,
     model_configs_mod, np) = _require_perch()

    db = load_db(args.db_dir)
    embedding_model, audio_sources = load_model_from_db(db)

    if getattr(args, "audio_dir", None):
        from perch_hoplite.agile import source_info as _si
        patched_globs = []
        _globs = getattr(audio_sources, 'audio_globs', getattr(audio_sources, 'audio_sources', []))
        for ag in _globs:
            patched = _si.AudioSourceConfig(
                dataset_name=ag.dataset_name,
                base_path=args.audio_dir,
                file_glob=ag.file_glob,
                min_audio_len_s=ag.min_audio_len_s,
                target_sample_rate_hz=ag.target_sample_rate_hz,
                shard_len_s=ag.shard_len_s,
            )
            patched_globs.append(patched)
        audio_sources = _si.AudioSources(tuple(patched_globs))
        log.info("Audio base path overridden to: %s", args.audio_dir)

    audio_filepath_loader, sr, window_size_s = make_audio_loader(
        db, embedding_model, audio_sources, args.sample_rate_hz
    )

    results_obj, all_scores, sr, window_size_s = _run_search(
        db, embedding_model,
        args.query_audio, args.offset_s, args.window_s,
        sr, args.num_results, args.score_fn,
        args.exact, args.target_score, np,
    )

    if args.output_csv:
        _write_search_csv(results_obj, db, args.output_csv)

    if args.plot_scores:
        hit_scores = [r.sort_score for r in results_obj.search_results]
        _save_histogram(all_scores, hit_scores, args.plot_scores,
                        f"Search results — {args.query_label}")

    if args.serve:
        _gui_fn = _src_launch_labeling_gui if _SRC_AVAILABLE else _launch_labeling_gui
        _gui_fn(
            db=db,
            results_obj=results_obj,
            audio_filepath_loader=audio_filepath_loader,
            sample_rate_hz=sr,
            query_label=args.query_label,
            annotator_id=args.annotator_id,
            host=args.host,
            port=args.port,
            share=args.share,
            db_dir=args.db_dir,
            classifier_path=None,
            spectrogram_type=getattr(args, "spectrogram_type", "linear"),
            colormap=getattr(args, "colormap", None),
            audio_base_dir=getattr(args, "audio_dir", ""),
        )
    else:
        log.info(
            "Search complete. Use --serve to launch the labeling GUI, "
            "or --output-csv to export results."
        )

    return 0


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def _coerce_eval_score(v):
    """Make an eval_scores value JSON-safe. Nested dicts (e.g. per_class_f1) pass
    through unchanged; scalars / 0-d arrays are floatified as before."""
    if isinstance(v, dict):
        return v
    return float(v.flat[0] if hasattr(v, "flat") else v)


def cmd_train(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     *_rest) = _require_perch()
    import numpy as np

    db = load_db(args.db_dir)

    data_manager = classifier_data_mod.AgileDataManager(
        target_labels=args.target_labels,
        db=db,
        train_ratio=args.train_ratio,
        min_eval_examples=1,
        batch_size=args.batch_size,
        weak_negatives_batch_size=args.weak_neg_batch_size,
        rng=np.random.default_rng(seed=args.seed),
    )

    target_labels = data_manager.get_target_labels()
    log.info("Training classifier for labels: %s", target_labels)
    log.info(
        "  steps=%d  lr=%.1e  weak_neg_weight=%.3f  loss=%s  seed=%d",
        args.num_steps, args.learning_rate, args.weak_neg_weight,
        args.loss_fn, args.seed,
    )

    t0 = time.monotonic()
    _train_fn = _src_torch_train if _SRC_AVAILABLE else _torch_train_linear_classifier
    linear_classifier, eval_scores = _train_fn(
        data_manager=data_manager,
        learning_rate=args.learning_rate,
        weak_neg_weight=args.weak_neg_weight,
        num_train_steps=args.num_steps,
    )
    elapsed = time.monotonic() - t0

    log.info("Training complete in %s", _format_duration(elapsed))
    log.info("  top1_acc : %.4f", eval_scores.get("top1_acc", float("nan")))
    log.info("  roc_auc  : %.4f", eval_scores.get("roc_auc", float("nan")))
    log.info("  cmap     : %.4f", eval_scores.get("cmap", float("nan")))
    log.info("  macro_f1 : %.4f", eval_scores.get("macro_f1", float("nan")))

    out_path = Path(args.classifier_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    linear_classifier.save(str(out_path))
    log.info("Classifier saved to %s", out_path)

    # Save eval metrics JSON alongside the model
    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "labels": target_labels,
                "eval_scores": {k: _coerce_eval_score(v) for k, v in eval_scores.items()},
                "train_args": {
                    "num_steps": args.num_steps,
                    "learning_rate": args.learning_rate,
                    "weak_neg_weight": args.weak_neg_weight,
                    "batch_size": args.batch_size,
                    "train_ratio": args.train_ratio,
                    "seed": args.seed,
                },
            },
            f, indent=2,
        )
    log.info("Eval metrics written to %s", metrics_path)

    # Save training provenance record
    import sqlite3 as _sq3t
    ann_counts = {}
    try:
        _con = _sq3t.connect(os.path.join(args.db_dir, "hoplite.sqlite"))
        rows = _con.execute(
            "SELECT label, label_type, COUNT(*) FROM annotations GROUP BY label, label_type"
        ).fetchall()
        _con.close()
        ann_counts = {f"{r[0]}_type{r[1]}": r[2] for r in rows}
    except Exception:
        pass
    # Build full annotation list for provenance — every labeled window
    full_anns = []
    try:
        _con2 = _sq3t.connect(os.path.join(args.db_dir, "hoplite.sqlite"))
        ann_rows = _con2.execute("""
            SELECT a.id, r.filename, w.id as window_id, w.offsets,
                   a.label, a.label_type, a.provenance
            FROM annotations a
            JOIN recordings r ON r.id = a.recording_id
            LEFT JOIN windows w ON w.recording_id = a.recording_id
                AND w.offsets = a.offsets
        """).fetchall()
        import struct as _st2
        for row in ann_rows:
            ann_id, fname, wid, off_blob, label, ltype, prov = row
            if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
                s, e = _st2.unpack_from("<dd", off_blob)
            else:
                s, e = 0.0, 5.0
            full_anns.append({
                "annotation_id": ann_id,
                "window_id":     wid,
                "filename":      fname,
                "offset_s":      round(s, 3),
                "end_s":         round(e, 3),
                "label":         label,
                "label_type":    ltype,
                "provenance":    prov,
            })
        # Get audio sources from DB metadata
        src_row = _con2.execute(
            "SELECT value FROM hoplite_metadata WHERE key='audio_sources'"
        ).fetchone()
        audio_srcs = json.loads(src_row[0]).get("audio_globs", []) if src_row else []
        _con2.close()
    except Exception as _e:
        log.warning("Could not build full annotation list for provenance: %s", _e)
        full_anns = []
        audio_srcs = []

    _save_training_provenance(
        db_dir=args.db_dir,
        classifier_out=str(out_path),
        target_labels=target_labels,
        train_args={
            "num_steps":       args.num_steps,
            "learning_rate":   args.learning_rate,
            "weak_neg_weight": args.weak_neg_weight,
            "batch_size":      args.batch_size,
            "train_ratio":     args.train_ratio,
            "seed":            args.seed,
        },
        eval_scores={k: _coerce_eval_score(v) for k, v in eval_scores.items()},
        annotation_counts=ann_counts,
        elapsed_s=elapsed,
        full_annotations=full_anns,
        audio_sources=audio_srcs,
    )
    return 0


# ---------------------------------------------------------------------------
# Sub-command: review
# ---------------------------------------------------------------------------

def cmd_review(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     embedding_display_mod, source_info_mod,
     brutalism_mod, interface_mod, score_functions_mod,
     search_results_mod, sqlite_usearch_impl_mod,
     model_configs_mod, np) = _require_perch()

    db = load_db(args.db_dir)
    embedding_model, audio_sources = load_model_from_db(db)

    # If --audio-dir is given, override every base_path in audio_sources.
    # This is needed when the DB was built on a different machine (e.g. Colab)
    # and the audio files are now at a different path on this server.
    if getattr(args, "audio_dir", None):
        from perch_hoplite.agile import source_info as _si
        patched_globs = []
        _globs = getattr(audio_sources, 'audio_globs', getattr(audio_sources, 'audio_sources', []))
        for ag in _globs:
            patched = _si.AudioSourceConfig(
                dataset_name=ag.dataset_name,
                base_path=args.audio_dir,
                file_glob=ag.file_glob,
                min_audio_len_s=ag.min_audio_len_s,
                target_sample_rate_hz=ag.target_sample_rate_hz,
                shard_len_s=ag.shard_len_s,
            )
            patched_globs.append(patched)
        audio_sources = _si.AudioSources(tuple(patched_globs))
        log.info("Audio base path overridden to: %s", args.audio_dir)

    audio_filepath_loader, sr, _ = make_audio_loader(
        db, embedding_model, audio_sources, args.sample_rate_hz
    )

    log.info("Loading classifier from %s", args.classifier)
    linear_classifier = classifier_mod.LinearClassifier.load(args.classifier)

    # Retrieve the weight vector for the target label
    try:
        target_labels = linear_classifier.get_labels()
    except Exception:
        # Fallback: try reading from companion metrics JSON
        metrics_path = Path(args.classifier).with_suffix(".metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                target_labels = json.load(f).get("labels", [])
        else:
            log.error(
                "Cannot determine classifier labels. "
                "Ensure the .metrics.json companion file exists."
            )
            return 1

    if args.target_label not in target_labels:
        log.error(
            "Target label '%s' not found in classifier labels: %s",
            args.target_label, target_labels,
        )
        return 1

    idx = target_labels.index(args.target_label)

    class_query = linear_classifier.beta[:, idx]
    bias_val = float(linear_classifier.beta_bias[idx])
    log.info(
        "Using classifier weight vector for label '%s' (index %d)",
        args.target_label, idx,
    )

    # Compute classifier scores directly on the main thread using
    # get_embeddings_batch (our patched version handles the USearch API
    # version difference). Fetch a random sample, score with dot product,
    # keep top-k. This is single-threaded so avoids the thread-safety issue
    # with threaded_brute_search.
    import numpy as _np_rev

    # ── If --detections-csv provided, load those windows directly ────────
    detections_csv = getattr(args, "detections_csv", None)
    if detections_csv:
        import csv as _csv
        log.info("Loading detections from CSV: %s", detections_csv)
        # Build a lookup: filename+offset -> window_id from DB
        import sqlite3 as _sq3r, struct as _str2
        _dbpath = None
        _cfg = getattr(db, "db_config", None)
        if _cfg: _dbpath = str(getattr(_cfg, "db_path", "") or "")
        if not _dbpath:
            for _a in ("db_path", "_db_path", "path"):
                _v = getattr(db, _a, None)
                if _v: _dbpath = str(_v); break
        if _dbpath and os.path.isdir(_dbpath):
            _dbpath = os.path.join(_dbpath, "hoplite.sqlite")
        _con_r = _sq3r.connect(_dbpath)

        results_obj = search_results_mod.TopKSearchResults(
            top_k=max(10000, args.num_results))
        det_rows = []
        with open(detections_csv, newline="") as _f:
            for row in _csv.DictReader(_f):
                det_rows.append(row)

        # Limit to num_results
        offset = getattr(args, "detections_offset", 0) or 0
        det_rows = det_rows[offset:]
        det_rows = det_rows[:args.num_results] if args.num_results else det_rows
        log.info("Loading detections %d–%d (offset=%d, n=%d)",
                 offset, offset + len(det_rows), offset, len(det_rows))
        matched = 0
        for row in det_rows:
            fname   = row["filename"]
            start_s = float(row["window_start"])
            score   = float(row["logits"])
            off_enc = _str2.pack("<dd", start_s, start_s + 5.0)
            # Find window_id by filename + offset
            _rec = _con_r.execute(
                "SELECT id FROM recordings WHERE filename=?", (fname,)
            ).fetchone()
            if _rec is None:
                log.warning("File not found in DB: %s", fname)
                continue
            _win = _con_r.execute(
                "SELECT id FROM windows WHERE recording_id=? AND offsets=?",
                (_rec[0], off_enc)
            ).fetchone()
            if _win is None:
                log.warning("Window not found: %s @ %.1fs", fname, start_s)
                continue
            results_obj.update(
                search_results_mod.SearchResult(int(_win[0]), float(score)))
            matched += 1
        _con_r.close()
        log.info("Matched %d / %d detections to DB windows.", matched, len(det_rows))
        all_scores = [r.sort_score for r in results_obj.search_results]

    else:
        # ── Standard scoring path ────────────────────────────────────────
        log.info("Scoring embeddings with classifier weight vector (%d dims)...",
                 len(class_query))

        all_ids = db.match_window_ids(limit=None)
        total = len(all_ids)

        # Random subsample
        sample_n = args.sample_size if args.sample_size and args.sample_size > 0 else total
        sample_n = min(sample_n, total)
        if sample_n < total:
            chosen = _np_rev.random.default_rng().choice(total, size=sample_n, replace=False)
            sample_ids = [all_ids[i] for i in sorted(chosen)]
        else:
            sample_ids = all_ids
        log.info("Scoring %d / %d embeddings...", sample_n, total)

        # Fetch all sampled embeddings in one batch on main thread
        emb_matrix = db.get_embeddings_batch(sample_ids)   # (N, D) float16
        emb_f32 = emb_matrix.astype(_np_rev.float32)
        query_f32 = _np_rev.array(class_query, dtype=_np_rev.float32)
        all_scores = emb_f32 @ query_f32 + bias_val        # (N,) logit scores

        # Build TopKSearchResults.
        results_obj = search_results_mod.TopKSearchResults(top_k=args.num_results)
        margin_target = args.margin_target_score
        for wid, score in zip(sample_ids, all_scores):
            if margin_target is not None:
                sort_score = -abs(float(score) - margin_target)
            else:
                sort_score = float(score)
            results_obj.update(search_results_mod.SearchResult(int(wid), sort_score))

        # Restore actual logit scores for display (not the sort key)
        score_map = {int(wid): float(s) for wid, s in zip(sample_ids, all_scores)}
        for r in results_obj.search_results:
            r.sort_score = score_map.get(r.window_id, r.sort_score)

        log.info("Scoring complete: %d candidates, top-%d selected (margin_target=%s).",
                 sample_n, args.num_results, margin_target)

    # Log the window IDs selected for review so they can be reconstructed
    selected_ids = [r.window_id for r in results_obj.search_results]
    log.info("Selected window IDs: %s", selected_ids)


    hit_scores = [r.sort_score for r in results_obj.search_results]
    log.info(
        "Review search: %d results, score range [%.4f, %.4f]",
        len(results_obj.search_results),
        min(hit_scores) if hit_scores else 0,
        max(hit_scores) if hit_scores else 0,
    )

    if args.output_csv:
        _write_search_csv(results_obj, db, args.output_csv)

    if args.plot_scores:
        _save_histogram(all_scores, hit_scores, args.plot_scores,
                        f"Classifier review — {args.target_label}")

    if args.serve:
        # Parse --classes if provided
        custom_classes = None
        if getattr(args, "classes", None):
            custom_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        # Build detections_info for header display
        _det_info_dict = None
        if getattr(args, "detections_csv", None):
            import csv as _csv2
            with open(args.detections_csv) as _f2:
                _total_dets = sum(1 for _ in _csv2.DictReader(_f2))
            _det_info_dict = {
                "offset": getattr(args, "detections_offset", 0) or 0,
                "total":  _total_dets,
                "csv":    args.detections_csv,
            }
        _gui_fn = _src_launch_labeling_gui if _SRC_AVAILABLE else _launch_labeling_gui
        _gui_fn(
            db=db,
            results_obj=results_obj,
            audio_filepath_loader=audio_filepath_loader,
            sample_rate_hz=sr,
            query_label=args.target_label,
            annotator_id=args.annotator_id,
            host=args.host,
            port=args.port,
            share=args.share,
            db_dir=args.db_dir,
            classifier_path=getattr(args, "classifier", None),
            label_classes=custom_classes,
            detections_info=_det_info_dict,
            spectrogram_type=getattr(args, "spectrogram_type", "linear"),
            colormap=getattr(args, "colormap", None),
            audio_base_dir=getattr(args, "audio_dir", ""),
        )

    return 0


# ---------------------------------------------------------------------------
# Sub-command: infer
# ---------------------------------------------------------------------------

def cmd_infer(args) -> int:
    (audio_loader_mod, classifier_mod, *_rest) = _require_perch()

    db = load_db(args.db_dir)

    log.info("Loading classifier from %s", args.classifier)
    linear_classifier = classifier_mod.LinearClassifier.load(args.classifier)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Delete existing CSV before inference — perch-hoplite's csv_worker_fn
    # opens in append mode, so a stale file causes row multiplication
    if out_path.exists():
        out_path.unlink()
        log.info("Removed existing output CSV (append-mode safeguard)")

    log.info(
        "Running inference (logit_threshold=%.3f, labels=%s)...",
        args.logit_threshold, args.labels or "all",
    )

    if _SRC_AVAILABLE:
        result = _src_run_inference(
            db=db,
            linear_classifier=linear_classifier,
            output_csv=str(out_path),
            logit_threshold=args.logit_threshold,
            labels=args.labels,
            classifier_mod=classifier_mod,
        )
        elapsed = result["elapsed_s"]
        detection_count = result["detection_count"]
    else:
        import tempfile, shutil
        from collections import Counter
        t0 = time.monotonic()
        tmp_csv = tempfile.mktemp(suffix=".csv", dir="/tmp", prefix="perch_infer_")
        classifier_mod.write_inference_csv(
            linear_classifier, db, tmp_csv,
            args.logit_threshold, labels=args.labels)
        shutil.copy(tmp_csv, str(out_path))
        os.unlink(tmp_csv)
        elapsed = time.monotonic() - t0
        detection_count = sum(1 for _ in open(str(out_path))) - 1
        log.info("Inference complete in %s", _format_duration(elapsed))
        log.info("Total detections written: %d", detection_count)

    if args.plot_distribution:
        try:
            import pandas as pd
            import seaborn as sns
            plt_mod, np_mod = _require_matplotlib()
            df = pd.read_csv(out_path)
            fig, ax = plt_mod.subplots(figsize=(12, 5))
            for lbl in df["label"].unique():
                subset = df[df["label"] == lbl]["logits"]
                ax.hist(subset, bins=50, alpha=0.6, label=lbl)
            ax.set_title("Inference logit distribution by class")
            ax.set_xlabel("Logit")
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            dist_path = Path(args.plot_distribution)
            dist_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(dist_path), dpi=150)
            plt_mod.close(fig)
            log.info("Distribution plot saved to %s", dist_path)
        except ImportError as exc:
            log.warning("Could not generate distribution plot: %s", exc)

    log.info("Results written to %s", out_path)
    return 0


# ---------------------------------------------------------------------------
# Gradio labeling GUI
# ---------------------------------------------------------------------------

def _launch_labeling_gui(
    db,
    results_obj,
    audio_filepath_loader,
    sample_rate_hz: int,
    query_label: str,
    annotator_id: str,
    host: str,
    port: int,
    share: bool,
    db_dir: str = "",
    classifier_path: str | None = None,
    label_classes: list[str] | None = None,
    detections_info: dict | None = None,
    spectrogram_type: str = "linear",
    colormap: str | None = None,
    audio_base_dir: str = "",
) -> None:
    """
    Launch a Gradio web app for interactive audio labeling.

    The app displays up to N search results; each result shows:
      - Waveform plot of the audio segment
      - An HTML5 audio player for playback
      - Positive / Negative / Unlabeled radio buttons

    On clicking "Save Labels", all annotations are written back to the DB.

    Access via http://<server-ip>:<port> from any browser on the LAN.
    """
    gr = _require_gradio()
    plt_mod, np_mod = _require_matplotlib()
    import io, base64, soundfile as sf

    from perch_hoplite.db import interface as iface

    log.info("Building Gradio labeling interface...")

    import datetime as _dt_gui
    session_id      = _dt_gui.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{annotator_id}"
    _db_dir_cap     = db_dir       # passed as parameter
    _classifier_cap = classifier_path  # passed as parameter
    log.info("Labeling session ID: %s", session_id)

    # Pre-load all result segments into memory.
    # make_filepath_loader returns loader(filename, offset_s, end_offset_s).
    segments = []
    for r in results_obj.search_results:
        wid = r.window_id
        try:
            source = _get_source(db, wid)
            recording_id = getattr(source, "source_id", str(wid))
            offsets = getattr(source, "offsets", (0.0, 5.0))
            offset_s = float(offsets[0]) if offsets else 0.0
            end_s    = float(offsets[1]) if offsets and len(offsets) > 1 else offset_s + 5.0
            audio, sr_actual = audio_filepath_loader(recording_id, offset_s)
        except Exception as exc:
            import traceback as _tb
            log.warning("Could not load audio for window %s: %s\n%s",
                        wid, exc, _tb.format_exc())
            continue
        segments.append({
            "window_id": int(wid),
            "recording_id": recording_id,
            "offset_s": offsets[0] if offsets else 0.0,
            "end_offset_s": offsets[1] if offsets else 5.0,
            "score": r.sort_score,
            "audio": audio,
            "sample_rate": sr_actual or sample_rate_hz,
        })

    log.info("Loaded %d audio segments for labeling.", len(segments))

    def _make_spectrogram_image(audio_array: "np.ndarray", sr: int,
                                spec_type: str = "linear",
                                highlight_start: float | None = None,
                                highlight_end: float | None = None,
                                colormap: str | None = None) -> str:
        if _SRC_AVAILABLE:
            return _src_make_spectrogram(audio_array, sr, spec_type,
                                         highlight_start, highlight_end,
                                         colormap)
        # Fallback: inline implementation (kept for safety)
        """Return a base64-encoded PNG spectrogram.

        spec_type options:
          "linear" — Linear-frequency STFT, 0–16 kHz (default).
                     Best for orca and dolphin (high-frequency clicks/whistles).
          "mel"    — Mel-scale spectrogram, 10 Hz–16 kHz floor, log power.
                     Best for humpback song (low-frequency, sustained).
          "perch"  — Exact Perch 2.0 frontend: 128-band mel, 60 Hz–16 kHz,
                     0.1·log(max(mel, 1e-5)), HTK scale, DC bin zeroed.
                     Shows exactly what the embedding model sees.
          "pcen"   — Per-Channel Energy Normalization on mel, 10 Hz floor.
                     Makes quiet signals pop against background noise.
        """
        import numpy as _np2

        # ── Compute time-frequency representation ─────────────────────────
        if spec_type == "linear":
            from scipy.signal import spectrogram as _spec
            nperseg = min(512, len(audio_array) // 4)
            f, t, Sxx = _spec(
                audio_array, fs=sr,
                nperseg=nperseg, noverlap=nperseg * 3 // 4,
                scaling="density",
            )
            S_plot = 10 * _np2.log10(Sxx + 1e-10)
            f_max = min(16000, sr // 2)
            f_mask = f <= f_max
            f_plot = f[f_mask]
            S_plot = S_plot[f_mask, :]
            ylabel = "Hz"
            title_suffix = "Linear STFT"
            cmap = "inferno"

        elif spec_type in ("mel", "pcen"):
            import librosa as _librosa
            n_fft = 512
            hop = n_fft // 4
            f_min = 10.0
            f_max = 16000.0
            n_mels = 128
            S = _librosa.feature.melspectrogram(
                y=audio_array.astype(_np2.float32), sr=sr,
                n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, fmin=f_min, fmax=f_max,
                power=2.0,
            )
            if spec_type == "pcen":
                S_plot = _librosa.pcen(
                    S * (2**31), sr=sr, hop_length=hop,
                    gain=0.98, bias=2, power=0.5, time_constant=0.4,
                    eps=1e-6,
                )
                title_suffix = "PCEN mel (10 Hz floor)"
                cmap = "magma"
            else:
                S_plot = _librosa.power_to_db(S, ref=_np2.max)
                title_suffix = "Mel spectrogram (10 Hz floor)"
                cmap = "inferno"
            # Mel band centers for y-axis
            f_plot = _librosa.mel_frequencies(
                n_mels=n_mels, fmin=f_min, fmax=f_max)
            t = _librosa.frames_to_time(
                _np2.arange(S_plot.shape[1]), sr=sr, hop_length=hop)
            ylabel = "Hz (mel)"

        elif spec_type == "perch":
            # Exact Perch 2.0 frontend:
            # 128-band mel, 60 Hz–16 kHz, HTK scale, DC bin zeroed
            # 0.1 · log(max(mel_energy, 1e-5))
            import librosa as _librosa
            n_fft  = 2048
            hop    = 320       # Perch uses 10ms hop at 32kHz
            n_mels = 128
            f_min  = 60.0
            f_max  = 16000.0
            S = _librosa.feature.melspectrogram(
                y=audio_array.astype(_np2.float32), sr=sr,
                n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, fmin=f_min, fmax=f_max,
                htk=True, power=1.0,    # power=1 → amplitude mel
            )
            S[0, :] = 0.0               # zero DC bin (Perch convention)
            S_plot  = 0.1 * _np2.log(_np2.maximum(S, 1e-5))
            f_plot  = _librosa.mel_frequencies(
                n_mels=n_mels, fmin=f_min, fmax=f_max, htk=True)
            t = _librosa.frames_to_time(
                _np2.arange(S_plot.shape[1]), sr=sr, hop_length=hop)
            ylabel = "Hz (Perch mel)"
            title_suffix = "Perch 2.0 frontend (what the model sees)"
            cmap = "viridis"

        else:
            raise ValueError(f"Unknown spec_type: {spec_type!r}. "
                             "Use 'linear', 'mel', 'perch', or 'pcen'.")

        # Use percentile-based normalization — robust against saturating noise
        # (fixed 60dB range clips broadband vessel noise to solid color)
        import numpy as _np_pct
        if spec_type == "pcen":
            vmax = float(_np_pct.percentile(S_plot, 99))
            vmin = float(_np_pct.percentile(S_plot,  1))
        else:
            vmax = float(_np_pct.percentile(S_plot, 99.5))
            vmin = vmax - 80.0   # 80dB range: shows quiet calls in noisy background

        # ── Plot ──────────────────────────────────────────────────────────
        fig, axes = plt_mod.subplots(
            2, 1, figsize=(7, 3),
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        )
        fig.patch.set_facecolor("#111827")

        ax_spec = axes[0]
        ax_spec.pcolormesh(
            t, f_plot, S_plot,
            vmin=vmin, vmax=vmax,
            cmap=cmap, shading="gouraud",
        )
        # Draw fiducial markers for the 5-second clip within a context window
        if highlight_start is not None and highlight_end is not None:
            # Semi-transparent yellow highlight band — draw OVER the spectrogram
            ax_spec.axvspan(highlight_start, highlight_end,
                            alpha=0.25, color="#facc15", zorder=10)
            # Bright yellow vertical lines at boundaries
            ax_spec.axvline(x=highlight_start, color="#facc15",
                            linewidth=2.5, linestyle="-", alpha=1.0, zorder=11)
            ax_spec.axvline(x=highlight_end,   color="#facc15",
                            linewidth=2.5, linestyle="-", alpha=1.0, zorder=11)
            # Label at bottom of spectrogram
            y_bot, y_top = ax_spec.get_ylim()
            mid_x = (highlight_start + highlight_end) / 2
            ax_spec.text(mid_x, y_bot + (y_top - y_bot) * 0.03,
                         "◄ 5s ►", color="#facc15", fontsize=8,
                         ha="center", va="bottom", fontweight="bold",
                         zorder=12,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="#0f172a", alpha=0.7,
                                   edgecolor="none"))
        ax_spec.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
        ax_spec.set_title(title_suffix, color="#64748b", fontsize=7, pad=2)
        ax_spec.tick_params(colors="#94a3b8", labelsize=7)
        ax_spec.set_facecolor("#111827")
        for spine in ax_spec.spines.values():
            spine.set_edgecolor("#334155")
        ax_spec.tick_params(bottom=False, labelbottom=False)

        ax_wave = axes[1]
        t_wave = _np2.linspace(0, len(audio_array) / sr, len(audio_array))
        ax_wave.plot(t_wave, audio_array, color="#38bdf8", linewidth=0.4)
        ax_wave.set_xlim([0, t_wave[-1]])
        ax_wave.set_xlabel("Time (s)", color="#94a3b8", fontsize=8)
        ax_wave.set_facecolor("#111827")
        ax_wave.tick_params(colors="#94a3b8", labelsize=7)
        for spine in ax_wave.spines.values():
            spine.set_edgecolor("#334155")
        ax_wave.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                    facecolor="#111827")
        plt_mod.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    def _make_audio_b64(audio_array: "np.ndarray", sr: int) -> str:
        """Return a base64-encoded WAV for HTML5 audio element."""
        if _SRC_AVAILABLE:
            return _src_make_audio_b64(audio_array, sr)
        buf = io.BytesIO()
        sf.write(buf, audio_array, sr, format="WAV")
        buf.seek(0)
        return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()

    def _load_30s_context(seg: dict):
        """Load a 30-second context window centered on the 5-second clip.

        Returns (spec_html: str, audio_tuple: (sr, np.ndarray) | None)
        """
        if _SRC_AVAILABLE:
            return _src_load_30s_context(seg, audio_base_dir, spectrogram_type)
        # Fallback: inline implementation
        import os as _os
        import soundfile as _sf_ctx
        import numpy as _np_ctx
        fname = seg["recording_id"].split("/")[-1]
        wav_path = _os.path.join(audio_base_dir, fname) if audio_base_dir else None
        if not wav_path or not _os.path.exists(wav_path):
            err = (f"<div style='color:#ef4444;font-size:11px;padding:8px;'>"
                   f"⚠ Audio file not found: {fname}</div>")
            return err, None
        try:
            sr_ctx = seg["sample_rate"]
            offset_s = seg["offset_s"]
            center_s = offset_s + 2.5
            file_info = _sf_ctx.info(wav_path)
            file_dur  = file_info.duration
            ctx_start = max(0.0, center_s - 15.0)
            ctx_end   = min(file_dur, ctx_start + 30.0)
            ctx_start = max(0.0, ctx_end - 30.0)
            start_smp = int(ctx_start * sr_ctx)
            end_smp   = int(ctx_end   * sr_ctx)
            audio_ctx, _ = _sf_ctx.read(wav_path, start=start_smp, stop=end_smp,
                                         dtype="float32", always_2d=False)
            ctx_spec_type   = "mel" if spectrogram_type == "linear" else spectrogram_type
            hl_start        = offset_s - ctx_start        # position in 30s window
            hl_end          = hl_start + 5.0
            spec_b64_ctx    = _make_spectrogram_image(audio_ctx, sr_ctx,
                                                      spec_type=ctx_spec_type,
                                                      highlight_start=hl_start,
                                                      highlight_end=hl_end)
            actual_dur = ctx_end - ctx_start
            clip_note  = "" if actual_dur >= 29.9 else f" (clipped to {actual_dur:.0f}s)"
            spec_html = (
                f"<div style='background:#0f172a;border:1px solid #334155;"
                f"border-radius:8px;padding:10px;margin-top:4px;'>"
                f"<span style='color:#fbbf24;font-size:10px;font-family:monospace;'>"
                f"▶ 30s context &nbsp; {ctx_start:.1f}s – {ctx_end:.1f}s{clip_note}</span>"
                f"<br><span style='color:#64748b;font-size:9px;'>"
                f"5s clip at {offset_s:.1f}s – {offset_s+5:.1f}s within this window</span>"
                f"<img src='{spec_b64_ctx}' style='width:100%;margin-top:6px;"
                f"border-radius:4px;display:block;'/>"
                f"</div>"
            )
            # Peak-normalize to match audio_filepath_loader's normalization
            # (raw hydrophone recordings are at very low levels; without this
            # the 30s context is much quieter than the 5-second clips)
            peak = _np_ctx.abs(audio_ctx).max()
            if peak > 1e-8:
                audio_ctx_norm = audio_ctx / peak * 0.5  # 50% of full scale
            else:
                audio_ctx_norm = audio_ctx
            audio_int16 = _np_ctx.clip(
                audio_ctx_norm * 32767, -32768, 32767).astype(_np_ctx.int16)
            return spec_html, (sr_ctx, audio_int16)
        except Exception as exc:
            return (f"<div style='color:#ef4444;font-size:11px;padding:8px;'>"
                    f"⚠ Could not load context: {exc}</div>"), None

    # Build per-segment HTML card
    def _segment_card(seg: dict, idx: int) -> str:
        wav_b64  = _make_audio_b64(seg["audio"], seg["sample_rate"])
        spec_b64 = _make_spectrogram_image(seg["audio"], seg["sample_rate"], spec_type=spectrogram_type, colormap=colormap)
        fname = seg["recording_id"].split("/")[-1]
        pid = f"player_{idx}"   # unique ID for JS targeting

        wav_b64  = _make_audio_b64(seg["audio"], seg["sample_rate"])
        player_html = f"<audio controls style='width:100%;margin-top:6px;height:40px;' src='{wav_b64}'></audio>"
        return (
            f"<div style='background:#1e293b;border-radius:8px;padding:12px;"
            f"margin-bottom:8px;color:#e2e8f0;font-family:monospace;font-size:11px;'>"
            f"<b>#{idx+1}</b> &nbsp; <span style='color:#7dd3fc'>{fname}</span>"
            f" &nbsp; <span style='color:#94a3b8'>"
            f"{seg['offset_s']:.1f}s – {seg['end_offset_s']:.1f}s</span>"
            f" &nbsp; <span style='color:#fbbf24'>score={seg['score']:.3f}</span><br>"
            f"<img src='{spec_b64}' style='width:100%;margin-top:6px;"
            f"border-radius:4px;display:block;'/>"
            f"{player_html}"
            f"</div>"
        )

    # State: labels assigned in the GUI
    label_state: dict[int, str] = {}  # window_id -> "positive"|"negative"|"unlabeled"

    # ── Determine label choices (before gr.Blocks so available in Markdown) ──
    _default_classes = ["positive", "negative", "unlabeled"]
    if label_classes:
        _choices = [c for c in label_classes if c != "unlabeled"] + ["unlabeled"]
    else:
        _choices = _default_classes

    with gr.Blocks(
        title="Perch Hoplite — Audio Labeling",
        css=(
            "body { background: #0f172a; color: #e2e8f0; font-family: 'Courier New', monospace; }"
            ".gr-button-primary { background: #0ea5e9 !important; }"
            ".gr-button { border-radius: 6px !important; }"
            ".label-radio .wrap { display: flex; flex-direction: column; gap: 6px !important; }"
            ".label-radio .wrap label { border-radius: 8px; padding: 7px 14px;"
            "  font-weight: 700; font-size: 13px; cursor: pointer;"
            "  transition: box-shadow 0.15s; }"
            ".label-radio .wrap label:nth-child(1) { background:#15803d; color:#dcfce7; }"
            ".label-radio .wrap label:nth-child(2) { background:#b45309; color:#fef3c7; }"
            ".label-radio .wrap label:nth-child(3) { background:#1d4ed8; color:#dbeafe; }"
            ".label-radio .wrap label:nth-child(4) { background:#7e22ce; color:#f3e8ff; }"
            ".label-radio .wrap label:nth-child(5) { background:#0e7490; color:#cffafe; }"
            ".label-radio .wrap label:nth-child(6) { background:#c2410c; color:#ffedd5; }"
            ".label-radio .wrap label:last-child   { background:#374151; color:#d1d5db; }"
            ".label-radio .wrap label:nth-child(1):has(input:checked)"
            "  { background:#16a34a; box-shadow:0 0 0 3px #86efac; }"
            ".label-radio .wrap label:nth-child(2):has(input:checked)"
            "  { background:#d97706; box-shadow:0 0 0 3px #fcd34d; }"
            ".label-radio .wrap label:nth-child(3):has(input:checked)"
            "  { background:#2563eb; box-shadow:0 0 0 3px #93c5fd; }"
            ".label-radio .wrap label:nth-child(4):has(input:checked)"
            "  { background:#9333ea; box-shadow:0 0 0 3px #d8b4fe; }"
            ".label-radio .wrap label:nth-child(5):has(input:checked)"
            "  { background:#0891b2; box-shadow:0 0 0 3px #67e8f9; }"
            ".label-radio .wrap label:nth-child(6):has(input:checked)"
            "  { background:#ea580c; box-shadow:0 0 0 3px #fdba74; }"
            ".label-radio .wrap label:last-child:has(input:checked)"
            "  { background:#4b5563; box-shadow:0 0 0 3px #9ca3af; }"
        ),
    ) as demo:
        _class_str = ", ".join(f"`{c}`" for c in _choices if c != "unlabeled")
        _det_info = ""
        if detections_info:
            _offset = detections_info.get("offset", 0)
            _total  = detections_info.get("total", len(segments))
            _batch_end = _offset + len(segments)
            _det_info = (
                f"**Detections:** showing {_offset + 1}–{_batch_end} of {_total} "
                f"&nbsp;&nbsp; **Batch offset:** {_offset}  \n"
            )
        gr.Markdown(
            f"""
# 🐋 Perch Hoplite — Audio Labeling Interface
**Query label:** `{query_label}` &nbsp;&nbsp; **Annotator:** `{annotator_id}`  
**Results loaded:** {len(segments)}  
{_det_info}**Label classes:** {_class_str}  
Click a label for each segment, then **Save Labels to DB**.
"""
        )

        save_btn = gr.Button("💾 Save Labels to DB", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False, lines=4)

        # ── Shared helpers ────────────────────────────────────────────────────
        import sqlite3 as _sq3g, os as _osg, struct as _stg

        def _sqlite_path():
            """Return the hoplite.sqlite file path, thread-safe."""
            p = None
            cfg = getattr(db, "db_config", None)
            if cfg: p = str(getattr(cfg, "db_path", "") or "")
            if not p:
                for a in ("db_path", "_db_path", "path"):
                    v = getattr(db, a, None)
                    if v: p = str(v); break
            if p and _osg.path.isdir(p):
                p = _osg.path.join(p, "hoplite.sqlite")
            return p

        # Accumulate per-session annotation records for provenance
        _session_annotations: dict = {}  # wid -> annotation record

        def _write_label(wid, choice):
            """Write a single label directly to SQLite (thread-safe).
            Returns True on success, False on skip, raises on error."""
            if choice == "unlabeled":
                _session_annotations.pop(int(wid), None)
                return False
            # Multi-class mode: each named class is a POSITIVE example of itself.
            # Legacy two-class mode: "positive" -> POSITIVE, anything else -> NEGATIVE.
            if label_classes:
                # Multi-class: label = choice string, type = POSITIVE
                lt = _LT.POSITIVE
            else:
                lt = _LT.POSITIVE if choice == "positive" else _LT.NEGATIVE
            dbp = _sqlite_path()
            con = _sq3g.connect(dbp)
            row = con.execute(
                "SELECT recording_id, offsets FROM windows WHERE id=?",
                (int(wid),)).fetchone()
            if row is None:
                con.close()
                raise KeyError(f"window {wid} not found")
            rec_id, off_blob = row
            if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
                start_s, end_s = _stg.unpack_from("<dd", off_blob)
            else:
                start_s, end_s = 0.0, 5.0
            off_enc = _stg.pack("<dd", start_s, end_s)
            prov = f"gradio_gui:{annotator_id}"
            # In multi-class mode the label stored is the choice itself
            # (e.g. "orca_call", "dolphin_call", "other").
            # In legacy mode the label is always query_label.
            store_label = choice if label_classes else query_label
            # DELETE only THIS ANNOTATOR's existing annotation for this window,
            # then INSERT fresh. Scoped by provenance so a second annotator ADDS an
            # opinion rather than destroying the first annotator's label. (Fixed
            # Aug 28 2026 -- the unscoped form silently overwrote other annotators,
            # which matters because the review UI is blind by default: the radio is
            # hardcoded value="unlabeled" and never reads existing labels from the DB,
            # so an annotator overwrote labels they could not even see.)
            con.execute("""
                DELETE FROM annotations
                WHERE recording_id=? AND offsets=? AND provenance=?
            """, (rec_id, off_enc, prov))
            con.execute("""
                INSERT INTO annotations
                    (recording_id, offsets, label, label_type, provenance)
                VALUES (?, ?, ?, ?, ?)
            """, (rec_id, off_enc, store_label, int(lt), prov))
            con.commit()
            # Get filename for provenance
            fname_row = con.execute(
                "SELECT filename FROM recordings WHERE id=?",
                (rec_id,)).fetchone()
            con.close()
            fname = fname_row[0] if fname_row else str(rec_id)
            # Find score for this window from segments list
            score = next((s["score"] for s in segments if s["window_id"] == int(wid)), None)
            _session_annotations[int(wid)] = {
                "window_id":  int(wid),
                "filename":   fname,
                "offset_s":   round(start_s, 3),
                "end_s":      round(end_s, 3),
                "label":      choice,
                "label_type": int(lt),
                "score":      round(score, 4) if score is not None else None,
            }
            return True

        def _label_counts():
            """Return (pos_dict, neg_dict) by direct SQLite query."""
            try:
                dbp = _sqlite_path()
                con = _sq3g.connect(dbp)
                pos = dict(con.execute(
                    "SELECT label, COUNT(*) FROM annotations WHERE label_type=? GROUP BY label",
                    (_LT.POSITIVE,)).fetchall())
                neg = dict(con.execute(
                    "SELECT label, COUNT(*) FROM annotations WHERE label_type=? GROUP BY label",
                    (_LT.NEGATIVE,)).fetchall())
                con.close()
                return pos, neg
            except Exception as exc:
                return {}, {"error": str(exc)}



        # ── Build radio buttons + wire auto-save ──────────────────────────────
        radio_components = []

        ctx_outputs = []   # (button, html_output) pairs for wiring below

        with gr.Column():
            for i, seg in enumerate(segments):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.HTML(_segment_card(seg, i))
                        # 30-second context — pre-computed at load time
                        log.info("Pre-computing 30s context for segment %d/%d...", i+1, len(segments))
                        _ctx_spec_html, _ctx_audio_data = _load_30s_context(seg)
                        ctx_spec = gr.HTML(_ctx_spec_html, visible=True)
                        ctx_audio = gr.Audio(
                            value=_ctx_audio_data if _ctx_audio_data is not None else None,
                            label="30s context", visible=True,
                            show_label=False, interactive=False)
                        ctx_btn = gr.Button(
                            "↺ Reload 30s context",
                            size="sm",
                            variant="secondary",
                            elem_classes=["ctx-btn"],
                            visible=False,  # hidden since already loaded
                        )
                        ctx_outputs.append((ctx_btn, ctx_spec, ctx_audio, seg))
                    with gr.Column(scale=1):
                        radio = gr.Radio(
                            choices=_choices,
                            value="unlabeled",
                            label=f"Label #{i+1}  (wid={seg['window_id']})",
                            elem_classes=["label-radio"],
                        )
                        radio_components.append((seg["window_id"], radio))

        # Auto-save: wire each radio to save immediately on change.
        # This means labels are written to the DB as you click,
        # so a page reload or crash never loses completed work.
        def _make_autosave(wid):
            def _autosave(choice):
                try:
                    saved = _write_label(wid, choice)
                    if saved:
                        log.info("Auto-saved: window %s -> %s", wid, choice)
                except Exception as exc:
                    log.warning("Auto-save failed for window %s: %s", wid, exc)
                return gr.update()  # no UI change needed
            return _autosave

        for wid, radio in radio_components:
            radio.change(
                fn=_make_autosave(wid),
                inputs=[radio],
                outputs=[],
            )

        # Wire 30s context buttons
        def _make_ctx_handler(s):
            def _handler():
                spec_html, audio_data = _load_30s_context(s)
                audio_update = (gr.Audio(value=audio_data, visible=True)
                                if audio_data is not None
                                else gr.Audio(visible=False))
                return gr.HTML(spec_html), audio_update
            return _handler

        for ctx_btn, ctx_spec, ctx_audio, seg in ctx_outputs:
            ctx_btn.click(
                fn=_make_ctx_handler(seg),
                inputs=[],
                outputs=[ctx_spec, ctx_audio],
            )

        def save_labels(*radio_values):
            # Batch save — writes all non-unlabeled choices to DB.
            # Even if auto-save already wrote them, ON CONFLICT DO UPDATE
            # makes this idempotent.
            saved = 0
            skipped = 0
            errors = 0
            for (wid, _), choice in zip(radio_components, radio_values):
                try:
                    if _write_label(wid, choice):
                        saved += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    log.warning("Batch save failed for window %s: %s", wid, exc)
                    errors += 1

            pos, neg = _label_counts()
            err_str = f"  ({errors} errors)" if errors else ""
            msg = (
                f"Saved {saved} labels ({skipped} unlabeled skipped){err_str}.\n"
                f"DB totals — positive: {pos}  negative: {neg}\n"
                f"Labels are also auto-saved on each click — reload is safe."
            )
            log.info(msg)
            # Write provenance record for this session
            _save_label_provenance(
                session_id=session_id,
                db_dir=_db_dir_cap,
                classifier_path=_classifier_cap,
                annotator_id=annotator_id,
                query_label=query_label,
                annotations=list(_session_annotations.values()),
            )
            return msg

        save_btn.click(
            fn=save_labels,
            inputs=[r for _, r in radio_components],
            outputs=status_box,
        )

    log.info("=" * 60)
    log.info("Launching Gradio labeling GUI")
    log.info("  Access at: http://%s:%d", host if host != "0.0.0.0" else "<server-ip>", port)
    log.info("  Press Ctrl+C to stop the server.")
    log.info("=" * 60)

    # allowed_paths is Gradio 6.x only — pass conditionally
    import gradio as _gr
    _launch_kwargs = dict(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        quiet=False,
    )
    _gr_major = int(_gr.__version__.split(".")[0])
    if _gr_major >= 5:
        _launch_kwargs["allowed_paths"] = ["/mnt/PAM_Analysis", "/mnt/PAM_Archive", "/home/duane", "/tmp"]
    demo.launch(**_launch_kwargs
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "search": cmd_search,
    "label": cmd_label,
    "train": cmd_train,
    "review": cmd_review,
    "infer": cmd_infer,
    "stats": cmd_stats,
}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_dir = Path(args.db_dir)
    log_dir = Path(args.log_dir) if args.log_dir else db_dir / "logs"
    _setup_logging(log_dir, args.verbose)

    log.info("Perch Hoplite Phase 2 — Search / Classify / Infer (PyTorch + optional TF)")
    log.info("Python %s  Command: %s", sys.version.split()[0], args.command)

    fn = COMMANDS.get(args.command)
    if fn is None:
        log.error("Unknown command: %s", args.command)
        return 1

    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
