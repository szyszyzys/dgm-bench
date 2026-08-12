#!/usr/bin/env python3
"""One-time prep for Texas-100 / Purchase-100 tabular datasets.

The loader (tabular_data_processor.py, source_type='numpy') expects
  data/tabular/texas100.npz   with arrays  features, labels
  data/tabular/purchase100.npz with arrays  features, labels
and SKIPS the network download when the file already exists. The URLs in
configs/tabular_datasets.yaml (dataset_*.npz) return HTTP 404 — the real
privacytrustlab files are .tgz in a different layout. This script downloads
those .tgz, parses them, and writes the .npz the loader wants. Run once on a
host with internet; after that all tabular cells run offline.

Label convention: source files are 1-indexed (1..100); we store 0-indexed
(0..99) so CrossEntropy targets are in range for a 100-class head.

Usage:  python prep_tabular_data.py [--data_dir <repo>/data/tabular]
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

TEXAS_URL = "https://github.com/privacytrustlab/datasets/raw/master/dataset_texas.tgz"
PURCHASE_URL = "https://github.com/privacytrustlab/datasets/raw/master/dataset_purchase.tgz"


def _download(url):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _to_zero_indexed(labels):
    labels = labels.astype(np.int64)
    lo = labels.min()
    if lo >= 1:
        labels = labels - 1
    return labels


def prep_texas(data_dir):
    out = data_dir / "texas100.npz"
    if out.exists():
        print(f"  texas100.npz already present — skipping ({out})")
        return out
    raw = _download(TEXAS_URL)
    feats = labels = None
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        # dataset uses the 100-procedure split: texas/100/feats + texas/100/labels
        f_m = tf.extractfile("texas/100/feats")
        l_m = tf.extractfile("texas/100/labels")
        feats = np.loadtxt(f_m, delimiter=",", dtype=np.float32)
        labels = np.loadtxt(l_m, dtype=np.int64)
    labels = _to_zero_indexed(labels)
    assert feats.shape[0] == labels.shape[0], (feats.shape, labels.shape)
    print(f"  texas: features {feats.shape}, labels {labels.shape}, "
          f"classes {labels.min()}..{labels.max()}")
    np.savez_compressed(out, features=feats, labels=labels)
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def prep_purchase(data_dir):
    out = data_dir / "purchase100.npz"
    if out.exists():
        print(f"  purchase100.npz already present — skipping ({out})")
        return out
    raw = _download(PURCHASE_URL)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        m = tf.extractfile("dataset_purchase")
        arr = np.loadtxt(m, delimiter=",", dtype=np.int64)
    labels = _to_zero_indexed(arr[:, 0])      # col 0 = label (1..100)
    feats = arr[:, 1:].astype(np.float32)     # cols 1.. = 600 binary features
    assert feats.shape[1] == 600, feats.shape
    print(f"  purchase: features {feats.shape}, labels {labels.shape}, "
          f"classes {labels.min()}..{labels.max()}")
    np.savez_compressed(out, features=feats, labels=labels)
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main():
    parents = Path(__file__).resolve().parents
    default_dir = (parents[2] / "data" / "tabular") if len(parents) > 2 else Path.cwd() / "data" / "tabular"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(default_dir))
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print("staging tabular datasets into", data_dir)
    prep_texas(data_dir)
    prep_purchase(data_dir)
    # sanity: reload as the app does
    for name in ("texas100", "purchase100"):
        d = np.load(data_dir / f"{name}.npz")
        print(f"  verify {name}: keys={list(d.keys())} "
              f"feat={d['features'].shape} lab={d['labels'].shape} "
              f"uniq_labels={len(np.unique(d['labels']))}")
    print("done.")


if __name__ == "__main__":
    main()
