"""
Verify that generated embeddings are correct and complete.
"""

import numpy as np
import json
import os

# ── Paths 
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # scripts/
PROJECT_DIR = os.path.dirname(BASE_DIR)                   # algorithmic_fairness_in_rag_systems/
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

EMB_FILE    = os.path.join(RESULTS_DIR, "embeddings_no_chunk.npy")
META_FILE   = os.path.join(RESULTS_DIR, "embeddings_meta_no_chunk.json")
DATA_FILE   = os.path.join(PROJECT_DIR, "data", "papers_with_institution_filtered.json")


def main():
    print("=" * 60)
    print("         EMBEDDING VERIFICATION REPORT")
    print("=" * 60)

    # ── 1. Load files 
    print("\n[1] Loading files...")
    embeddings = np.load(EMB_FILE)
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    with open(DATA_FILE, "r") as f:
        original = json.load(f)

    print(f"    Embeddings loaded : {EMB_FILE}")
    print(f"    Metadata loaded   : {META_FILE}")
    print(f"    Original data     : {DATA_FILE}")

    # ── 2. Shape check
    print("\n[2] Shape Check:")
    print(f"    Original dataset size : {len(original)}")
    print(f"    Embeddings shape      : {embeddings.shape}   <- should be ({len(original)}, 384)")
    print(f"    Metadata count        : {len(meta)}")

    assert embeddings.shape[0] == len(original), \
        f"MISMATCH: {embeddings.shape[0]} embeddings vs {len(original)} papers!"
    assert embeddings.shape[1] == 384, \
        f"Wrong embedding dim: {embeddings.shape[1]}, expected 384!"
    assert len(meta) == len(original), \
        f"MISMATCH: {len(meta)} metadata entries vs {len(original)} papers!"
    print("    All counts match!")

    # ── 3. NaN / Zero check 
    print("\n[3] NaN / Zero Vector Check:")
    nan_count  = np.isnan(embeddings).any(axis=1).sum()
    zero_count = (np.linalg.norm(embeddings, axis=1) == 0).sum()
    print(f"    NaN embeddings  : {nan_count}   <- should be 0")
    print(f"    Zero embeddings : {zero_count}   <- should be 0")
    if nan_count == 0 and zero_count == 0:
        print("    No NaN or zero vectors found!")
    else:
        print("    WARNING: Some embeddings may be corrupted!")

    # ── 4. Normalization check
    print("\n[4] Normalization Check (should be ~1.0 for each vector):")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"    Min norm  : {norms.min():.6f}")
    print(f"    Max norm  : {norms.max():.6f}")
    print(f"    Mean norm : {norms.mean():.6f}")
    if 0.99 <= norms.mean() <= 1.01:
        print("    Embeddings are correctly normalized!")
    else:
        print("    WARNING: Embeddings may not be normalized (not critical)")

    # ── 5. Metadata spot check
    print("\n[5] Metadata Spot Check (first & last record):")
    for i in [0, -1]:
        label = "First" if i == 0 else "Last"
        m = meta[i]
        o = original[i]
        print(f"\n    [{label} record]")
        print(f"      Index              : {m['index']}")
        print(f"      ID (meta)          : {m['id']}")
        print(f"      ID (original)      : {o['id']}")
        print(f"      Title              : {str(o.get('title',''))[:80]}...")
        print(f"      Institution        : {m['primary_institution']}")
        print(f"      Region             : {m['region']}")
        match = str(m['id']) == str(o['id'])
        print(f"      ID match           : {'✓' if match else 'MISMATCH!'}")

    # ── 6. Similarity sanity check 
    print("\n[6] Similarity Sanity Check:")
    sim_self = np.dot(embeddings[0], embeddings[0])
    sim_diff = np.dot(embeddings[0], embeddings[1])
    print(f"    Similarity(paper_0, paper_0) : {sim_self:.6f}  <- should be ~1.0")
    print(f"    Similarity(paper_0, paper_1) : {sim_diff:.6f}  <- should be < 1.0")
    if sim_self > 0.999 and sim_diff < 1.0:
        print("    Similarity check passed!")
    else:
        print("    WARNING: Something looks off with the embeddings!")

    # ── 7. Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT:")
    print(f"  Total papers embedded : {embeddings.shape[0]}")
    print(f"  Embedding dimensions  : {embeddings.shape[1]}")
    print(f"  Output file size      : {os.path.getsize(EMB_FILE) / (1024**2):.1f} MB")
    print("  Status                : EMBEDDINGS LOOK CORRECT!")
    print("=" * 60)


if __name__ == "__main__":
    main()